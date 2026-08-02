"""Portable local mesh-SDF pose refinement.

This is deliberately a high-level replacement for V2's Warp BVH kernels: it
uses deterministic sampled mesh points and eager PyTorch ICP on CPU or MPS.
It is suitable when an initial pose is available.  Raw Warp mesh IDs,
BVH traversal, CUDA graphs, and the two-pass kernel ABI remain unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import torch

from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose, matrix_to_quaternion

from .detection_result import DetectionResult
from .mesh_robot import RobotMesh
from .pose_detector import PoseDetector
from .sdf_pose_detector_cfg import SDFDetectorCfg
from .util import extract_observed_points, resample_points


@dataclass
class SDFRefinementState:
    """Portable refinement state with V2's accepted-pose fields.

    The first three positional fields preserve the earlier compatibility
    facade.  The additional values document and retain an actual optimization
    session without pretending to own CUDA graph buffers.
    """

    position: torch.Tensor
    quaternion: torch.Tensor
    loss: torch.Tensor
    iterations: int = 0
    observed_points: Optional[torch.Tensor] = None
    best_n_valid: Optional[torch.Tensor] = None
    lambda_damping: Optional[torch.Tensor] = None
    translation_change: Optional[torch.Tensor] = None
    rotation_change: Optional[torch.Tensor] = None

    @property
    def best_position(self) -> torch.Tensor:
        return self.position

    @property
    def best_quaternion(self) -> torch.Tensor:
        return self.quaternion

    @property
    def best_error(self) -> torch.Tensor:
        return self.loss

    def clone(self) -> "SDFRefinementState":
        def clone(value):
            return None if value is None else value.clone()
        return type(self)(
            clone(self.position), clone(self.quaternion), clone(self.loss), self.iterations,
            clone(self.observed_points), clone(self.best_n_valid), clone(self.lambda_damping),
            clone(self.translation_change), clone(self.rotation_change),
        )

    def copy_(self, other: "SDFRefinementState") -> "SDFRefinementState":
        for name in (
            "position", "quaternion", "loss", "observed_points", "best_n_valid",
            "lambda_damping", "translation_change", "rotation_change",
        ):
            source = getattr(other, name)
            target = getattr(self, name)
            if source is None:
                setattr(self, name, None)
            elif target is None:
                setattr(self, name, source.clone())
            else:
                target.copy_(source)
        self.iterations = other.iterations
        return self


def _resolve_device(value: torch.device) -> torch.device:
    """Translate upstream CUDA defaults to the available portable backend."""
    if value.type == "cuda":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return value


def _rotation_angle(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.acos(cosine)


def _skew(value: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), device=value.device, dtype=value.dtype)
    x, y, z = value.unbind(-1)
    return torch.stack((
        torch.stack((zero, -z, y)), torch.stack((z, zero, -x)), torch.stack((-y, x, zero)),
    ))


def _axis_angle_matrix(omega: torch.Tensor) -> torch.Tensor:
    """Native CPU/MPS Rodrigues update, including the zero-angle series."""
    theta = torch.linalg.vector_norm(omega)
    cross = _skew(omega)
    theta2 = theta.square()
    small = theta <= torch.finfo(omega.dtype).eps
    sine_scale = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    cosine_scale = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
    return eye + sine_scale * cross + cosine_scale * (cross @ cross)


def _gauss_newton_step(
    source_world: torch.Tensor, target_world: torch.Tensor, damping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a local rigid update with ordinary tensor linear algebra.

    Avoids SVD/matrix-rank because those MPS operators are not implemented
    without fallback.  Damping makes rank-deficient symmetric meshes stable.
    """
    residual = source_world - target_world
    cross = torch.stack((
        torch.stack((torch.zeros_like(source_world[:, 0]), -source_world[:, 2], source_world[:, 1]), -1),
        torch.stack((source_world[:, 2], torch.zeros_like(source_world[:, 0]), -source_world[:, 0]), -1),
        torch.stack((-source_world[:, 1], source_world[:, 0], torch.zeros_like(source_world[:, 0])), -1),
    ), -2)
    jacobian = torch.cat((torch.eye(3, device=source_world.device, dtype=source_world.dtype).expand(
        len(source_world), -1, -1), -cross), dim=-1)
    hessian = torch.einsum("nij,nik->jk", jacobian, jacobian)
    gradient = torch.einsum("nij,ni->j", jacobian, residual)
    delta = torch.linalg.solve(
        hessian + damping * torch.eye(6, device=source_world.device, dtype=source_world.dtype), -gradient,
    )
    return delta[:3], delta[3:]


class SDFPoseDetector(PoseDetector):
    """Initial-pose local registration against a sampled :class:`RobotMesh`."""

    def __init__(self, robot_mesh: RobotMesh, config: Optional[SDFDetectorCfg] = None) -> None:
        self.robot_mesh = robot_mesh
        self.config = config or SDFDetectorCfg()
        self.device = _resolve_device(self.config.device_cfg.device)
        # ``PoseDetector`` is not initialized: its centroid detector has a
        # different config and intentionally does not implement SDF semantics.
        self.geometry = robot_mesh

    def _model_points(self, n_points: int) -> torch.Tensor:
        return self.robot_mesh.sample_surface_points(n_points)[0]

    def _initial_pose(self, initial_pose: Pose) -> Pose:
        if not isinstance(initial_pose, Pose):
            raise TypeError("initial_pose must be a curobo Pose")
        position = initial_pose.position.reshape(-1, 3)
        quaternion = initial_pose.quaternion.reshape(-1, 4)
        if len(position) != 1 or len(quaternion) != 1:
            raise ValueError("portable SDFPoseDetector supports one initial pose")
        return Pose(position.to(self.device), quaternion.to(self.device))

    def _evaluate(self, model: torch.Tensor, points: torch.Tensor, rotation: torch.Tensor,
                  translation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed = model @ rotation.mT + translation
        distances = torch.cdist(transformed, points)
        residual, nearest = distances.min(dim=-1)
        valid = residual <= self.config.distance_threshold
        return transformed, nearest, valid

    def detect(self, camera_obs: CameraObservation, config: Optional[torch.Tensor] = None,
               initial_pose: Optional[Pose] = None) -> DetectionResult:
        return self.detect_from_points(extract_observed_points(camera_obs), config, initial_pose)

    def detect_from_points(self, observed_points: torch.Tensor, config: Optional[torch.Tensor] = None,
                           initial_pose: Optional[Pose] = None) -> DetectionResult:
        """Refine ``initial_pose`` using deterministic nearest-surface ICP.

        Unlike the CUDA implementation, no global rotation sweep is attempted;
        callers must provide a local initial estimate.  A configuration updates
        articulated meshes before points are sampled.
        """
        started = time.perf_counter()
        if initial_pose is None:
            raise ValueError("SDFPoseDetector requires an initial_pose estimate")
        if config is not None:
            self.robot_mesh.update(config.to(self.device))
        if not isinstance(observed_points, torch.Tensor):
            raise TypeError("observed_points must be a torch.Tensor")
        if observed_points.ndim != 2 or observed_points.shape[-1] != 3:
            raise ValueError("observed_points must have shape [N, 3]")
        if len(observed_points) == 0:
            raise ValueError("observed_points must contain at least one point")
        points = observed_points.to(device=self.device, dtype=self.config.device_cfg.dtype)
        points = points[torch.isfinite(points).all(dim=-1)]
        if len(points) == 0:
            raise ValueError("observed_points contains no finite point")
        points = resample_points(points, min(len(points), self.config.n_points))
        model = self._model_points(min(len(points), self.config.n_points)).to(points)
        if len(model) == 0:
            raise ValueError("detector mesh contains no sampleable points")

        pose = self._initial_pose(initial_pose)
        rotation = pose.get_rotation()[0]
        translation = pose.position[0]
        minimum_valid = max(1, math.ceil(self.config.min_valid_ratio * len(model)))
        iterations = 0
        lambda_damping = points.new_tensor(self.config.lambda_initial)

        for iteration in range(self.config.max_iterations):
            transformed, nearest, valid = self._evaluate(model, points, rotation, translation)
            if int(valid.sum().item()) < minimum_valid:
                break
            delta_translation, delta_rotation_vector = _gauss_newton_step(
                transformed[valid], points[nearest[valid]], lambda_damping
            )
            candidate_rotation = _axis_angle_matrix(delta_rotation_vector) @ rotation
            candidate_translation = translation + delta_translation
            candidate_transformed, _, candidate_valid = self._evaluate(
                model, points, candidate_rotation, candidate_translation
            )
            candidate_distances = torch.cdist(candidate_transformed, points).min(dim=-1).values
            candidate_loss = candidate_distances[candidate_valid].square().mean()
            current_loss = torch.cdist(transformed, points).min(dim=-1).values[valid].square().mean()
            # Keep the accepted pose monotonic.  This is the portable analogue
            # of V2's LM trust-region acceptance without CUDA graph state.
            if candidate_loss <= current_loss:
                accepted_rotation = candidate_rotation @ rotation.mT
                accepted_translation = candidate_translation - translation
                rotation, translation = candidate_rotation, candidate_translation
                lambda_damping = (lambda_damping / self.config.lambda_factor).clamp_min(self.config.lambda_min)
            else:
                accepted_rotation = torch.eye(3, device=points.device, dtype=points.dtype)
                accepted_translation = torch.zeros(3, device=points.device, dtype=points.dtype)
                lambda_damping = (lambda_damping * self.config.lambda_factor).clamp_max(self.config.lambda_max)
            iterations = iteration + 1
            if (accepted_translation.norm() <= self.config.convergence_threshold
                    and _rotation_angle(accepted_rotation) <= self.config.rotation_convergence_threshold):
                break
        transformed, _, valid = self._evaluate(model, points, rotation, translation)
        residual = torch.cdist(transformed, points).min(dim=-1).values
        final_loss = residual[valid].square().mean() if bool(valid.any().item()) else torch.full((), torch.inf, device=points.device, dtype=points.dtype)
        pose = Pose(translation[None], matrix_to_quaternion(rotation[None]))
        valid_ratio = float(valid.float().mean().detach().cpu())
        confidence = min(1.0, valid_ratio / self.config.min_valid_ratio)
        return DetectionResult(
            pose=pose,
            config=config,
            confidence=confidence,
            alignment_error=float(final_loss.sqrt().detach().cpu()),
            n_iterations=iterations,
            compute_time=time.perf_counter() - started,
        )


__all__ = ["SDFPoseDetector", "SDFRefinementState"]
