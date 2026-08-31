"""Portable local mesh-SDF pose refinement.

The pinned CUDA implementation uses Warp BVH signed-distance kernels and
CUDA-graph-captured Levenberg--Marquardt iterations.  This module preserves its
high-level state and refinement lifecycle on CPU/MPS using deterministic mesh
surface samples and ordinary PyTorch normal equations.  It deliberately does
not claim Warp mesh-ID, BVH, CUDA graph, or signed-distance numerical parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple

import torch

from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.perception.optim_pose_lm import compute_predicted_reduction, solve_lm_step, trust_region_update
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose, matrix_to_quaternion
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_profiler_decorator
from curobo._src.util.warp import get_warp_device_stream

from .detection_result import DetectionResult
from .mesh_robot import RobotMesh
from .pose_detector import PoseDetector
from .sdf_pose_detector_cfg import SDFDetectorCfg
from .util import extract_observed_points, huber_loss, resample_points
from .wp_mesh_sdf_alignment import jacobian_reduce_kernel, mesh_surface_distance_query_kernel

wp = None


class SDFRefinementState:
    """State for one portable LM-style refinement session.

    Both the original compact portable constructor
    ``(position, quaternion, loss, ...)`` and V2's named state-buffer fields
    are accepted.  That allows callers to retain and copy a state session
    without depending on CUDA graph-owned buffers.
    """

    def _portable_init(self, *args, **kwargs) -> None:
        # Upstream positional layout is (observed_points, n_points,
        # best_position, best_quaternion, ...); compact portable layout has a
        # quaternion tensor as its second argument.
        if len(args) >= 2 and isinstance(args[1], int):
            names = (
                "observed_points", "n_points", "best_position", "best_quaternion",
                "best_error", "best_sum_sq", "best_n_valid", "best_JtJ", "best_Jtr",
                "lambda_damping", "translation_change", "rotation_change",
            )
        else:
            names = (
                "position", "quaternion", "loss", "iterations", "observed_points",
                "best_n_valid", "lambda_damping", "translation_change", "rotation_change",
            )
        if len(args) > len(names):
            raise TypeError("too many SDFRefinementState positional arguments")
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f"{name} was provided twice")
            kwargs[name] = value

        self.position = kwargs.pop("position", kwargs.pop("best_position", None))
        self.quaternion = kwargs.pop("quaternion", kwargs.pop("best_quaternion", None))
        self.loss = kwargs.pop("loss", kwargs.pop("best_error", None))
        if self.position is None or self.quaternion is None or self.loss is None:
            raise TypeError("position/quaternion/loss (or their best_* aliases) are required")
        self.iterations = int(kwargs.pop("iterations", 0))
        self.observed_points = kwargs.pop("observed_points", None)
        self.n_points = int(kwargs.pop(
            "n_points", len(self.observed_points) if self.observed_points is not None else 0
        ))
        self.best_n_valid = kwargs.pop("best_n_valid", None)
        self.best_sum_sq = kwargs.pop("best_sum_sq", None)
        self.best_JtJ = kwargs.pop("best_JtJ", None)
        self.best_Jtr = kwargs.pop("best_Jtr", None)
        self.lambda_damping = kwargs.pop("lambda_damping", None)
        self.translation_change = kwargs.pop("translation_change", None)
        self.rotation_change = kwargs.pop("rotation_change", None)
        if kwargs:
            raise TypeError(f"unexpected SDFRefinementState fields: {', '.join(sorted(kwargs))}")

    @property
    def _portable_best_position(self) -> torch.Tensor:
        return self.position

    @property
    def _portable_best_quaternion(self) -> torch.Tensor:
        return self.quaternion

    @property
    def _portable_best_error(self) -> torch.Tensor:
        return self.loss

    def clone(self) -> "SDFRefinementState":
        def copied(value):
            return None if value is None else value.clone()

        return type(self)(
            position=copied(self.position), quaternion=copied(self.quaternion), loss=copied(self.loss),
            iterations=self.iterations, observed_points=copied(self.observed_points), n_points=self.n_points,
            best_n_valid=copied(self.best_n_valid), best_sum_sq=copied(self.best_sum_sq),
            best_JtJ=copied(self.best_JtJ), best_Jtr=copied(self.best_Jtr),
            lambda_damping=copied(self.lambda_damping),
            translation_change=copied(self.translation_change), rotation_change=copied(self.rotation_change),
        )

    def copy_(self, other: "SDFRefinementState") -> "SDFRefinementState":
        for name in (
            "position", "quaternion", "loss", "observed_points", "best_n_valid", "best_sum_sq",
            "best_JtJ", "best_Jtr", "lambda_damping", "translation_change", "rotation_change",
        ):
            source, target = getattr(other, name), getattr(self, name)
            if source is None:
                setattr(self, name, None)
            elif target is None:
                setattr(self, name, source.clone())
            else:
                target.copy_(source)
        self.iterations, self.n_points = other.iterations, other.n_points
        return self


def _resolve_device(value: torch.device) -> torch.device:
    """Translate the upstream CUDA default to the available portable backend."""
    if value.type == "cuda":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return value


def _skew(value: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), device=value.device, dtype=value.dtype)
    x, y, z = value.unbind(-1)
    return torch.stack((
        torch.stack((zero, -z, y)), torch.stack((z, zero, -x)), torch.stack((-y, x, zero)),
    ))


def _axis_angle_matrix(omega: torch.Tensor) -> torch.Tensor:
    """Native CPU/MPS Rodrigues update including a stable zero-angle series."""
    theta = torch.linalg.vector_norm(omega)
    cross = _skew(omega)
    theta_sq = theta.square()
    small = theta <= torch.finfo(omega.dtype).eps
    sine_scale = torch.where(small, 1 - theta_sq / 6, torch.sin(theta) / theta)
    cosine_scale = torch.where(small, 0.5 - theta_sq / 24, (1 - torch.cos(theta)) / theta_sq)
    return (torch.eye(3, device=omega.device, dtype=omega.dtype)
            + sine_scale * cross + cosine_scale * (cross @ cross))


class SDFPoseDetector(PoseDetector):
    """Required-initial-pose local mesh registration with V2-like LM state."""

    def __init__(self, robot_mesh: RobotMesh, config: Optional[SDFDetectorCfg] = None):
        if not isinstance(robot_mesh, RobotMesh):
            raise TypeError("robot_mesh must be a RobotMesh")
        self.robot_mesh = robot_mesh
        self.config = config or SDFDetectorCfg()
        self.device = _resolve_device(self.config.device_cfg.device)
        if self.device.type not in {"cpu", "mps"}:
            raise ValueError("portable SDF pose refinement supports only CPU and MPS devices")
        # The pinned CUDA implementation checks float32 tensor inputs before
        # launching its Warp kernels.  Make the portable equivalent explicit
        # instead of letting a mixed float64 model/float32 observation fail in
        # matrix multiplication deep inside an iteration.
        if self.config.device_cfg.dtype != torch.float32:
            raise TypeError("portable SDF pose refinement supports only float32 tensors")
        self.dtype = self.config.device_cfg.dtype
        self._eye6 = torch.eye(6, device=self.device, dtype=self.dtype)
        self._last_state: Optional[SDFRefinementState] = None
        self._run_count = 0
        # PoseDetector's centroid config is intentionally not applicable here.
        self.geometry = robot_mesh

    @property
    def _portable_last_refinement_state(self) -> Optional[SDFRefinementState]:
        """Return an isolated snapshot of the most recently completed run."""
        return None if self._last_state is None else self._last_state.clone()

    @property
    def _portable_run_count(self) -> int:
        """Number of completed calls to :meth:`detect_from_points`."""
        return self._run_count

    def _portable_reset(self) -> None:
        """Discard the retained portable state snapshot without changing mesh state."""
        self._last_state = None
        self._run_count = 0

    def _model_points(self, n_points: int) -> torch.Tensor:
        return self.robot_mesh.sample_surface_points(n_points)[0]

    def _initial_pose(self, initial_pose: Pose) -> Pose:
        if not isinstance(initial_pose, Pose):
            raise TypeError("initial_pose must be a curobo Pose")
        position, quaternion = initial_pose.position.reshape(-1, 3), initial_pose.quaternion.reshape(-1, 4)
        if len(position) != 1 or len(quaternion) != 1:
            raise ValueError("portable SDFPoseDetector supports one initial pose")
        position = position.to(device=self.device, dtype=self.dtype)
        quaternion = quaternion.to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(position).all().item()) or not bool(torch.isfinite(quaternion).all().item()):
            raise ValueError("initial_pose must contain only finite values")
        if bool((torch.linalg.vector_norm(quaternion, dim=-1) <= torch.finfo(self.dtype).eps).any().item()):
            raise ValueError("initial_pose quaternion must be nonzero")
        return Pose(position, quaternion, normalize_rotation=True)

    def _evaluate(
        self, model: torch.Tensor, points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transformed = model @ rotation.mT + translation
        residual, nearest = torch.cdist(transformed, points).min(dim=-1)
        return transformed, nearest, residual <= self.config.distance_threshold

    def _evaluate_at_pose(
        self, observed_points: torch.Tensor, n_points: int, position: torch.Tensor, quaternion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Form V2-shaped LM normal equations using the portable SDF proxy.

        Warp's signed distance/gradient query is replaced by deterministic
        nearest surface samples.  The residual Jacobian is nevertheless an
        exact rigid point-to-point Jacobian, including robust Huber weights.
        """
        if n_points < 1:
            raise ValueError("n_points must be positive")
        points = observed_points.reshape(-1, 3)[:n_points]
        if len(points) != n_points:
            raise ValueError("n_points exceeds observed_points length")
        model = self._model_points(n_points).to(points)
        if len(model) == 0:
            raise ValueError("detector mesh contains no sampleable points")
        rotation = Pose(position[None], quaternion[None]).get_rotation()[0]
        transformed, nearest, valid = self._evaluate(model, points, rotation, position)
        residual = transformed - points[nearest]
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        cross = torch.stack((
            torch.stack((torch.zeros_like(transformed[:, 0]), -transformed[:, 2], transformed[:, 1]), -1),
            torch.stack((transformed[:, 2], torch.zeros_like(transformed[:, 0]), -transformed[:, 0]), -1),
            torch.stack((-transformed[:, 1], transformed[:, 0], torch.zeros_like(transformed[:, 0])), -1),
        ), -2)
        jacobian = torch.cat((torch.eye(3, device=points.device, dtype=points.dtype).expand(
            n_points, -1, -1), -cross), dim=-1)
        weights = valid.to(points.dtype)
        if self.config.use_huber:
            weights = weights * torch.where(
                residual_norm <= self.config.huber_delta, torch.ones_like(residual_norm),
                self.config.huber_delta / residual_norm.clamp_min(torch.finfo(points.dtype).eps),
            )
        jtj = torch.einsum("nai,n,naj->ij", jacobian, weights, jacobian)
        jtr = torch.einsum("nai,n,na->i", jacobian, weights, residual)
        sum_sq = ((huber_loss(residual_norm, self.config.huber_delta) * 2
                   if self.config.use_huber else residual_norm.square()) * valid).sum()
        return jtj, jtr, sum_sq, valid.sum().to(torch.int64)

    def _setup_refinement(self, observed_points: torch.Tensor, initial_pose: Pose) -> SDFRefinementState:
        """Allocate a portable refinement session in the pinned state shape."""
        if not isinstance(observed_points, torch.Tensor) or observed_points.ndim != 2 or observed_points.shape[-1] != 3:
            raise ValueError("observed_points must have shape [N, 3]")
        observed_points = observed_points.to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(observed_points).all().item()):
            raise ValueError("observed_points must contain only finite values")
        pose = self._initial_pose(initial_pose)
        points = observed_points.reshape(-1, 3)
        jtj, jtr, sum_sq, n_valid = self._evaluate_at_pose(
            points, len(points), pose.position[0], pose.quaternion[0]
        )
        error = torch.sqrt(sum_sq / n_valid.clamp_min(1).to(sum_sq.dtype))
        return SDFRefinementState(
            observed_points=points, n_points=len(points), best_position=pose.position[0],
            best_quaternion=pose.quaternion[0], best_error=error, best_sum_sq=sum_sq,
            best_n_valid=n_valid, best_JtJ=jtj, best_Jtr=jtr,
            lambda_damping=points.new_tensor(self.config.lambda_initial),
            translation_change=points.new_zeros(3), rotation_change=points.new_zeros(3),
        )

    def _refine_iteration(self, state: SDFRefinementState) -> SDFRefinementState:
        """Perform one monotonic trust-region update on CPU/MPS tensors."""
        if any(value is None for value in (state.observed_points, state.best_JtJ, state.best_Jtr,
                                            state.best_sum_sq, state.best_n_valid, state.lambda_damping)):
            raise ValueError("state lacks normal-equation buffers required for refinement")
        delta = torch.linalg.solve(state.best_JtJ + state.lambda_damping * self._eye6, -state.best_Jtr)
        delta_t, delta_r = delta[:3], delta[3:]
        current_rotation = Pose(state.position[None], state.quaternion[None]).get_rotation()[0]
        candidate_position = state.position + delta_t
        candidate_quaternion = matrix_to_quaternion((_axis_angle_matrix(delta_r) @ current_rotation)[None])[0]
        cand_jtj, cand_jtr, cand_sum_sq, cand_n_valid = self._evaluate_at_pose(
            state.observed_points, state.n_points, candidate_position, candidate_quaternion
        )
        predicted = -torch.dot(delta, state.best_Jtr) - 0.5 * torch.dot(delta, state.best_JtJ @ delta)
        actual = state.best_sum_sq - cand_sum_sq
        rho = actual / predicted.clamp_min(torch.finfo(actual.dtype).eps)
        min_valid = max(1, math.ceil(self.config.min_valid_ratio * state.n_points))
        accept = bool((cand_n_valid >= min_valid).item() and (actual > 0).item()
                      and (rho >= self.config.rho_min).item())
        if accept:
            position, quaternion, jtj, jtr, sum_sq, n_valid = (
                candidate_position, candidate_quaternion, cand_jtj, cand_jtr, cand_sum_sq, cand_n_valid
            )
            damping = (state.lambda_damping / self.config.lambda_factor).clamp_min(self.config.lambda_min)
            translation_change, rotation_change = delta_t, delta_r
        else:
            position, quaternion, jtj, jtr, sum_sq, n_valid = (
                state.position, state.quaternion, state.best_JtJ, state.best_Jtr,
                state.best_sum_sq, state.best_n_valid,
            )
            damping = (state.lambda_damping * self.config.lambda_factor).clamp_max(self.config.lambda_max)
            translation_change, rotation_change = torch.zeros_like(delta_t), torch.zeros_like(delta_r)
        return SDFRefinementState(
            observed_points=state.observed_points, n_points=state.n_points, best_position=position,
            best_quaternion=quaternion, best_error=torch.sqrt(sum_sq / n_valid.clamp_min(1).to(sum_sq.dtype)),
            best_sum_sq=sum_sq, best_n_valid=n_valid, best_JtJ=jtj, best_Jtr=jtr, lambda_damping=damping,
            translation_change=translation_change, rotation_change=rotation_change,
            iterations=state.iterations + 1,
        )

    def _refine_inner_iterations(self, state: SDFRefinementState) -> SDFRefinementState:
        """Portable eager equivalent of V2's captured inner iteration batch."""
        for _ in range(self.config.inner_iterations):
            state = self._refine_iteration(state)
        return state

    def _extract_observed_points(self, camera_obs: CameraObservation) -> torch.Tensor:
        return extract_observed_points(camera_obs)

    def detect(
        self, camera_obs: CameraObservation, config: Optional[torch.Tensor] = None,
        initial_pose: Optional[Pose] = None,
    ) -> DetectionResult:
        return self.detect_from_points(self._extract_observed_points(camera_obs), config, initial_pose)

    def detect_from_points(
        self, observed_points: torch.Tensor, config: Optional[torch.Tensor] = None,
        initial_pose: Optional[Pose] = None,
    ) -> DetectionResult:
        """Refine a required local estimate using deterministic robust samples."""
        started = time.perf_counter()
        if initial_pose is None:
            raise ValueError("SDFPoseDetector requires an initial_pose estimate")
        if config is not None:
            if not isinstance(config, torch.Tensor):
                raise TypeError("config must be a torch.Tensor when supplied")
            if not bool(torch.isfinite(config).all().item()):
                raise ValueError("config must contain only finite values")
            self.robot_mesh.update(config.to(device=self.device, dtype=self.dtype))
        if not isinstance(observed_points, torch.Tensor):
            raise TypeError("observed_points must be a torch.Tensor")
        if observed_points.ndim != 2 or observed_points.shape[-1] != 3:
            raise ValueError("observed_points must have shape [N, 3]")
        points = observed_points.to(device=self.device, dtype=self.dtype)
        points = points[torch.isfinite(points).all(dim=-1)]
        if len(points) == 0:
            raise ValueError("observed_points must contain at least one finite point")
        points = resample_points(points, min(len(points), self.config.n_points))
        if len(self._model_points(len(points))) == 0:
            raise ValueError("detector mesh contains no sampleable points")
        state = self._setup_refinement(points, initial_pose)
        remaining = self.config.max_iterations
        while remaining:
            iteration_count = min(remaining, self.config.inner_iterations)
            for _ in range(iteration_count):
                state = self._refine_iteration(state)
            remaining -= iteration_count
            if (state.translation_change.norm() <= self.config.convergence_threshold
                    and state.rotation_change.norm() <= self.config.rotation_convergence_threshold):
                break
        valid_ratio = float(state.best_n_valid.to(points.dtype).div(state.n_points).detach().cpu())
        self._last_state = state.clone()
        self._run_count += 1
        return DetectionResult(
            pose=Pose(state.position[None], state.quaternion[None]), config=config,
            confidence=min(1.0, valid_ratio / self.config.min_valid_ratio),
            alignment_error=float(state.loss.detach().cpu()), n_iterations=state.iterations,
            compute_time=time.perf_counter() - started,
        )


# Keep portable state diagnostics available dynamically without widening the
# pinned public declaration shape.
SDFRefinementState.__init__ = SDFRefinementState._portable_init
SDFRefinementState.best_position = SDFRefinementState._portable_best_position
SDFRefinementState.best_quaternion = SDFRefinementState._portable_best_quaternion
SDFRefinementState.best_error = SDFRefinementState._portable_best_error
SDFPoseDetector.last_refinement_state = SDFPoseDetector._portable_last_refinement_state
SDFPoseDetector.run_count = SDFPoseDetector._portable_run_count
SDFPoseDetector.reset = SDFPoseDetector._portable_reset


__all__ = ["SDFPoseDetector", "SDFRefinementState"]
