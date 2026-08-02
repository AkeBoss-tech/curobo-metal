"""Portable robot-depth segmentation using the configured collision spheres.

This is the CPU/MPS counterpart of cuRobo V2's sphere renderer.  It projects
depth pixels with the camera intrinsics, transforms them into the robot base
frame and removes pixels whose signed distance to a collision sphere is below
``distance_threshold``.  The implementation deliberately uses regular
PyTorch tensor operations: CUDA graph capture, Warp mesh rasterisation and
the packed CUDA sphere kernel are not part of the Metal backend.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
from torch.profiler import record_function

from curobo._src.geom.cv import get_projection_rays, project_depth_using_rays
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.types import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.logging import log_and_raise
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


def _canonical_depth(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return depth as ``[B,H,W]`` and whether the caller supplied one image."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("camera depth_image must be a torch.Tensor")
    if image.ndim == 2:
        return image.unsqueeze(0), True
    if image.ndim == 3:
        return image, False
    raise ValueError("depth_image must have shape [height, width] or [batch, height, width]")


def _expand_batch(value: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    """Broadcast a tensor's leading batch dimension without materialising it."""
    if value.shape[0] == batch:
        return value
    if value.shape[0] == 1:
        return value.expand((batch, *value.shape[1:]))
    raise ValueError(f"{name} batch must be 1 or {batch}, got {value.shape[0]}")


def _mask_image(
    image: torch.Tensor, distance: torch.Tensor, distance_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn a per-pixel signed distance into the V2 boolean mask.

    Positive depth is part of the contract: an invalid/zero depth pixel must
    never be claimed as robot geometry merely because its projected point is
    at the camera origin.
    """
    depth, squeezed = _canonical_depth(image)
    if distance.numel() != depth.numel():
        raise ValueError("distance must have one value for every depth pixel")
    signed_distance = distance.reshape_as(depth)
    mask = (depth > 0.0) & (signed_distance < float(distance_threshold))
    filtered = torch.where(mask, torch.zeros_like(depth), depth)
    return (mask[0], filtered[0]) if squeezed else (mask, filtered)


def _mask_spheres_image(
    image: torch.Tensor,
    robot_spheres: torch.Tensor,
    points: torch.Tensor,
    distance_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask a depth image from ``[B,N,3]`` points and ``[B,S,4]`` spheres.

    This broadcasting version mirrors the CUDA-compiled upstream path and is
    retained for callers that import the low-level helper.  Production uses
    :func:`_mask_spheres_image_cdist` to avoid allocating ``B*N*S*3``.
    """
    return _mask_spheres_image_cdist(
        image, robot_spheres, points, distance_threshold
    )


def _mask_spheres_image_cdist(
    image: torch.Tensor,
    robot_spheres: torch.Tensor,
    world_points: torch.Tensor,
    distance_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CPU/MPS-safe sphere segmentation without a Warp or CUDA dependency."""
    depth, squeezed = _canonical_depth(image)
    if world_points.ndim != 3 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape [batch, points, 3]")
    if robot_spheres.ndim != 3 or robot_spheres.shape[-1] != 4:
        raise ValueError("robot_spheres must have shape [batch, spheres, 4]")
    if world_points.shape[1] != depth.shape[1] * depth.shape[2]:
        raise ValueError("world_points must have one point for every depth pixel")
    if robot_spheres.device != world_points.device or depth.device != world_points.device:
        raise ValueError("depth, world_points, and robot_spheres must share a device")

    batch = max(depth.shape[0], world_points.shape[0], robot_spheres.shape[0])
    depth = _expand_batch(depth, batch, "depth")
    points = _expand_batch(world_points, batch, "world_points")
    spheres = _expand_batch(robot_spheres, batch, "robot_spheres")
    if not (torch.is_floating_point(points) and torch.is_floating_point(spheres)):
        raise TypeError("world_points and robot_spheres must be floating point tensors")

    # ``cdist`` does not have a bfloat16 CPU implementation.  Upstream's
    # bfloat16 option is an accelerator detail, so promote only the distance
    # calculation while keeping the output image in its original dtype.
    dtype = torch.promote_types(points.dtype, spheres.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    points = points.to(dtype=dtype)
    spheres = spheres.to(dtype=dtype)
    if spheres.shape[1] == 0:
        signed_distance = torch.full(
            points.shape[:2], float("inf"), dtype=dtype, device=points.device
        )
    else:
        radii = spheres[..., 3]
        valid = radii > 0.0
        # Disabled/negative-radius sphere slots are used by collision buffers.
        # They must not silently turn an origin pixel into a robot pixel.
        signed = torch.cdist(points, spheres[..., :3], p=2.0) - radii.unsqueeze(1)
        signed = signed.masked_fill(~valid.unsqueeze(1), float("inf"))
        signed_distance = signed.amin(dim=-1)
    mask, filtered = _mask_image(depth, signed_distance, distance_threshold)
    return (mask[0], filtered[0]) if squeezed and batch == 1 else (mask, filtered)


class RobotSegmenter:
    """Segment robot pixels using portable forward kinematics and spheres.

    ``use_cuda_graph`` retains source compatibility.  On CPU/MPS it selects a
    shape-aware :class:`GraphExecutor` direct executor; no CUDA graph is
    captured, and callers needing a mesh/renderer mask should use an external
    renderer rather than assuming Warp/Blox behaviour.
    """

    def __init__(
        self,
        kinematics: Kinematics,
        distance_threshold: float = 0.05,
        use_cuda_graph: bool = True,
        ops_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not isinstance(kinematics, Kinematics):
            raise TypeError("kinematics must be a curobo Kinematics instance")
        if not isinstance(distance_threshold, (int, float)) or not torch.isfinite(
            torch.tensor(float(distance_threshold))
        ):
            raise ValueError("distance_threshold must be finite")
        if distance_threshold < 0.0:
            raise ValueError("distance_threshold must be nonnegative")
        if not isinstance(ops_dtype, torch.dtype):
            raise TypeError("ops_dtype must be torch.dtype")
        self._kinematics = kinematics
        self.device_cfg = kinematics.device_cfg
        self.distance_threshold = float(distance_threshold)
        self._ops_dtype = ops_dtype
        self._projection_rays: Optional[torch.Tensor] = None
        self.ready = False
        self._graph_executor = GraphExecutor(
            self._mask_op,
            device=self.device_cfg.device,
            use_cuda_graph=use_cuda_graph,
            clone_outputs=True,
        )

    @staticmethod
    def from_robot_file(
        robot_file: Union[str, Dict],
        collision_sphere_buffer: Optional[float] = None,
        distance_threshold: float = 0.05,
        use_cuda_graph: bool = True,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> "RobotSegmenter":
        """Construct a segmenter from a portable YAML/configuration mapping."""
        if not isinstance(robot_file, (str, dict)):
            raise TypeError("robot_file must be a string path or dict")
        if collision_sphere_buffer is not None:
            if not isinstance(collision_sphere_buffer, (int, float)) or not torch.isfinite(
                torch.tensor(float(collision_sphere_buffer))
            ):
                raise ValueError("collision_sphere_buffer must be finite")
        # Keep user mappings reusable.  The CUDA implementation mutates this
        # nested mapping, but copying prevents a segmentation-only buffer from
        # unexpectedly changing a caller's collision checker configuration.
        source: Union[str, Dict] = deepcopy(robot_file) if isinstance(robot_file, dict) else robot_file
        cfg = KinematicsCfg.from_robot_yaml_file(source, device_cfg=device_cfg)
        if collision_sphere_buffer is not None:
            robot = cfg.kinematics_config.robot_cfg
            for sphere in robot.collision_spheres:
                sphere.radius = max(0.0, sphere.radius + float(collision_sphere_buffer))
            # KinematicsParams caches the packed sphere tensor.  Recreate it
            # after mutation so the configured buffer is actually observable.
            cfg.kinematics_config = KinematicsParams(robot)
        return RobotSegmenter(
            Kinematics(cfg, compute_spheres=True),
            distance_threshold=distance_threshold,
            use_cuda_graph=use_cuda_graph,
        )

    def update_camera_projection(self, camera_obs: CameraObservation) -> None:
        """Cache projection rays for the current camera intrinsics/resolution."""
        if not isinstance(camera_obs, CameraObservation):
            raise TypeError("camera_obs must be CameraObservation")
        depth, _ = _canonical_depth(camera_obs.depth_image)
        if camera_obs.intrinsics is None:
            raise ValueError("camera_obs.intrinsics is required for robot segmentation")
        intrinsics = camera_obs.intrinsics
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError("camera_obs.intrinsics must have shape [3,3] or [batch,3,3]")
        if intrinsics.device != depth.device:
            raise ValueError("camera depth_image and intrinsics must share a device")
        intrinsics = _expand_batch(intrinsics, depth.shape[0], "intrinsics")
        rays = get_projection_rays(
            depth.shape[-2], depth.shape[-1], intrinsics, camera_obs.depth_to_meter
        )
        self._projection_rays = rays
        # Expose the same computed values on the observation as the V2 camera
        # helper, without changing its ownership/device/dtype contract.
        camera_obs.projection_rays = rays
        self.ready = True

    def get_pointcloud_from_depth(self, camera_obs: CameraObservation) -> torch.Tensor:
        """Project depth to camera-frame points; pose conversion happens in masking."""
        depth, _ = _canonical_depth(camera_obs.depth_image)
        expected = (*depth.shape, 3)
        if self._projection_rays is None or self._projection_rays.shape != expected:
            self.update_camera_projection(camera_obs)
        assert self._projection_rays is not None
        rays = _expand_batch(self._projection_rays, depth.shape[0], "projection rays")
        return project_depth_using_rays(depth, rays)

    def _points_in_robot_frame(self, camera_obs: CameraObservation) -> torch.Tensor:
        points = self.get_pointcloud_from_depth(camera_obs)
        if camera_obs.pose is None:
            raise ValueError("camera_obs.pose (camera pose in robot frame) is required")
        pose = camera_obs.pose
        if pose.position is None or pose.quaternion is None:
            raise ValueError("camera_obs.pose must contain position and quaternion")
        if pose.position.device != points.device:
            raise ValueError("camera pose and depth_image must share a device")
        rotation = pose.get_rotation()
        if rotation is None:
            raise ValueError("camera pose must have a rotation")
        rotation = _expand_batch(rotation.reshape(-1, 3, 3), points.shape[0], "camera pose")
        position = _expand_batch(pose.position.reshape(-1, 3), points.shape[0], "camera pose")
        return torch.einsum("bij,bhwj->bhwi", rotation, points) + position[:, None, None, :]

    @record_function("robot_segmenter/_mask_op")
    def _mask_op(
        self, camera_obs: CameraObservation, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.ndim != 2:
            raise ValueError("active joint positions must have shape [dof] or [batch, dof]")
        if q.shape[-1] != self._kinematics.dof:
            raise ValueError(f"active joint positions must end in dof {self._kinematics.dof}")
        if not self.device_cfg.is_same_torch_device(q.device):
            raise ValueError("active joint positions must share the kinematics device")
        state = self._kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=self._kinematics.joint_names)
        )
        spheres = state.robot_spheres
        if spheres is None:
            raise ValueError("kinematics was created without collision spheres")
        points = self._points_in_robot_frame(camera_obs)
        if points.device != spheres.device:
            raise ValueError("camera observation and kinematics must share a device")
        return _mask_spheres_image_cdist(
            camera_obs.depth_image,
            spheres[:, 0],
            points.reshape(points.shape[0], -1, 3),
            self.distance_threshold,
        )

    def _call_op(
        self, camera_obs: CameraObservation, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._graph_executor(camera_obs, q)

    @record_function("robot_segmenter/get_robot_mask")
    def get_robot_mask(
        self, camera_obs: CameraObservation, joint_state: JointState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a robot mask and depth image with robot pixels zeroed.

        Like V2, the high-level entrypoint requires an explicit image batch;
        :meth:`get_robot_mask_from_active_js` also accepts one unbatched image
        for small interactive CPU/MPS workflows.
        """
        if not isinstance(camera_obs, CameraObservation):
            raise TypeError("camera_obs must be CameraObservation")
        if camera_obs.depth_image is None or camera_obs.depth_image.ndim != 3:
            log_and_raise("Send depth image as (batch, height, width)")
        if not isinstance(joint_state, JointState):
            raise TypeError("joint_state must be JointState")
        return self.get_robot_mask_from_active_js(
            camera_obs, self._kinematics.get_active_js(joint_state)
        )

    def get_robot_mask_from_active_js(
        self, camera_obs: CameraObservation, active_joint_state: JointState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment from a joint state already ordered for this kinematics model."""
        if not isinstance(active_joint_state, JointState):
            raise TypeError("active_joint_state must be JointState")
        return self._call_op(camera_obs, active_joint_state.position)

    @property
    def kinematics(self) -> Kinematics:
        return self._kinematics

    @property
    def base_link(self) -> str:
        return self._kinematics.base_link


__all__ = [
    "RobotSegmenter", "_mask_image", "_mask_spheres_image",
    "_mask_spheres_image_cdist",
]
