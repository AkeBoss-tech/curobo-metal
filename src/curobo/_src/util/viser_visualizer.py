"""Optional Viser visualization adapter with the pinned public shape."""

from __future__ import annotations

import tempfile
from typing import Any as trimesh
from typing import Any as yourdfpy
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
try:
    import trimesh as _trimesh
except ImportError:  # Optional visualization dependency.
    _trimesh = None
else:
    trimesh = _trimesh
try:
    import yourdfpy as _yourdfpy
except ImportError:  # Optional visualization dependency.
    _yourdfpy = None
else:
    yourdfpy = _yourdfpy

from curobo._src.geom.types import SceneCfg, Sphere
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.robot.loader.util import load_robot_yaml
from curobo._src.state.state_joint import JointState
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise
from curobo._src.util_file import join_path


def _unavailable():
    try:
        import viser
    except ImportError as error:
        raise ImportError("Viser not installed. Install with: pip install viser") from error
    raise NotImplementedError(
        "the cuRobo Viser robot adapter is unavailable on the portable backend"
    )


class ViserVisualizer:
    def __init__(
        self,
        content_path: Optional[ContentPath] = None,
        device_cfg: DeviceCfg = DeviceCfg(
            device=torch.device("cuda"), dtype=torch.float32
        ),
        add_robot_to_scene: bool = False,
        connect_ip: str = "0.0.0.0",
        connect_port: int = 8080,
        initialize_viser: bool = True,
        add_control_frames: bool = True,
        visualize_robot_spheres: bool = False,
        visualize_collision_meshes: bool = False,
    ):
        del (
            content_path, device_cfg, add_robot_to_scene, connect_ip, connect_port,
            initialize_viser, add_control_frames, visualize_robot_spheres,
            visualize_collision_meshes,
        )
        _unavailable()

    @property
    def joint_names(self) -> List[str]:
        return _unavailable()

    def update_robot_spheres(self, joint_state: JointState):
        return _unavailable()

    def add_frame(self, frame_name: str, frame_pose: Pose, scale: float = 0.2) -> viser.FrameHandle:
        return _unavailable()

    def add_batched_frames(self, frame_name: str, frame_poses: Pose):
        return _unavailable()

    def add_control_frame(
        self, frame_name: str, frame_pose: Pose, scale: float = 0.2
    ) -> viser.TransformControlsHandle:
        return _unavailable()

    def reset_robot(self):
        return _unavailable()

    def set_joint_positions(
        self, joint_positions: Sequence[float], joint_names: List[str], **kwargs
    ) -> None:
        _unavailable()

    def get_control_frame_pose(self) -> Dict[str, Pose]:
        return _unavailable()

    def set_joint_state(self, joint_state: JointState) -> None:
        _unavailable()

    def add_batched_spheres(self, spheres: List[Sphere]):
        return _unavailable()

    def add_batched_spheres_from_position(
        self,
        position: np.ndarray,
        radius: np.ndarray,
        color=None,
        name: str = "curobo_spheres",
    ):
        return _unavailable()

    def add_sphere(self, sphere: Sphere):
        return _unavailable()

    def add_line_segments(self, line_segments: np.ndarray, color: np.ndarray):
        return _unavailable()

    def add_mesh(self, mesh_trimesh: trimesh.Trimesh, name: str = "mesh"):
        return _unavailable()

    def add_scene(self, scene_cfg: SceneCfg, add_control_frames: bool = False):
        return _unavailable()

    def add_point_cloud(
        self,
        pointcloud: np.ndarray,
        colors=[200, 200, 200],
        point_size: float = 0.005,
        name: str = "pointcloud",
    ):
        return _unavailable()

    def add_image(
        self,
        image: np.ndarray,
        render_width: float,
        render_height: float,
        pose: Pose,
        name: str = "image",
    ):
        return _unavailable()
