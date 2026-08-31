"""OpenUSD writer facade; imports safely without usd-core."""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

from curobo._src.geom.types import (
    Capsule, Cuboid, Cylinder, Material, Mesh, Obstacle, SceneCfg, Sphere,
)
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise, log_info, log_warn
from curobo._src.util_file import (
    file_exists, get_assets_path, get_filename, get_files_from_dir,
    get_robot_configs_path, join_path, load_yaml,
)

from .usd_util import *
from . import usd_util as _usd_util
from .usd_util import _usd


def _require_usd():
    _usd()


def _unsupported(*args, **kwargs):
    del args, kwargs
    _require_usd()
    raise NotImplementedError("this USD operation is not implemented by the portable backend")


def join_usd_path(path1: str, path2: str) -> str:
    return _usd_util.join_usd_path(path1, path2)


def set_prim_translate(prim, translation):
    return _usd_util.set_prim_translate(prim, translation)


def set_prim_transform(
    prim, pose: List[float], scale: List[float] = [1, 1, 1], use_float: bool = False
):
    return _usd_util.set_prim_transform(prim, pose, scale, use_float)


def get_prim_world_pose(
    cache: UsdGeom.XformCache, prim: Usd.Prim, inverse: bool = False
):
    return _usd_util.get_prim_world_pose(cache, prim, inverse)


def get_transform(pose):
    return _usd_util.get_transform(pose)


def get_position_quat(pose, use_float: bool = True):
    return _usd_util.get_position_quat(pose, use_float)


def create_stage(name: str = "curobo_stage.usd", base_frame: str = "/world"):
    return _usd_util.create_stage(name, base_frame)


def set_geom_mesh_attrs(mesh_geom: UsdGeom.Mesh, obs: Mesh, timestep=None):
    return _unsupported(mesh_geom, obs, timestep)


def set_geom_cube_attrs(
    cube_geom: UsdGeom.Cube, dims: List[float], pose: List[float], timestep=None
):
    return _unsupported(cube_geom, dims, pose, timestep)


def set_geom_cylinder_attrs(
    cube_geom: UsdGeom.Cylinder, radius, height, pose: List[float], timestep=None
):
    return _unsupported(cube_geom, radius, height, pose, timestep)


def set_geom_sphere_attrs(
    sphere_geom: UsdGeom.Sphere, radius: float, pose: List[float], timestep=None
):
    return _unsupported(sphere_geom, radius, pose, timestep)


def set_cylinder_attrs(
    prim: UsdGeom.Cylinder, radius: float, height: float, pose, color=[]
):
    return _unsupported(prim, radius, height, pose, color)


def get_cylinder_attrs(prim, cache=None, transform=None) -> Cylinder:
    return _unsupported(prim, cache, transform)


def get_capsule_attrs(prim, cache=None, transform=None) -> Cylinder:
    return _unsupported(prim, cache, transform)


def get_cube_attrs(prim, cache=None, transform=None) -> Cuboid:
    return _unsupported(prim, cache, transform)


def get_sphere_attrs(prim, cache=None, transform=None) -> Sphere:
    return _unsupported(prim, cache, transform)


def get_mesh_attrs(prim, cache=None, transform=None) -> Mesh:
    return _unsupported(prim, cache, transform)


class UsdWriter:
    def __init__(self, use_float=True) -> None:
        _require_usd()
        self.use_float = use_float
        self.stage = None

    def create_stage(
        self,
        name: str = "curobo_stage.usd",
        base_frame: str = "/world",
        timesteps: Optional[int] = None,
        dt=0.02,
        interpolation_steps: float = 1,
    ):
        del timesteps, dt, interpolation_steps
        self.stage = create_stage(name, base_frame)
        return self.stage

    def add_subroot(self, root="/world", sub_root="/obstacles", pose: Optional[Pose] = None):
        return _unsupported(root, sub_root, pose)

    def load_stage_from_file(self, file_path: str):
        _, Usd, _ = _usd()
        self.stage = Usd.Stage.Open(file_path)
        return self.stage

    def load_stage(self, stage: Usd.Stage):
        self.stage = stage
        return self

    def get_pose(
        self, prim_path: str, timecode: float = 0.0, inverse: bool = False
    ) -> np.matrix:
        return _unsupported(prim_path, timecode, inverse)

    def get_obstacles_from_stage(
        self,
        only_paths: Optional[List[str]] = None,
        ignore_paths: Optional[List[str]] = None,
        only_substring: Optional[List[str]] = None,
        ignore_substring: Optional[List[str]] = None,
        reference_prim_path: Optional[str] = None,
        timecode: float = 0,
    ) -> SceneCfg:
        return _unsupported(
            only_paths, ignore_paths, only_substring, ignore_substring,
            reference_prim_path, timecode,
        )

    def add_world_to_stage(
        self,
        obstacles: SceneCfg,
        base_frame: str = "/world",
        obstacles_frame: str = "obstacles",
        base_t_obstacle_pose: Optional[Pose] = None,
        timestep: Optional[float] = None,
    ):
        return _unsupported(
            obstacles, base_frame, obstacles_frame, base_t_obstacle_pose, timestep
        )

    def get_prim_from_obstacle(
        self, obstacle: Obstacle, base_frame: str = "/world/obstacles", timestep=None
    ):
        return _unsupported(obstacle, base_frame, timestep)

    def add_cuboid_to_stage(
        self, obstacle: Cuboid, base_frame: str = "/world/obstacles",
        timestep=None, enable_physics: bool = False,
    ):
        return _unsupported(obstacle, base_frame, timestep, enable_physics)

    def add_cylinder_to_stage(
        self, obstacle: Cylinder, base_frame: str = "/world/obstacles",
        timestep=None, enable_physics: bool = False,
    ):
        return _unsupported(obstacle, base_frame, timestep, enable_physics)

    def add_sphere_to_stage(
        self, obstacle: Sphere, base_frame: str = "/world/obstacles",
        timestep=None, enable_physics: bool = False,
    ):
        return _unsupported(obstacle, base_frame, timestep, enable_physics)

    def add_mesh_to_stage(
        self, obstacle: Mesh, base_frame: str = "/world/obstacles",
        timestep=None, enable_physics: bool = False,
    ):
        return _unsupported(obstacle, base_frame, timestep, enable_physics)

    def get_obstacle_from_prim(self, prim_path: str) -> Obstacle:
        return _unsupported(prim_path)

    def write_stage_to_file(self, file_path: str, flatten: bool = False):
        if self.stage is None:
            raise RuntimeError("no USD stage loaded")
        self.stage.Flatten().Export(file_path) if flatten else self.stage.Export(file_path)

    def create_animation(
        self, robot_world_cfg: SceneCfg, pose: Pose, base_frame="/world",
        robot_frame="/robot", dt: float = 1.0,
    ):
        return _unsupported(robot_world_cfg, pose, base_frame, robot_frame, dt)

    def create_obstacle_animation(
        self, obstacles: List[List[Obstacle]], base_frame: str = "/world",
        obstacles_frame: str = "robot_base",
    ):
        return _unsupported(obstacles, base_frame, obstacles_frame)

    def create_linkpose_robot_animation(
        self, robot_usd_path: str, tool_frames: List[str], joint_names: List[str],
        pose: Pose, robot_base_frame="/world/robot", local_asset_path="assets/",
        write_robot_usd_path="assets/", robot_asset_prim_path="/panda",
    ):
        return _unsupported(
            robot_usd_path, tool_frames, joint_names, pose, robot_base_frame,
            local_asset_path, write_robot_usd_path, robot_asset_prim_path,
        )

    def add_material(
        self, material_name: str, object_path: str, color: List[float],
        obj_prim: Usd.Prim, material: Material = Material(),
    ):
        return _unsupported(material_name, object_path, color, obj_prim, material)

    def save(self):
        if self.stage is None:
            raise RuntimeError("no USD stage loaded")
        return self.stage.Save()

    @staticmethod
    def write_trajectory_animation(
        robot_model_file: Optional[str], scene_model: SceneCfg, q_start: JointState,
        q_traj: JointState, dt: float = 0.02, save_path: str = "out.usd",
        device_cfg: DeviceCfg = DeviceCfg(), interpolation_steps: float = 1.0,
        robot_base_frame="robot", base_frame="/world",
        kin_model: Optional[Kinematics] = None, visualize_robot_spheres: bool = True,
        robot_color: Optional[List[float]] = None, flatten_usd: bool = False,
        goal_pose: Optional[Union[Pose, GoalToolPose]] = None,
        goal_color: Optional[List[float]] = None,
    ):
        return _unsupported(
            robot_model_file, scene_model, q_start, q_traj, dt, save_path, device_cfg,
            interpolation_steps, robot_base_frame, base_frame, kin_model,
            visualize_robot_spheres, robot_color, flatten_usd, goal_pose, goal_color,
        )

    @staticmethod
    def load_robot(
        robot_model_file: str, device_cfg: DeviceCfg = DeviceCfg()
    ) -> Kinematics:
        return _unsupported(robot_model_file, device_cfg)

    @staticmethod
    def write_trajectory_animation_with_robot_usd(
        robot_model_file: str, scene_model: Union[SceneCfg, None], q_start: JointState,
        q_traj: JointState, dt: float = 0.02, save_path: str = "out.usd",
        device_cfg: DeviceCfg = DeviceCfg(), interpolation_steps: float = 1.0,
        write_robot_usd_path: str = "assets/", robot_base_frame: str = "robot",
        robot_usd_local_reference: str = "assets/", base_frame="/world",
        kin_model: Optional[Kinematics] = None, visualize_robot_spheres: bool = True,
        robot_asset_prim_path=None, robot_color: Optional[List[float]] = None,
        flatten_usd: bool = False,
        goal_pose: Optional[Union[Pose, GoalToolPose]] = None,
        goal_color: Optional[List[float]] = None,
        robot_usd_path: Optional[str] = None,
    ):
        return _unsupported(
            robot_model_file, scene_model, q_start, q_traj, dt, save_path, device_cfg,
            interpolation_steps, write_robot_usd_path, robot_base_frame,
            robot_usd_local_reference, base_frame, kin_model, visualize_robot_spheres,
            robot_asset_prim_path, robot_color, flatten_usd, goal_pose, goal_color,
            robot_usd_path,
        )

    @staticmethod
    def create_grid_usd(
        usds_path: Union[str, List[str]], save_path: str, base_frame: str,
        max_envs: int, max_timecode: float, x_space: float, y_space: float,
        x_per_row: int, local_asset_path: str, dt: float = 0.02,
        interpolation_steps: int = 1, prefix_string: Optional[str] = None,
        flatten_usd: bool = False,
    ):
        return _unsupported(
            usds_path, save_path, base_frame, max_envs, max_timecode, x_space,
            y_space, x_per_row, local_asset_path, dt, interpolation_steps,
            prefix_string, flatten_usd,
        )

    def load_robot_usd(
        self, robot_usd_path: str, tool_frames: List[str], joint_names: List[str],
        robot_base_frame="/world/robot", write_asset_path="assets/",
        local_asset_path="assets/", robot_asset_prim_path="/panda",
    ):
        return _unsupported(
            robot_usd_path, tool_frames, joint_names, robot_base_frame,
            write_asset_path, local_asset_path, robot_asset_prim_path,
        )

    def get_robot_prims(
        self, tool_frames: List[str], joint_names: List[str],
        robot_base_path: str = "/world/robot",
    ):
        return _unsupported(tool_frames, joint_names, robot_base_path)

    def update_robot_joint_state(
        self, joint_prims: List[Usd.Prim], joint_state: JointState, timestep: int
    ):
        return _unsupported(joint_prims, joint_state, timestep)


__all__=["UsdWriter","join_usd_path","set_prim_translate","set_prim_transform","get_prim_world_pose","get_transform","get_position_quat","create_stage","set_geom_mesh_attrs","set_geom_cube_attrs","set_geom_cylinder_attrs","set_geom_sphere_attrs","set_cylinder_attrs","get_cylinder_attrs","get_capsule_attrs","get_cube_attrs","get_sphere_attrs","get_mesh_attrs"]
