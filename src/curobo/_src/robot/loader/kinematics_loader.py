"""Portable URDF/configuration compiler for cuRobo-shaped kinematics metadata.

The original loader emits CUDA packed launch buffers.  This implementation
instead owns a mutable :class:`RobotCfg` and recompiles its ordinary PyTorch
metadata after topology-changing updates.  That retains the useful loading,
cache, and configuration lifecycle on CPU and Apple Metal without pretending
to expose CUDA/Warp ABI buffers.
"""

from __future__ import annotations

import copy
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.autograd.profiler as profiler

from curobo._src.curobolib.cuda_ops.kinematics import KinematicsFusedFunction
from curobo._src.geom.types import tensor_sphere
from curobo._src.robot.parser import UrdfRobotParser
from curobo._src.robot.types import (
    CSpaceParams, JointLimits, KinematicsParams, LinkParams, SelfCollisionKinematicsCfg,
)
from curobo._src.robot.types.joint_types import JointType
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.state.state_joint_ops import append_joints_to_state
from curobo._src.util.logging import log_and_raise
from curobo_metal.config.robot import CSpaceConfig, CollisionSphere, JointConfig, JointLimits as ScalarJointLimits, LinkConfig
from curobo_metal.config.loaders import load_urdf

from .kinematics_loader_cfg import KinematicsLoaderCfg


class KinematicsLoader(KinematicsLoaderCfg):
    """Compile a validated :class:`KinematicsLoaderCfg` into portable metadata.

    ``initialize_tensors`` is intentionally safe to call repeatedly.  It
    rebuilds derived values from the same value-model source while maintaining
    a stable loader object, which is useful for robot configuration workflows
    that add a tool link or mutate a collision-sphere bank between solves.
    """

    def __init__(self, config: KinematicsLoaderCfg) -> None:
        if not isinstance(config, KinematicsLoaderCfg):
            raise TypeError("config must be a KinematicsLoaderCfg")
        super().__init__(**deepcopy(config.__dict__))
        if self.urdf_path is None:
            raise ValueError("urdf_path is required")
        self.cpu_tensor_args = DeviceCfg(device="cpu", dtype=torch.float32)
        self._joint_limits: Optional[JointLimits] = None
        self._self_collision_data: Optional[SelfCollisionKinematicsCfg] = None
        self.lock_jointstate: Optional[JointState] = None
        self.non_fixed_joint_names: List[str] = []
        self._num_dof = 0
        self._robot = load_urdf(
            self.urdf_path, base_link=self.base_link, tool_frames=self.tool_frames
        )
        self._robot.device_cfg = self.device_cfg
        self._apply_config_to_robot()
        self._parser = UrdfRobotParser(
            self.urdf_path,
            load_meshes=self.load_meshes,
            mesh_root=self.asset_root_path,
            extra_links=self.extra_links,
        )
        self.initialize_tensors()

    @property
    def kinematics_config(self) -> KinematicsParams:
        """Compiled CPU/MPS kinematic metadata owned by this loader."""
        return self._kinematics_config

    @property
    def self_collision_config(self) -> SelfCollisionKinematicsCfg:
        """Cached self-collision pairs, rebuilt with loader tensor metadata."""
        if self._self_collision_data is None:
            self._self_collision_data = self._build_self_collision_config()
        return self._self_collision_data

    @property
    def kinematics_parser(self):
        """Parser for the original URDF plus configured portable extra links."""
        return self._parser

    @property
    def _portable_joint_names(self) -> List[str]:
        return self._robot.joint_names.copy()

    @property
    def _portable_num_dof(self) -> int:
        return self._num_dof

    @property
    def _portable_total_spheres(self) -> int:
        return self._kinematics_config.total_spheres

    def _apply_config_to_robot(self) -> None:
        """Apply non-URDF configuration data before creating cached tensors."""
        self._robot.tool_frames = list(self.tool_frames)
        self._robot.collision_link_names = list(self.collision_link_names or [])
        self._robot.self_collision_ignore = deepcopy(self.self_collision_ignore or {})
        self._robot.self_collision_buffer = deepcopy(self.self_collision_buffer or {})
        self._robot.metadata.update({
            "debug": deepcopy(self.debug),
            "mesh_link_names": list(self.mesh_link_names or []),
            "grasp_contact_link_names": deepcopy(self.grasp_contact_link_names),
        })
        self._set_collision_spheres()
        self._set_cspace()
        for link in self.extra_links.values():
            self._append_link(link)
        self._validate_configured_cspace()
        self.non_fixed_joint_names = [
            joint.name for joint in self._robot.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]
        self._apply_locked_joints()
        self._validate_model_references()

    def _validate_configured_cspace(self) -> None:
        names = list(self._robot.cspace.joint_names)
        locked = set(self.lock_joints or {})
        if len(names) != len(set(names)):
            raise ValueError("cspace contains duplicate joint names")
        joints = {joint.name: joint for joint in self._robot.joints}
        active = {
            joint.name for joint in self._robot.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        }
        mimic = {joint.name: joint.mimic_joint for joint in self._robot.joints if joint.mimic_joint}
        mimic_locks = sorted(locked & set(mimic))
        if mimic_locks:
            raise ValueError(
                f"mimic lock joints {mimic_locks} are invalid; lock the active joint instead"
            )
        absent = sorted(locked - set(joints) - set(names))
        if absent:
            raise ValueError(
                f"configured lock joints {absent} were not found in the kinematic tree or cspace"
            )
        non_parser_locks = sorted((locked & set(names)) - active)
        if non_parser_locks:
            raise ValueError(
                f"configured cspace lock joints {non_parser_locks} are not parser actuated joints"
            )
        cspace_only = sorted(set(names) - active - locked)
        if cspace_only:
            raise ValueError(
                f"cspace joints {cspace_only} are not active in the configured tree"
            )
        missing = sorted(active - set(names) - locked)
        if missing:
            raise ValueError(f"cspace is missing active tree joints: {missing}")

    def _set_collision_spheres(self) -> None:
        spheres: List[CollisionSphere] = []
        for link_name, rows in (self.collision_spheres or {}).items():
            for row in rows:
                spheres.append(CollisionSphere(
                    link_name, tuple(float(value) for value in row["center"]), float(row["radius"])
                ))
        self._robot.collision_spheres = spheres

    def _set_cspace(self) -> None:
        if self.cspace is None:
            return
        self._robot.metadata["cspace_configured"] = True
        source = self.cspace
        # RobotCfg deliberately stores serializable lists.  The public loader
        # configuration retains its tensor-valued CSpaceParams separately, so
        # a configuration can safely be saved or compiled on either device.
        def values(name: str):
            value = getattr(source, name, None)
            return None if value is None else torch.as_tensor(value).detach().cpu().tolist()

        self._robot.cspace = CSpaceConfig(
            joint_names=source.joint_names.copy(),
            default_joint_position=values("default_joint_position") or [],
            max_velocity=self._robot.cspace.max_velocity,
            max_acceleration=values("max_acceleration"),
            max_jerk=values("max_jerk"),
            cspace_distance_weight=values("cspace_distance_weight"),
            null_space_weight=values("null_space_weight"),
        )

    def _append_link(self, link_params: LinkParams) -> None:
        if link_params.link_name in {link.name for link in self._robot.links}:
            raise ValueError(f"link already exists: {link_params.link_name}")
        if link_params.parent_link_name is None:
            raise ValueError("extra links must provide parent_link_name")
        if link_params.parent_link_name not in {link.name for link in self._robot.links}:
            raise ValueError(f"extra link parent does not exist: {link_params.parent_link_name}")
        if link_params.joint_name in {joint.name for joint in self._robot.joints}:
            raise ValueError(f"joint already exists: {link_params.joint_name}")
        kind = self._joint_kind(link_params.joint_type)
        matrix = np.asarray(link_params.fixed_transform, dtype=float)
        rpy = self._matrix_to_rpy(matrix[:, :3])
        limits = link_params.joint_limits or [-float("inf"), float("inf")]
        if len(limits) != 2:
            raise ValueError("joint_limits must contain lower and upper values")
        velocity = max(abs(float(value)) for value in link_params.joint_velocity_limits)
        effort = max(abs(float(value)) for value in link_params.joint_effort_limit)
        self._robot.links.append(LinkConfig(
            link_params.link_name,
            float(link_params.link_mass), tuple(float(value) for value in link_params.link_com),
            tuple(float(value) for value in link_params.link_inertia),
        ))
        self._robot.joints.append(JointConfig(
            link_params.joint_name, kind, link_params.parent_link_name, link_params.link_name,
            axis=tuple(float(value) for value in (link_params.joint_axis if link_params.joint_axis is not None else (0.0, 0.0, 1.0))),
            xyz=tuple(float(value) for value in matrix[:, 3]), rpy=rpy,
            limits=ScalarJointLimits(float(limits[0]), float(limits[1]), velocity, effort),
            mimic_joint=link_params.mimic_joint_name,
            mimic_multiplier=float(link_params.joint_offset[0]),
            mimic_offset=float(link_params.joint_offset[1]),
        ))

    @staticmethod
    def _joint_kind(value: JointType) -> str:
        if value is JointType.FIXED:
            return "fixed"
        return "prismatic" if "PRISM" in value.name else "revolute"

    @staticmethod
    def _matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
        """Convert a proper 3x3 fixed transform to a URDF rpy tuple."""
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("LinkParams.fixed_transform must contain an orthonormal rotation")
        pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
        if abs(abs(pitch) - np.pi / 2) < 1e-7:
            roll, yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1])), 0.0
        else:
            roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return roll, pitch, yaw

    def _apply_locked_joints(self) -> None:
        locked = self.lock_joints or {}
        if not locked:
            self._robot.metadata["lock_joints"] = {}
            self.lock_jointstate = None
            return
        self._robot.metadata["lock_joints"] = dict(locked)
        from curobo._src.robot.kinematics.kinematics_cfg import _apply_locked_joints

        _apply_locked_joints(self._robot)
        self.lock_jointstate = JointState.from_position(
            self.device_cfg.to_device(list(locked.values())), joint_names=list(locked)
        )

    def _get_link_poses(self, q, query_link_names, kinematics_config):
        """Evaluate portable FK for loader-internal lock transform queries."""
        from curobo._src.robot.kinematics.kinematics import Kinematics

        state = Kinematics(kinematics_config).compute_kinematics(
            JointState.from_position(q, joint_names=kinematics_config.joint_names)
        )
        poses = [state.tool_poses[name] for name in query_link_names]
        return Pose(
            torch.stack([pose.position.reshape(-1, 3)[0] for pose in poses]).unsqueeze(0),
            torch.stack([pose.quaternion.reshape(-1, 4)[0] for pose in poses]).unsqueeze(0),
        )

    def _validate_model_references(self) -> None:
        link_names = {link.name for link in self._robot.links}
        unknown_tools = sorted(set(self._robot.tool_frames) - link_names)
        if unknown_tools:
            raise ValueError(f"tool_frames contain unknown links: {unknown_tools}")
        unknown_collision = sorted(set(self._robot.collision_link_names) - link_names)
        if unknown_collision:
            raise ValueError(f"collision_link_names contain unknown links: {unknown_collision}")
        sphere_links = {sphere.link_name for sphere in self._robot.collision_spheres}
        unknown_spheres = sorted(sphere_links - link_names)
        if unknown_spheres:
            raise ValueError(f"collision_spheres contain unknown links: {unknown_spheres}")
        # Production robot YAML commonly reserves entries such as
        # ``attached_object`` before an attachment is created.  Those names
        # are meaningful mutable configuration state, but do not participate
        # in the currently compiled pair bank, so preserve rather than reject
        # them here.

    def initialize_tensors(self):
        """Rebuild all cached portable metadata from the current robot model."""
        self._validate_model_references()
        self._kinematics_config = KinematicsParams(self._robot)
        self._kinematics_config.load_cspace_cfg_from_kinematics()
        self._kinematics_config.set_num_envs(self.num_envs)
        self._kinematics_config.make_contiguous()
        self._joint_limits = self._kinematics_config.joint_limits
        self._num_dof = self._kinematics_config.num_dof
        self.cspace = self._kinematics_config.cspace
        self._self_collision_data = self._build_self_collision_config()

    def _build_self_collision_config(self) -> SelfCollisionKinematicsCfg:
        params = self._kinematics_config
        if params.total_spheres == 0:
            return SelfCollisionKinematicsCfg(num_spheres=0)
        names = list(self._robot.collision_link_names) or list(
            dict.fromkeys(sphere.link_name for sphere in self._robot.collision_spheres)
        )
        sphere_names = {sphere.link_name for sphere in self._robot.collision_spheres}
        missing = sorted(sphere_names - set(names))
        if missing:
            raise ValueError(
                "collision_link_names must include every link with collision spheres: "
                f"{missing}"
            )
        link_index = {name: index for index, name in enumerate(names)}
        per_sphere = torch.tensor(
            [link_index[sphere.link_name] for sphere in self._robot.collision_spheres],
            dtype=torch.int64, device=self.device_cfg.device,
        )
        ignored = {name: [item for item in values if item in link_index]
                   for name, values in self._robot.self_collision_ignore.items() if name in link_index}
        padding = {name: value for name, value in self._robot.self_collision_buffer.items() if name in link_index}
        spheres = params.link_spheres[0]
        # Extra collision spheres use V2's negative-radius disabled sentinel.
        # The portable pair compiler correctly rejects those as physical
        # spheres, so compile only enabled rows then remap compact pairs back
        # to their original public sphere indices.
        enabled = torch.nonzero(spheres[:, 3] >= 0, as_tuple=False).flatten()
        if enabled.numel() == spheres.shape[0]:
            return SelfCollisionKinematicsCfg.create_from_link_pairs(
                names, link_index, ignored, padding, spheres, per_sphere, self.device_cfg,
            )
        active = SelfCollisionKinematicsCfg.create_from_link_pairs(
            names, link_index,
            ignored, padding, spheres.index_select(0, enabled), per_sphere.index_select(0, enabled), self.device_cfg,
        )
        active_pairs = active.collision_pairs
        pairs = None if active_pairs is None else enabled.index_select(
            0, active_pairs.reshape(-1)
        ).reshape(-1, 2)
        full_padding = torch.zeros((params.total_spheres,), **self.device_cfg.as_torch_dict())
        if active.sphere_padding is not None:
            full_padding.index_copy_(0, enabled, active.sphere_padding)
        return SelfCollisionKinematicsCfg(
            num_spheres=params.total_spheres, sphere_padding=full_padding, collision_pairs=pairs,
        )

    def add_link(self, link_params: LinkParams):
        """Add an extra link and atomically rebuild portable cached metadata."""
        if not isinstance(link_params, LinkParams):
            raise TypeError("link_params must be a LinkParams")
        self._append_link(link_params)
        self.extra_links[link_params.link_name] = link_params
        self._parser.extra_links[link_params.link_name] = link_params
        self._parser.build_link_parent()
        self.initialize_tensors()

    def add_fixed_link(
        self,
        link_name: str,
        parent_link_name: str,
        joint_name: Optional[str] = None,
        transform: Optional[Pose] = None,
    ):
        """Add a fixed link with an identity or caller-provided rigid offset."""
        if transform is None:
            matrix = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        else:
            value = transform.get_matrix()
            if value.numel() != 16:
                raise ValueError("transform must contain exactly one pose")
            matrix = value.reshape(4, 4)[:3].detach().cpu().numpy()
        self.add_link(LinkParams(
            link_name, joint_name or f"{link_name}_j_{parent_link_name}", JointType.FIXED,
            matrix, parent_link_name=parent_link_name,
        ))

    def _build_chain(self, base_link: str, other_links: List[str]) -> List[str]:
        """Return the stable union of tree paths needed by the requested links."""
        if base_link != self.base_link:
            raise ValueError("base_link must match the loader base_link")
        result: List[str] = []
        for name in other_links:
            for link in self._parser.get_chain(base_link, name):
                if link not in result:
                    result.append(link)
        return result

    def _get_mimic_joint_data(self) -> Dict[str, List[int]]:
        names = self._robot.joint_names
        return {
            joint.name: [names.index(joint.mimic_joint), joint.mimic_multiplier, joint.mimic_offset]
            for joint in self._robot.joints
            if joint.mimic_joint is not None and joint.mimic_joint in names
        }

    def get_joint_limits(self) -> JointLimits:
        """Return the cached tensor limits in the active C-space order."""
        if self._joint_limits is None:
            self._joint_limits = self._kinematics_config.joint_limits
        return self._joint_limits.clone()

    def _get_joint_position_velocity_limits(self) -> Dict[str, torch.Tensor]:
        limits = self.get_joint_limits()
        return {"position": limits.position, "velocity": limits.velocity}

    def _update_joint_limits(self) -> None:
        self._kinematics_config.load_cspace_cfg_from_kinematics()
        self._joint_limits = self._kinematics_config.joint_limits


from curobo._src.state.state_joint import JointState  # kept as pinned public re-export

KinematicsLoader.joint_names = KinematicsLoader._portable_joint_names
KinematicsLoader.num_dof = KinematicsLoader._portable_num_dof
KinematicsLoader.total_spheres = KinematicsLoader._portable_total_spheres

__all__ = ["KinematicsLoader"]
