"""Robot loader that compiles URDF/config data into portable kinematics."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from curobo._src.robot.parser import UrdfRobotParser
from curobo._src.robot.types import (
    CSpaceParams, JointLimits, KinematicsParams, LinkParams, SelfCollisionKinematicsCfg,
)
from curobo._src.types.pose import Pose
from curobo._src.state.state_joint import JointState
from curobo._src.robot.types.joint_types import JointType
from curobo_metal.config.loaders import load_urdf
from curobo_metal.config.robot import JointConfig, JointLimits as ScalarJointLimits, LinkConfig

from .kinematics_loader_cfg import KinematicsLoaderCfg


class KinematicsLoader(KinematicsLoaderCfg):
    def __init__(self, config: KinematicsLoaderCfg) -> None:
        super().__init__(**config.__dict__)
        if self.urdf_path is None:
            raise ValueError("urdf_path is required")
        self._robot = load_urdf(
            self.urdf_path, base_link=self.base_link, tool_frames=self.tool_frames or None
        )
        self._parser = UrdfRobotParser(
            self.urdf_path, load_meshes=self.load_meshes,
            mesh_root=self.asset_root_path, extra_links=self.extra_links,
        )
        if self.cspace is not None:
            self._robot.cspace.joint_names = list(self.cspace.joint_names)
            if self.cspace.default_joint_position is not None:
                self._robot.cspace.default_joint_position = (
                    self.cspace.default_joint_position.detach().cpu().tolist()
                )
        if self.lock_joints:
            self._robot.metadata["lock_joints"] = dict(self.lock_joints)
        self._kinematics_config = KinematicsParams(self._robot)
        self.initialize_tensors()

    @property
    def kinematics_config(self) -> KinematicsParams:
        return self._kinematics_config

    @property
    def self_collision_config(self) -> SelfCollisionKinematicsCfg:
        params = self._kinematics_config
        if params.total_spheres == 0:
            return SelfCollisionKinematicsCfg(num_spheres=0)
        names = list(self._robot.collision_link_names)
        if not names:
            names = list(dict.fromkeys(sphere.link_name for sphere in self._robot.collision_spheres))
        link_index = {name: index for index, name in enumerate(names)}
        per_sphere = torch.tensor(
            [link_index[sphere.link_name] for sphere in self._robot.collision_spheres],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )
        return SelfCollisionKinematicsCfg.create_from_link_pairs(
            names,
            link_index,
            self._robot.self_collision_ignore,
            self._robot.self_collision_buffer,
            params.link_spheres[0],
            per_sphere,
            self.device_cfg,
        )

    @property
    def kinematics_parser(self) -> UrdfRobotParser:
        return self._parser

    def initialize_tensors(self) -> None:
        self._kinematics_config.make_contiguous()

    def add_link(self, link_params: LinkParams) -> None:
        if link_params.link_name in [link.name for link in self._robot.links]:
            raise ValueError(f"link already exists: {link_params.link_name}")
        self._robot.links.append(LinkConfig(
            link_params.link_name, link_params.link_mass,
            tuple(link_params.link_com), tuple(link_params.link_inertia),
        ))
        if link_params.parent_link_name is not None:
            kind = (
                "fixed" if link_params.joint_type.name == "FIXED"
                else "prismatic" if "PRISM" in link_params.joint_type.name else "revolute"
            )
            axis = tuple(
                [0.0, 0.0, 0.0] if link_params.joint_axis is None
                else link_params.joint_axis.tolist()
            )
            limits = link_params.joint_limits or [-float("inf"), float("inf")]
            self._robot.joints.append(JointConfig(
                link_params.joint_name, kind, link_params.parent_link_name,
                link_params.link_name, axis=axis,
                xyz=tuple(link_params.fixed_transform[:, 3]),
                limits=ScalarJointLimits(
                    limits[0], limits[1],
                    abs(link_params.joint_velocity_limits[-1]),
                    abs(link_params.joint_effort_limit[-1]),
                ),
                mimic_joint=link_params.mimic_joint_name,
                mimic_multiplier=link_params.joint_offset[0],
                mimic_offset=link_params.joint_offset[1],
            ))
        self._kinematics_config = KinematicsParams(self._robot)
        self.initialize_tensors()

    def add_fixed_link(
        self,
        link_name: str,
        parent_link_name: str,
        joint_name: Optional[str] = None,
        transform: Optional[Pose] = None,
    ) -> None:
        matrix = torch.eye(4) if transform is None else transform.get_matrix().reshape(4, 4).cpu()
        self.add_link(LinkParams(
            link_name, joint_name or f"{link_name}_joint",
            joint_type=__import__(
                "curobo._src.robot.types.joint_types", fromlist=["JointType"]
            ).JointType.FIXED,
            fixed_transform=matrix[:3].numpy(), parent_link_name=parent_link_name,
        ))

    def _build_chain(self, base_link: str, other_links: List[str]) -> List[str]:
        result = []
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
        joints = [
            joint for joint in self._robot.joints if joint.name in self._robot.joint_names
        ]
        def pair(values: List[float]) -> torch.Tensor:
            return self.device_cfg.to_device(values).T.contiguous()
        return JointLimits(
            [joint.name for joint in joints],
            pair([[j.limits.lower, j.limits.upper] for j in joints]),
            pair([[-j.limits.velocity, j.limits.velocity] for j in joints]),
            pair([[-10.0, 10.0] for _ in joints]),
            pair([[-500.0, 500.0] for _ in joints]),
            pair([[-j.limits.effort, j.limits.effort] for j in joints]),
            self.device_cfg,
        )

    def _get_joint_position_velocity_limits(self) -> Dict[str, torch.Tensor]:
        limits = self.get_joint_limits()
        return {"position": limits.position, "velocity": limits.velocity}

    def _update_joint_limits(self) -> None:
        self._kinematics_config.load_cspace_cfg_from_kinematics()


__all__ = ["KinematicsLoader"]
