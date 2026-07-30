"""Deterministic kinematic-tree reduction utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import torch

from curobo._src.robot.types import CSpaceParams, JointLimits, KinematicsParams
from curobo._src.state.state_joint import JointState


class KinematicsReducer:
    @classmethod
    def reduce_dof(
        cls,
        original_config: KinematicsParams,
        desired_link_names: List[str],
        remove_collision_spheres: bool = True,
    ) -> KinematicsParams:
        robot = deepcopy(original_config.robot_cfg)
        known = {link.name for link in robot.links}
        missing = set(desired_link_names) - known
        if missing:
            raise ValueError(f"unknown desired links: {sorted(missing)}")
        parent_joint = {joint.child: joint for joint in robot.joints}
        needed = {robot.base_link}
        for name in desired_link_names:
            while name not in needed:
                needed.add(name)
                joint = parent_joint.get(name)
                if joint is None:
                    raise ValueError(f"{name!r} is not connected to base link")
                name = joint.parent
        robot.links = [link for link in robot.links if link.name in needed]
        robot.joints = [
            joint for joint in robot.joints
            if joint.parent in needed and joint.child in needed
        ]
        robot.tool_frames = list(desired_link_names)
        active = [
            joint.name for joint in robot.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]
        old = dict(zip(robot.cspace.joint_names, robot.cspace.default_joint_position))
        robot.cspace.joint_names = active
        robot.cspace.default_joint_position = [old.get(name, 0.0) for name in active]
        if remove_collision_spheres:
            robot.collision_spheres = [
                sphere for sphere in robot.collision_spheres if sphere.link_name in needed
            ]
        return KinematicsParams(robot)

    @classmethod
    def reconstruct_joint_state(
        cls,
        reduced_joint_state: JointState,
        lock_jointstate: Optional[JointState],
        target_joint_names: Optional[List[str]] = None,
    ) -> JointState:
        if reduced_joint_state.joint_names is None:
            raise ValueError("reduced_joint_state.joint_names is required")
        locked_names = [] if lock_jointstate is None or lock_jointstate.joint_names is None else lock_jointstate.joint_names
        names = (
            list(target_joint_names) if target_joint_names is not None
            else list(reduced_joint_state.joint_names) + list(locked_names)
        )
        values = []
        for name in names:
            if name in reduced_joint_state.joint_names:
                values.append(reduced_joint_state.position[..., reduced_joint_state.joint_names.index(name)])
            elif lock_jointstate is not None and name in locked_names:
                values.append(lock_jointstate.position[..., locked_names.index(name)])
            else:
                raise ValueError(f"joint {name!r} is absent from reduced and locked states")
        return JointState.from_position(torch.stack(values, dim=-1), joint_names=names)

    @staticmethod
    def _create_joint_limits_subset(
        original_limits: JointLimits, joint_names: List[str]
    ) -> JointLimits:
        indices = [original_limits.joint_names.index(name) for name in joint_names]
        return JointLimits(
            joint_names,
            original_limits.position[:, indices].clone(),
            original_limits.velocity[:, indices].clone(),
            original_limits.acceleration[:, indices].clone(),
            original_limits.jerk[:, indices].clone(),
            None if original_limits.effort is None else original_limits.effort[:, indices].clone(),
            original_limits.device_cfg,
        )

    @staticmethod
    def _create_cspace_subset(
        original_cspace: CSpaceParams, joint_names: List[str]
    ) -> CSpaceParams:
        result = original_cspace.clone()
        result.inplace_reindex(joint_names)
        return result


__all__ = ["KinematicsReducer"]
