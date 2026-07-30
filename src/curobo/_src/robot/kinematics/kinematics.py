"""Portable implementation of the pinned public robot model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState, ToolPose
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo_metal.config.robot import _topological_links
from curobo_metal.ops.whole_body import WholeBodyModel, tree_forward_kinematics
from curobo_metal.reference.tree_kinematics import TreeRobot


@dataclass
class Sphere:
    """Robot sphere value compatible with cuRobo geometry consumers."""

    name: str
    pose: list[float]
    radius: float


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Differentiable wxyz conversion with stable branch-free normalization."""
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    values = torch.stack(
        (
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ),
        dim=-1,
    ).clamp_min(0)
    magnitudes = 0.5 * torch.sqrt(values)
    w, x, y, z = magnitudes.unbind(-1)
    x = torch.copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    y = torch.copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    z = torch.copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    result = torch.stack((w, x, y, z), dim=-1)
    return result / result.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(matrix.dtype).eps
    )


class Kinematics:
    def __init__(
        self,
        config: KinematicsCfg,
        compute_jacobian: bool = False,
        compute_spheres: bool = True,
        compute_com: bool = False,
    ):
        if not isinstance(config, KinematicsCfg):
            raise TypeError("config must be KinematicsCfg")
        self.config = config
        self.device_cfg = config.device_cfg
        self.compute_jacobian = compute_jacobian
        self.compute_spheres = compute_spheres
        self.compute_com = compute_com
        mapping = config.kinematics_config.robot_cfg._tree_mapping()
        # FK does not consume inertia.  Some visualization-only upstream URDF
        # tensors are rounded and fail dynamics' positive-semidefinite check.
        for link in mapping["links"]:
            link["inertial"]["inertia"] = [0.0] * 6
        self._model = WholeBodyModel(
            TreeRobot.from_dict(mapping),
            device=config.device_cfg.device,
            dtype=config.device_cfg.dtype,
        )
        self._link_names = self._model.link_names
        self._tool_indices = tuple(self._link_names.index(name) for name in config.tool_frames)
        self._batch = self._horizon = 0
        self._buffers: dict[str, torch.Tensor] = {}
        self.update_batch_size(1, 1, reset_buffers=True)

    @property
    def tool_frames(self) -> List[str]:
        return self.config.tool_frames

    def update_batch_size(
        self, batch: int, horizon: int, force_update: bool = False, reset_buffers: bool = False
    ):
        del force_update
        if batch <= 0 or horizon <= 0:
            raise ValueError("batch and horizon must be > 0")
        if self._batch != batch or self._horizon != horizon or reset_buffers:
            self._batch, self._horizon = batch, horizon
            self._buffers = {
                "idxs_env": torch.zeros(
                    batch, dtype=torch.int32, device=self.device_cfg.device
                )
            }

    def _forward(
        self, joint_position: torch.Tensor, idxs_env: Optional[torch.Tensor] = None
    ) -> KinematicsState:
        if not isinstance(joint_position, torch.Tensor):
            raise TypeError("joint_position must be a torch.Tensor")
        if joint_position.ndim != 3:
            raise ValueError(
                f"joint_position must be [batch, horizon, dof], got shape {joint_position.shape}"
            )
        if joint_position.shape[-1] != self.dof:
            raise ValueError(f"q should have dof = {self.dof}, got {joint_position.shape[-1]}")
        if idxs_env is not None:
            if idxs_env.ndim != 1 or idxs_env.shape[0] != joint_position.shape[0]:
                raise ValueError("idxs_env must have shape [batch]")
        batch, horizon, _ = joint_position.shape
        self.update_batch_size(batch, horizon)
        flat = joint_position.reshape(batch * horizon, self.dof)
        fk = tree_forward_kinematics(self._model, flat)
        transforms = fk.transforms.reshape(batch, horizon, len(self._link_names), 4, 4)
        selected = transforms[..., self._tool_indices, :, :]
        poses = ToolPose(
            self.tool_frames,
            selected[..., :3, 3],
            _matrix_to_quaternion(selected[..., :3, :3]),
        )
        jacobian = None
        if self.compute_jacobian:
            jacobian = fk.geometric_jacobian.reshape(
                batch, horizon, len(self._link_names), 6, self.dof
            )[..., self._tool_indices, :, :]
        spheres = self._sphere_positions(transforms, idxs_env) if self.compute_spheres else None
        com = self._center_of_mass(transforms) if self.compute_com else None
        return KinematicsState(poses, jacobian, spheres, com, None)

    def _sphere_positions(
        self, transforms: torch.Tensor, idxs_env: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        robot = self.config.kinematics_config.robot_cfg
        if not robot.collision_spheres:
            return transforms.new_empty((*transforms.shape[:2], 0, 4))
        params = self.config.kinematics_config
        indices = params.link_sphere_idx_map.to(transforms.device)
        environments = params.link_spheres.to(transforms)
        if idxs_env is None:
            env = torch.zeros(transforms.shape[0], dtype=torch.long, device=transforms.device)
        else:
            env = idxs_env.to(device=transforms.device, dtype=torch.long)
        if bool(((env < 0) | (env >= environments.shape[0])).any().item()):
            raise ValueError("sphere environment index is out of range")
        values = environments.index_select(0, env)
        local = torch.cat(
            (values[..., :3], torch.ones_like(values[..., :1])), dim=-1
        )
        selected = transforms.index_select(2, indices)
        xyz = torch.einsum("bhsij,bsj->bhsi", selected, local)[..., :3]
        radius = values[:, None, :, 3:].expand(*xyz.shape[:-1], 1)
        return torch.cat((xyz, radius), dim=-1)

    def _center_of_mass(self, transforms: torch.Tensor) -> torch.Tensor:
        mass = self._model.mass
        local = torch.cat(
            (self._model.com, torch.ones_like(self._model.com[:, :1])), dim=-1
        )
        points = torch.einsum("bhlij,lj->bhli", transforms, local)[..., :3]
        total = mass.sum()
        if bool((total <= 0).item()):
            raise ValueError("center of mass requires positive robot mass")
        xyz = (points * mass[None, None, :, None]).sum(dim=2) / total
        return torch.cat((xyz, total.expand(*xyz.shape[:-1], 1)), dim=-1)

    def compute_kinematics(
        self, joint_state: JointState, idxs_env: Optional[torch.Tensor] = None
    ) -> KinematicsState:
        if not isinstance(joint_state, JointState):
            raise TypeError("joint_state must be JointState")
        if (
            joint_state.joint_names is not None
            and list(joint_state.joint_names) != self.joint_names
        ):
            raise ValueError("Joint names do not match, reorder joints before forward kinematics")
        q = joint_state.position
        if q.ndim == 1:
            q = q.unsqueeze(0).unsqueeze(0)
        elif q.ndim == 2:
            q = q.unsqueeze(1)
        return self._forward(q, idxs_env)

    def get_robot_as_spheres(
        self, q: torch.Tensor, filter_valid: bool = True
    ) -> Union[list, List[list]]:
        if q.ndim == 1:
            raise ValueError("q should be [batch_size, dof]")
        state = self._forward(q.unsqueeze(1) if q.ndim == 2 else q)
        values = state.robot_spheres.squeeze(1).detach().cpu().tolist()
        result = []
        for batch in values:
            result.append(
                [
                    Sphere(
                        f"curobo/robot_sphere_{index}",
                        [*sphere[:3], 1, 0, 0, 0],
                        sphere[3],
                    )
                    for index, sphere in enumerate(batch)
                    if not filter_valid or sphere[3] > 0
                ]
            )
        return result

    def get_link_poses(self, joint_position: torch.Tensor, query_link_names: List[str]) -> Pose:
        unknown = set(query_link_names) - set(self.tool_frames)
        if unknown:
            raise ValueError(f"query links are not configured tool frames: {sorted(unknown)}")
        q = joint_position.unsqueeze(1) if joint_position.ndim == 2 else joint_position
        tool = self._forward(q).tool_poses
        indices = torch.tensor(
            [self.tool_frames.index(name) for name in query_link_names],
            device=joint_position.device,
        )
        return Pose(
            tool.position.index_select(-2, indices).squeeze(1),
            tool.quaternion.index_select(-2, indices).squeeze(1),
        )

    @property
    def all_articulated_joint_names(self) -> List[str]:
        return self.config.kinematics_config.non_fixed_joint_names

    def get_self_collision_config(self):
        return self.config.self_collision_config

    def get_link_transform(self, link_name: str) -> Pose:
        index = self._link_names.index(link_name)
        matrix = self._model.origins[index]
        return Pose(matrix[:3, 3], _matrix_to_quaternion(matrix[:3, :3]))

    def get_all_link_transforms(self) -> Pose:
        matrix = self._model.origins
        return Pose(matrix[:, :3, 3], _matrix_to_quaternion(matrix[:, :3, :3]))

    def get_dof(self) -> int:
        return self.dof

    @property
    def dof(self) -> int:
        return self.config.kinematics_config.num_dof

    @property
    def joint_names(self) -> List[str]:
        return self.config.kinematics_config.joint_names

    @property
    def total_spheres(self) -> int:
        return self.config.kinematics_config.total_spheres

    @property
    def lock_jointstate(self) -> JointState:
        return self.config.kinematics_config.lock_jointstate

    def get_active_js(self, full_js: JointState):
        return full_js.reorder(self.joint_names)

    @property
    def base_link(self) -> str:
        return self.config.kinematics_config.base_link

    @property
    def robot_spheres(self):
        return self.config.kinematics_config.link_spheres[0]

    @property
    def default_joint_position(self) -> torch.Tensor:
        return self.device_cfg.to_device(self.config.cspace.default_joint_position)

    @property
    def default_joint_state(self) -> JointState:
        return JointState.from_position(self.default_joint_position, joint_names=self.joint_names)

    def get_joint_limits(self):
        return self.config.get_joint_limits()

    @property
    def kinematics_config(self):
        return self.config.kinematics_config
