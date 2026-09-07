"""Portable implementation of the pinned public robot model."""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.autograd.profiler as profiler

try:  # Optional geometry integration; mesh construction remains explicitly unsupported below.
    import trimesh as _trimesh
except ModuleNotFoundError:  # Keep the import-compatible public name on minimal installs.
    _trimesh = None
trimesh = _trimesh

from curobo._src.curobolib.cuda_ops.kinematics import KinematicsFusedFunction
from curobo._src.geom.types import Mesh, Sphere
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.robot.types import JointLimits, KinematicsParams, SelfCollisionKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import augment_joint_state
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.logging import log_and_raise
from curobo_metal.config.robot import _topological_links
from curobo_metal.ops.whole_body import WholeBodyModel, tree_forward_kinematics
from curobo_metal.reference.tree_kinematics import TreeRobot


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to wxyz without a singular sign branch.

    Choosing the largest quaternion component is important for poses at or
    near pi radians.  A component-wise ``copysign`` conversion is ambiguous
    when two off-diagonal differences are zero and made finite-difference FK
    discontinuous for several G1 hand and ankle frames.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must end in shape [3, 3]")
    m = matrix
    q_abs = torch.sqrt(torch.clamp(torch.stack((
        1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
        1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
        1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
        1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
    ), dim=-1), min=0.0))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0] ** 2, m[..., 2, 1] - m[..., 1, 2],
                     m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), dim=-1),
        torch.stack((m[..., 2, 1] - m[..., 1, 2], q_abs[..., 1] ** 2,
                     m[..., 1, 0] + m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0]), dim=-1),
        torch.stack((m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1],
                     q_abs[..., 2] ** 2, m[..., 2, 1] + m[..., 1, 2]), dim=-1),
        torch.stack((m[..., 1, 0] - m[..., 0, 1], m[..., 2, 0] + m[..., 0, 2],
                     m[..., 2, 1] + m[..., 1, 2], q_abs[..., 3] ** 2), dim=-1),
    ), dim=-2)
    denominator = (2 * q_abs).clamp_min(torch.finfo(m.dtype).eps)[..., None]
    choice = q_abs.argmax(dim=-1)
    gather_row = choice[..., None, None].expand(choice.shape + (1, 4))
    gather_denominator = choice[..., None, None].expand(choice.shape + (1, 1))
    result = candidates.gather(-2, gather_row).squeeze(-2)
    result = result / denominator.gather(-2, gather_denominator).squeeze(-2)
    result = result / result.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(m.dtype).eps)
    return torch.where(result[..., :1] < 0, -result, result)


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
        self._compile_model()
        self._batch = self._horizon = 0
        self._buffers: dict[str, torch.Tensor] = {}
        self.update_batch_size(1, 1, reset_buffers=True)

    def _compile_model(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Compile the current portable tree while retaining public config state."""
        config = self.config
        mapping = config.kinematics_config.robot_cfg._tree_mapping()
        # FK does not consume inertia.  Some visualization-only upstream URDF
        # tensors are rounded and fail dynamics' positive-semidefinite check.
        for link in mapping["links"]:
            link["inertial"]["inertia"] = [0.0] * 6
        self._model = WholeBodyModel(
            TreeRobot.from_dict(mapping),
            device=config.device_cfg.device if device is None else device,
            dtype=config.device_cfg.dtype if dtype is None else dtype,
        )
        self._link_names = self._model.link_names
        self._tool_indices = tuple(self._link_names.index(name) for name in config.tool_frames)
        model_names = list(self._model.joint_names)
        public_names = list(config.kinematics_config.joint_names)
        if set(model_names) != set(public_names) or len(model_names) != len(public_names):
            raise ValueError("compiled and public kinematics joint names do not match")
        self._model_input_order = tuple(public_names.index(name) for name in model_names)
        self._public_jacobian_order = tuple(model_names.index(name) for name in public_names)

    def _ensure_model_for(self, value: torch.Tensor) -> None:
        """Compile lazily for the caller's CPU/MPS device and floating dtype."""
        # ``resolve_device`` intentionally canonicalizes ``mps:0`` to ``mps``
        # because this backend has one logical Metal device.  PyTorch's
        # ``torch.device`` equality does not make those spellings equivalent,
        # though, so comparing them directly causes every MPS call to rebuild
        # the model.  Apart from being needlessly expensive, repeated MPS
        # materialization can trigger an asynchronous ``scatter: index -1``
        # failure while compiling otherwise valid link origins.
        same_device = (
            self._model.device.type == value.device.type
            and self._model.device.index in (None, 0)
            and value.device.index in (None, 0)
        )
        if not same_device or self._model.dtype != value.dtype:
            self._compile_model(value.device, value.dtype)
            self._buffers = {}

    @property
    def tool_frames(self) -> List[str]:
        return self.config.tool_frames

    @profiler.record_function("cuda_robot_model/update_batch_size")
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
                    batch, dtype=torch.int32, device=self._model.device
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
        self._ensure_model_for(joint_position)
        if idxs_env is not None:
            if idxs_env.ndim != 1 or idxs_env.shape[0] != joint_position.shape[0]:
                raise ValueError("idxs_env must have shape [batch]")
        batch, horizon, _ = joint_position.shape
        self.update_batch_size(batch, horizon)
        flat = joint_position.reshape(batch * horizon, self.dof)
        input_order = torch.tensor(
            self._model_input_order, dtype=torch.long, device=joint_position.device
        )
        fk = tree_forward_kinematics(self._model, flat.index_select(-1, input_order))
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
            public_order = torch.tensor(
                self._public_jacobian_order, dtype=torch.long, device=joint_position.device
            )
            jacobian = jacobian.index_select(-1, public_order)
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
    ) -> Union[List[Sphere], List[List[Sphere]]]:
        if q.ndim == 1:
            raise ValueError("q should be [batch_size, dof]")
        state = self._forward(q.unsqueeze(1) if q.ndim == 2 else q)
        values = state.robot_spheres.squeeze(1).detach().cpu().tolist()

        def as_sphere(index: int, sphere: list[float]) -> Sphere:
            # Pinned cuRobo exposes disabled slots with a negative-radius
            # sentinel when filter_valid=False.  The portable geometry value
            # validates physical radii at construction, so construct safely
            # and restore the public sentinel afterwards.
            radius = sphere[3]
            result = Sphere(
                name=f"curobo/robot_sphere_{index}",
                pose=[*sphere[:3], 1, 0, 0, 0],
                radius=max(radius, 0.0),
            )
            result.radius = radius
            return result

        result = []
        for batch in values:
            result.append(
                [
                    as_sphere(index, sphere)
                    for index, sphere in enumerate(batch)
                    if not filter_valid or sphere[3] > 0
                ]
            )
        return result

    def get_link_poses(self, joint_position: torch.Tensor, query_link_names: List[str]) -> Pose:
        unknown = set(query_link_names) - set(self._link_names)
        if unknown:
            raise ValueError(f"unknown robot links: {sorted(unknown)}")
        if joint_position.ndim == 1:
            q = joint_position.reshape(1, 1, -1)
        elif joint_position.ndim == 2:
            q = joint_position.unsqueeze(1)
        elif joint_position.ndim == 3:
            q = joint_position
        else:
            raise ValueError("joint_position must have rank 1, 2, or 3")
        if q.shape[-1] != self.dof:
            raise ValueError(f"q should have dof = {self.dof}, got {q.shape[-1]}")
        self._ensure_model_for(q)
        input_order = torch.tensor(
            self._model_input_order, dtype=torch.long, device=q.device
        )
        transforms = tree_forward_kinematics(
            self._model, q.reshape(-1, self.dof).index_select(-1, input_order)
        ).transforms.reshape(*q.shape[:2], len(self._link_names), 4, 4)
        indices = torch.tensor(
            [self._link_names.index(name) for name in query_link_names],
            device=joint_position.device,
        )
        selected = transforms.index_select(-3, indices)
        return Pose(
            selected[..., :3, 3].squeeze(1),
            _matrix_to_quaternion(selected[..., :3, :3]).squeeze(1),
        )

    @property
    def all_articulated_joint_names(self) -> List[str]:
        return self.config.kinematics_config.non_fixed_joint_names

    def get_self_collision_config(self) -> SelfCollisionKinematicsCfg:
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

    def get_full_js(self, joint_state: JointState) -> JointState:
        """Expand an active state with its configured locked joint values."""
        if not isinstance(joint_state, JointState):
            raise TypeError("joint_state must be a JointState")
        if joint_state.joint_names is None:
            joint_state = JointState(
                position=joint_state.position,
                velocity=joint_state.velocity,
                acceleration=joint_state.acceleration,
                jerk=joint_state.jerk,
                joint_names=list(self.joint_names),
            )
        else:
            joint_state = joint_state.reorder(self.joint_names)
        locked = self.lock_jointstate
        if locked is None or not locked.joint_names:
            return joint_state
        if locked.device_cfg != joint_state.device_cfg:
            locked = locked.to(joint_state.device_cfg)
        return joint_state.append_joints(locked)

    def get_mimic_js(self, joint_state: JointState) -> JointState:
        """Expose the compiled state for chains whose mimic joints are reduced."""
        mimic_joints = self.config.kinematics_config.mimic_joints
        if not mimic_joints:
            return None
        result = None
        full_state = self.get_full_js(joint_state)
        for name, (source, multiplier, offset) in mimic_joints.items():
            source_state = full_state.reorder([source])
            mimic = JointState.from_position(
                source_state.position * multiplier + offset,
                joint_names=[name],
            )
            for channel in ("velocity", "acceleration", "jerk"):
                value = getattr(source_state, channel)
                if value is not None:
                    setattr(mimic, channel, value * multiplier)
            result = mimic if result is None else result.append_joints(mimic)
        return result

    def update_kinematics_config(self, new_kin_config: KinematicsParams):
        """Update the model parameters and recompile its portable tree.

        cuRobo accepts a ``KinematicsParams`` record here.  Supporting that
        form lets callers mutate spheres/inertials or replace a reduced tree
        without depending on a CUDA-packed buffer implementation.
        """
        from curobo._src.robot.types import KinematicsParams

        if isinstance(new_kin_config, KinematicsCfg):
            self.config = new_kin_config
        elif isinstance(new_kin_config, KinematicsParams):
            self.config.kinematics_config = new_kin_config
            self.config.tool_frames = list(new_kin_config.tool_frames)
        else:
            raise TypeError("new_kin_config must be KinematicsParams or KinematicsCfg")
        self.device_cfg = self.config.device_cfg
        self._compile_model()
        self.update_batch_size(self._batch or 1, self._horizon or 1, reset_buffers=True)

    def get_link_mesh(self, link_name: str) -> Mesh:
        del link_name
        raise NotImplementedError(
            "portable Kinematics does not construct mesh assets; use RobotParser.get_link_mesh"
        )

    def get_robot_link_meshes(self) -> List[Mesh]:
        raise NotImplementedError(
            "portable Kinematics does not construct mesh assets; use RobotParser instead"
        )

    def get_robot_as_mesh(self, joint_position: torch.Tensor) -> List[Mesh]:
        del joint_position
        return self.get_robot_link_meshes()

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

    def get_joint_limits(self) -> JointLimits:
        return self.config.get_joint_limits()

    @property
    def kinematics_config(self) -> KinematicsParams:
        return self.config.kinematics_config
