"""Differentiable whole-body tree kinematics and rigid-body dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from curobo_metal.backend import resolve_device, validate_tensor_device
from curobo_metal.ops.kinematics.forward import _motion
from curobo_metal.reference.tree_kinematics import TreeRobot


@dataclass(frozen=True)
class WholeBodyKinematicsResult:
    """Full-link transforms and Jacobians, resident on the input device."""

    transforms: torch.Tensor
    transform_jacobian: torch.Tensor
    geometric_jacobian: torch.Tensor
    link_names: tuple[str, ...]
    end_effector_indices: tuple[int, ...]
    input_was_batched: bool

    @property
    def end_effector_transforms(self) -> torch.Tensor:
        return self.transforms[:, self.end_effector_indices]

    @property
    def end_effector_geometric_jacobian(self) -> torch.Tensor:
        return self.geometric_jacobian[:, self.end_effector_indices]


@dataclass(frozen=True)
class InverseDynamicsResult:
    torque: torch.Tensor
    input_was_batched: bool


@dataclass(frozen=True)
class WholeBodyState:
    """Upstream-shaped joint state accepted by the dynamics facade."""

    position: torch.Tensor
    velocity: torch.Tensor | None = None
    acceleration: torch.Tensor | None = None
    joint_names: tuple[str, ...] | None = None

    @classmethod
    def from_joint_state(cls, state: object) -> "WholeBodyState":
        """Adapt the package or upstream-style JointState surface."""
        if not hasattr(state, "position"):
            raise TypeError("joint state must expose position")
        names = getattr(state, "joint_names", None)
        return cls(
            getattr(state, "position"),
            getattr(state, "velocity", None),
            getattr(state, "acceleration", None),
            None if names is None else tuple(names),
        )


@dataclass(frozen=True)
class ForwardDynamicsResult:
    acceleration: torch.Tensor
    mass_matrix: torch.Tensor
    bias: torch.Tensor
    input_was_batched: bool


@dataclass(frozen=True)
class DynamicsRolloutResult:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    torque: torch.Tensor


@dataclass(frozen=True)
class DynamicsCostConfig:
    effort_weight: float = 1.0
    limit_weight: float = 1.0


@dataclass(frozen=True)
class DynamicsCostResult:
    total: torch.Tensor
    effort: torch.Tensor
    limit: torch.Tensor


class WholeBodyModel:
    """Validated immutable tree metadata compiled for one dtype and device."""

    def __init__(
        self,
        robot: TreeRobot,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(robot, TreeRobot):
            raise TypeError("robot must be a TreeRobot")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("whole-body metadata dtype must be float32 or float64")
        resolved = resolve_device(device)
        if resolved.type == "mps" and dtype != torch.float32:
            raise TypeError("MPS whole-body operators support only float32")
        self._robot = robot
        self.device = resolved
        self.dtype = dtype
        self.dof = robot.dof
        self.link_count = len(robot.links)
        self.link_names = tuple(link.name for link in robot.links)
        self.joint_names = robot.joint_names
        self.end_effector_indices = robot.end_effectors
        self.parents = tuple(link.parent for link in robot.links)
        self.kinds = tuple(link.kind for link in robot.links)
        self.q_indices = tuple(link.q_index for link in robot.links)
        self.multipliers = tuple(link.multiplier for link in robot.links)
        self.offsets = tuple(link.offset for link in robot.links)

        def tensor(value: object) -> torch.Tensor:
            return torch.as_tensor(value, device=resolved, dtype=dtype).clone()

        self.origins = torch.stack([tensor(link.origin) for link in robot.links])
        self.axes = torch.stack([tensor(link.axis) for link in robot.links])
        self.mass = tensor([link.mass for link in robot.links])
        self.com = torch.stack([tensor(link.com) for link in robot.links])
        self.inertia = torch.stack([tensor(link.inertia) for link in robot.links])
        self.gravity = tensor(robot.gravity)
        self.effort_limits = tensor(robot.effort_limits)

    def inverse_dynamics(
        self, state: WholeBodyState, *, gravity: torch.Tensor | Sequence[float] | None = None
    ) -> InverseDynamicsResult:
        q, qd, qdd = _state_tensors(self, state, require_acceleration=True)
        return inverse_dynamics(self, q, qd, qdd, gravity=gravity)

    def forward_dynamics(
        self, state: WholeBodyState, torque: torch.Tensor
    ) -> ForwardDynamicsResult:
        q, qd, _ = _state_tensors(self, state, require_acceleration=False)
        return forward_dynamics(self, q, qd, torque)

    @classmethod
    def from_tree_robot(
        cls,
        robot: TreeRobot,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "WholeBodyModel":
        return cls(robot, device=device, dtype=dtype)

    def to(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "WholeBodyModel":
        return type(self)(
            self._robot,
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )


def _model_for(
    model: WholeBodyModel | TreeRobot, value: torch.Tensor
) -> WholeBodyModel:
    if isinstance(model, TreeRobot):
        return WholeBodyModel(model, device=value.device, dtype=value.dtype)
    if not isinstance(model, WholeBodyModel):
        raise TypeError("model must be a WholeBodyModel or TreeRobot")
    return model


def _batch(
    model: WholeBodyModel, value: torch.Tensor, name: str
) -> tuple[torch.Tensor, bool]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    validate_tensor_device(value, expected=model.device)
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64")
    if value.dtype != model.dtype:
        raise TypeError(f"{name} dtype {value.dtype} does not match model dtype {model.dtype}")
    if value.device.type == "mps" and value.dtype != torch.float32:
        raise TypeError("MPS whole-body operators support only float32")
    was_batched = value.ndim == 2
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != model.dof:
        raise ValueError(
            f"{name} must have shape [{model.dof}] or [B, {model.dof}]"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value, was_batched


def _state_tensors(
    model: WholeBodyModel, state: WholeBodyState | object, *, require_acceleration: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(state, WholeBodyState):
        state = WholeBodyState.from_joint_state(state)
    q = state.position
    if state.joint_names is not None:
        if len(state.joint_names) != model.dof or set(state.joint_names) != set(model.joint_names):
            raise ValueError("joint_names must be a permutation of model joint_names")
        order = torch.tensor(
            [state.joint_names.index(name) for name in model.joint_names],
            device=q.device,
        )
        q = q.index_select(-1, order)
        qd = None if state.velocity is None else state.velocity.index_select(-1, order)
        qdd = None if state.acceleration is None else state.acceleration.index_select(-1, order)
    else:
        qd, qdd = state.velocity, state.acceleration
    if qd is None:
        qd = torch.zeros_like(q)
    if qdd is None:
        if require_acceleration:
            raise ValueError("state.acceleration is required for inverse dynamics")
        qdd = torch.zeros_like(q)
    return q, qd, qdd


def tree_forward_kinematics(
    model: WholeBodyModel | TreeRobot, q: torch.Tensor
) -> WholeBodyKinematicsResult:
    """Evaluate all links of a parent-index tree with exact transform Jacobians."""
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    compiled = _model_for(model, q)
    configurations, was_batched = _batch(compiled, q, "q")
    batch, dof = configurations.shape
    identity = torch.eye(
        4, dtype=compiled.dtype, device=compiled.device
    ).expand(batch, 4, 4)
    zero_jacobian = configurations.new_zeros((batch, dof, 4, 4))
    transforms: list[torch.Tensor] = []
    jacobians: list[torch.Tensor] = []

    for i in range(compiled.link_count):
        parent = compiled.parents[i]
        parent_transform = identity if parent < 0 else transforms[parent]
        parent_jacobian = zero_jacobian if parent < 0 else jacobians[parent]
        q_index = compiled.q_indices[i]
        position = (
            configurations.new_zeros((batch,))
            if q_index is None
            else configurations[:, q_index] * compiled.multipliers[i]
            + compiled.offsets[i]
        )
        motion, derivative = _motion(
            compiled.kinds[i], compiled.axes[i], position
        )
        origin = compiled.origins[i].expand(batch, 4, 4)
        local = origin @ motion
        local_derivative = origin @ derivative
        link_jacobian = parent_jacobian @ local[:, None]
        if q_index is not None:
            selector = (
                torch.arange(dof, device=compiled.device) == q_index
            ).to(compiled.dtype)
            link_jacobian = link_jacobian + (
                parent_transform @ local_derivative * compiled.multipliers[i]
            )[:, None] * selector[None, :, None, None]
        transforms.append(parent_transform @ local)
        jacobians.append(link_jacobian)

    transform_tensor = torch.stack(transforms, dim=1)
    derivative_tensor = torch.stack(jacobians, dim=1).permute(0, 1, 3, 4, 2)
    rotation = transform_tensor[:, :, :3, :3]
    d_rotation = derivative_tensor[:, :, :3, :3, :]
    omega_matrix = torch.einsum("blikj,blmk->blimj", d_rotation, rotation)
    angular = torch.stack(
        (
            omega_matrix[:, :, 2, 1],
            omega_matrix[:, :, 0, 2],
            omega_matrix[:, :, 1, 0],
        ),
        dim=2,
    )
    geometric = torch.cat((derivative_tensor[:, :, :3, 3], angular), dim=2)
    return WholeBodyKinematicsResult(
        transform_tensor,
        derivative_tensor,
        geometric,
        compiled.link_names,
        compiled.end_effector_indices,
        was_batched,
    )


def _cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cross(left, right, dim=-1)


def inverse_dynamics(
    model: WholeBodyModel | TreeRobot,
    q: torch.Tensor,
    qd: torch.Tensor,
    qdd: torch.Tensor,
    *,
    gravity: torch.Tensor | Sequence[float] | None = None,
) -> InverseDynamicsResult:
    """Batched differentiable recursive Newton-Euler inverse dynamics."""
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    compiled = _model_for(model, q)
    q_b, was_batched = _batch(compiled, q, "q")
    qd_b, _ = _batch(compiled, qd, "qd")
    qdd_b, _ = _batch(compiled, qdd, "qdd")
    if q_b.shape != qd_b.shape or q_b.shape != qdd_b.shape:
        raise ValueError("q, qd, and qdd batch dimensions must match")
    if gravity is None:
        gravity_tensor = compiled.gravity
    else:
        gravity_tensor = torch.as_tensor(
            gravity, dtype=compiled.dtype, device=compiled.device
        )
    if gravity_tensor.shape != (3,) or not bool(torch.isfinite(gravity_tensor).all().item()):
        raise ValueError("gravity must have shape [3] and be finite")

    fk = tree_forward_kinematics(compiled, q_b)
    batch = q_b.shape[0]
    zeros = q_b.new_zeros((batch, 3))
    omegas: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    accelerations: list[torch.Tensor] = []
    forces: list[torch.Tensor] = []
    moments: list[torch.Tensor] = []
    axes: list[torch.Tensor] = []

    for i in range(compiled.link_count):
        parent = compiled.parents[i]
        wp = zeros if parent < 0 else omegas[parent]
        ap = zeros if parent < 0 else alphas[parent]
        vp = zeros if parent < 0 else velocities[parent]
        alp = zeros if parent < 0 else accelerations[parent]
        parent_position = (
            zeros if parent < 0 else fk.transforms[:, parent, :3, 3]
        )
        displacement = fk.transforms[:, i, :3, 3] - parent_position
        velocity = vp + _cross(wp, displacement)
        acceleration = (
            alp + _cross(ap, displacement) + _cross(wp, _cross(wp, displacement))
        )
        parent_rotation = (
            torch.eye(3, dtype=compiled.dtype, device=compiled.device)
            .expand(batch, 3, 3)
            if parent < 0
            else fk.transforms[:, parent, :3, :3]
        )
        axis = (parent_rotation @ compiled.origins[i, :3, :3]) @ compiled.axes[i]
        axes.append(axis)
        q_index = compiled.q_indices[i]
        qdi = (
            q_b.new_zeros((batch,))
            if q_index is None
            else qd_b[:, q_index] * compiled.multipliers[i]
        )
        qddi = (
            q_b.new_zeros((batch,))
            if q_index is None
            else qdd_b[:, q_index] * compiled.multipliers[i]
        )
        if compiled.kinds[i] == "revolute":
            omega = wp + axis * qdi[:, None]
            alpha = ap + axis * qddi[:, None] + _cross(wp, axis * qdi[:, None])
        else:
            omega, alpha = wp, ap
            if compiled.kinds[i] == "prismatic":
                velocity = velocity + axis * qdi[:, None]
                acceleration = (
                    acceleration
                    + axis * qddi[:, None]
                    + 2 * _cross(wp, axis * qdi[:, None])
                )
        rotation = fk.transforms[:, i, :3, :3]
        com = (rotation @ compiled.com[i]).reshape(batch, 3)
        com_acceleration = (
            acceleration
            + _cross(alpha, com)
            + _cross(omega, _cross(omega, com))
        )
        force = compiled.mass[i] * (com_acceleration - gravity_tensor)
        inertia = rotation @ compiled.inertia[i] @ rotation.transpose(-1, -2)
        angular_momentum = (inertia @ omega[..., None]).squeeze(-1)
        moment = (
            (inertia @ alpha[..., None]).squeeze(-1)
            + _cross(omega, angular_momentum)
            + _cross(com, force)
        )
        omegas.append(omega)
        alphas.append(alpha)
        velocities.append(velocity)
        accelerations.append(acceleration)
        forces.append(force)
        moments.append(moment)

    torque = q_b.new_zeros((batch, compiled.dof))
    for i in range(compiled.link_count - 1, -1, -1):
        q_index = compiled.q_indices[i]
        if q_index is not None:
            effort = (
                (axes[i] * moments[i]).sum(-1)
                if compiled.kinds[i] == "revolute"
                else (axes[i] * forces[i]).sum(-1)
            )
            contribution = effort * compiled.multipliers[i]
            selector = (
                torch.arange(compiled.dof, device=compiled.device) == q_index
            ).to(compiled.dtype)
            torque = torque + contribution[:, None] * selector
        parent = compiled.parents[i]
        if parent >= 0:
            displacement = (
                fk.transforms[:, i, :3, 3]
                - fk.transforms[:, parent, :3, 3]
            )
            child_force = forces[i]
            forces[parent] = forces[parent] + child_force
            moments[parent] = (
                moments[parent] + moments[i] + _cross(displacement, child_force)
            )
    return InverseDynamicsResult(torque, was_batched)


def mass_matrix(
    model: WholeBodyModel | TreeRobot, q: torch.Tensor
) -> torch.Tensor:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    compiled = _model_for(model, q)
    q_b, _ = _batch(compiled, q, "q")
    zeros = torch.zeros_like(q_b)
    columns = []
    for j in range(compiled.dof):
        unit = torch.nn.functional.one_hot(
            torch.tensor(j, device=compiled.device), compiled.dof
        ).to(compiled.dtype)
        unit = unit.expand(q_b.shape[0], compiled.dof)
        columns.append(
            inverse_dynamics(
                compiled, q_b, zeros, unit, gravity=q_b.new_zeros(3)
            ).torque
        )
    matrix = torch.stack(columns, dim=-1)
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def gravity_torque(
    model: WholeBodyModel | TreeRobot, q: torch.Tensor
) -> torch.Tensor:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    compiled = _model_for(model, q)
    q_b, _ = _batch(compiled, q, "q")
    zeros = torch.zeros_like(q_b)
    return inverse_dynamics(compiled, q_b, zeros, zeros).torque


def bias_torque(
    model: WholeBodyModel | TreeRobot, q: torch.Tensor, qd: torch.Tensor
) -> torch.Tensor:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    compiled = _model_for(model, q)
    q_b, _ = _batch(compiled, q, "q")
    return inverse_dynamics(compiled, q_b, qd, torch.zeros_like(q_b)).torque


def forward_dynamics(
    model: WholeBodyModel | TreeRobot,
    q: torch.Tensor,
    qd: torch.Tensor,
    torque: torch.Tensor,
) -> ForwardDynamicsResult:
    """Solve ``M(q) qdd + bias(q, qd) = torque`` on CPU or MPS."""
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    compiled = _model_for(model, q)
    q_b, was_batched = _batch(compiled, q, "q")
    qd_b, _ = _batch(compiled, qd, "qd")
    torque_b, _ = _batch(compiled, torque, "torque")
    if q_b.shape != qd_b.shape or q_b.shape != torque_b.shape:
        raise ValueError("q, qd, and torque batch dimensions must match")
    matrix = mass_matrix(compiled, q_b)
    bias = bias_torque(compiled, q_b, qd_b)
    acceleration = torch.linalg.solve(matrix, (torque_b - bias).unsqueeze(-1)).squeeze(-1)
    return ForwardDynamicsResult(acceleration, matrix, bias, was_batched)


def rollout_dynamics(
    model: WholeBodyModel | TreeRobot,
    initial: WholeBodyState,
    torque: torch.Tensor,
    timestep: float | torch.Tensor,
) -> DynamicsRolloutResult:
    """Deterministic semi-implicit Euler rollout for ``[..., T, dof]`` torques."""
    if not isinstance(torque, torch.Tensor) or torque.ndim < 2:
        raise ValueError("torque must end in [T, dof]")
    compiled = _model_for(model, torque)
    q, qd, _ = _state_tensors(compiled, initial, require_acceleration=False)
    q_b, _ = _batch(compiled, q, "position")
    qd_b, _ = _batch(compiled, qd, "velocity")
    tau = torque if torque.ndim == 3 else torque.unsqueeze(0)
    if tau.shape[0] != q_b.shape[0] or tau.shape[-1] != compiled.dof:
        raise ValueError("torque batch and dof must match the initial state")
    dt = torch.as_tensor(timestep, dtype=compiled.dtype, device=compiled.device)
    if dt.ndim != 0 or not bool(torch.isfinite(dt).item()) or float(dt) <= 0:
        raise ValueError("timestep must be a positive finite scalar")
    positions, velocities, accelerations = [q_b], [qd_b], []
    for step in range(tau.shape[1]):
        fd = forward_dynamics(compiled, positions[-1], velocities[-1], tau[:, step])
        next_velocity = velocities[-1] + fd.acceleration * dt
        next_position = positions[-1] + next_velocity * dt
        accelerations.append(fd.acceleration)
        velocities.append(next_velocity)
        positions.append(next_position)
    return DynamicsRolloutResult(
        torch.stack(positions, 1),
        torch.stack(velocities, 1),
        (torch.stack(accelerations, 1) if accelerations
         else q_b.new_empty((q_b.shape[0], 0, compiled.dof))),
        tau,
    )


def dynamics_cost(
    model: WholeBodyModel | TreeRobot,
    torque: torch.Tensor,
    config: DynamicsCostConfig = DynamicsCostConfig(),
) -> DynamicsCostResult:
    """Quadratic generalized-effort and torque-limit violation costs."""
    if not isinstance(torque, torch.Tensor):
        raise TypeError("torque must be a torch.Tensor")
    compiled = _model_for(model, torque)
    values, _ = _batch(compiled, torque, "torque")
    if (
        not math.isfinite(config.effort_weight)
        or not math.isfinite(config.limit_weight)
        or config.effort_weight < 0
        or config.limit_weight < 0
    ):
        raise ValueError("dynamics cost weights must be finite and nonnegative")
    effort = config.effort_weight * values.square().sum(-1)
    violation = (values.abs() - compiled.effort_limits).clamp_min(0)
    limit = config.limit_weight * violation.square().sum(-1)
    return DynamicsCostResult(effort + limit, effort, limit)
