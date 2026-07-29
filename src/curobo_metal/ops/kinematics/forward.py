"""Composed PyTorch forward kinematics for CPU and Apple MPS."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from curobo_metal.backend import resolve_device, validate_tensor_device
from curobo_metal.reference.forward_kinematics import SerialRobot


@dataclass(frozen=True)
class FKResult:
    """Forward-kinematics values, all resident on the input device."""

    transforms: torch.Tensor
    transform_jacobian: torch.Tensor
    geometric_jacobian: torch.Tensor
    link_names: tuple[str, ...]
    input_was_batched: bool


class KinematicChain:
    """Immutable tensor metadata for a validated serial robot.

    Compile a chain once for a target device and dtype, then reuse it across
    calls. This keeps metadata transfer outside measured or repeated FK work.
    """

    def __init__(
        self,
        robot: SerialRobot,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("kinematic metadata dtype must be float32 or float64")
        resolved = resolve_device(device)
        if resolved.type == "mps" and dtype != torch.float32:
            raise TypeError("MPS forward kinematics supports only float32")

        self._robot = robot
        self.device = resolved
        self.dtype = dtype
        self.origins = torch.stack(
            [
                torch.as_tensor(joint.origin, device=resolved, dtype=dtype).clone()
                for joint in robot.joints
            ]
        )
        self.axes = torch.stack(
            [
                torch.as_tensor(joint.axis, device=resolved, dtype=dtype).clone()
                for joint in robot.joints
            ]
        )
        self.kinds = tuple(joint.kind for joint in robot.joints)
        self.q_indices = tuple(joint.q_index for joint in robot.joints)
        self.link_names = tuple(joint.name for joint in robot.joints)
        self.dof = robot.dof
        if resolved.type == "mps":
            self._metal_kinds = torch.tensor(
                [
                    {"fixed": 0, "revolute": 1, "prismatic": 2}[kind]
                    for kind in self.kinds
                ],
                dtype=torch.int32,
                device=resolved,
            )
            self._metal_q_indices = torch.tensor(
                [-1 if value is None else value for value in self.q_indices],
                dtype=torch.int32,
                device=resolved,
            )

    @classmethod
    def from_serial_robot(
        cls,
        robot: SerialRobot,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "KinematicChain":
        return cls(robot, device=device, dtype=dtype)

    def to(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "KinematicChain":
        return type(self)(
            self._robot,
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )


def _validate_q(chain: KinematicChain, q: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    validate_tensor_device(q, expected=chain.device)
    if q.dtype not in (torch.float32, torch.float64):
        raise TypeError("q must have dtype float32 or float64")
    if q.dtype != chain.dtype:
        raise TypeError(f"q dtype {q.dtype} does not match chain dtype {chain.dtype}")
    if q.device.type == "mps" and q.dtype != torch.float32:
        raise TypeError("MPS forward kinematics supports only float32")
    input_was_batched = q.ndim == 2
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if q.ndim != 2 or q.shape[1] != chain.dof:
        raise ValueError(
            f"q must have shape [{chain.dof}] or [B, {chain.dof}]"
        )
    if not bool(torch.isfinite(q).all().item()):
        raise ValueError("q must contain only finite values")
    return q, input_was_batched


def _skew(axis: torch.Tensor) -> torch.Tensor:
    zero = axis.new_zeros(())
    x, y, z = axis.unbind()
    return torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )


def _motion(
    kind: str, axis: torch.Tensor, position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = position.shape[0]
    eye3 = torch.eye(3, dtype=position.dtype, device=position.device)
    zero_column = position.new_zeros((batch, 3, 1))
    bottom = position.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(batch, 1, 4)
    derivative_bottom = position.new_zeros((batch, 1, 4))

    if kind == "fixed":
        transform = torch.eye(
            4, dtype=position.dtype, device=position.device
        ).expand(batch, 4, 4)
        return transform, torch.zeros_like(transform)
    if kind == "prismatic":
        translation = position[:, None] * axis[None, :]
        top = torch.cat((eye3.expand(batch, 3, 3), translation[..., None]), dim=-1)
        derivative_top = torch.cat(
            (position.new_zeros((batch, 3, 3)), axis.expand(batch, 3)[..., None]),
            dim=-1,
        )
        return (
            torch.cat((top, bottom), dim=-2),
            torch.cat((derivative_top, derivative_bottom), dim=-2),
        )

    k = _skew(axis)
    k2 = k @ k
    sine = torch.sin(position)[:, None, None]
    cosine = torch.cos(position)[:, None, None]
    rotation = eye3 + sine * k + (1.0 - cosine) * k2
    rotation_derivative = cosine * k + sine * k2
    transform_top = torch.cat((rotation, zero_column), dim=-1)
    derivative_top = torch.cat((rotation_derivative, zero_column), dim=-1)
    return (
        torch.cat((transform_top, bottom), dim=-2),
        torch.cat((derivative_top, derivative_bottom), dim=-2),
    )


def forward_kinematics(
    chain: KinematicChain | SerialRobot, q: torch.Tensor
) -> FKResult:
    """Evaluate serial-chain FK without implicit dtype or device transfers."""
    if isinstance(chain, SerialRobot):
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        chain = KinematicChain(
            chain,
            device=q.device,
            dtype=q.dtype,
        )
    if not isinstance(chain, KinematicChain):
        raise TypeError("chain must be a KinematicChain or SerialRobot")
    configurations, input_was_batched = _validate_q(chain, q)
    if chain.device.type == "mps":
        from curobo_metal.ops.kinematics.metal import (
            fused_forward_kinematics,
            supports_fused_chain,
        )

        if supports_fused_chain(chain):
            transform_tensor, derivative_tensor, geometric = (
                fused_forward_kinematics(chain, configurations)
            )
            return FKResult(
                transforms=transform_tensor,
                transform_jacobian=derivative_tensor,
                geometric_jacobian=geometric,
                link_names=chain.link_names,
                input_was_batched=input_was_batched,
            )
    batch = configurations.shape[0]
    dof = chain.dof
    current = torch.eye(
        4, dtype=configurations.dtype, device=configurations.device
    ).expand(batch, 4, 4)
    current_derivatives = configurations.new_zeros((batch, dof, 4, 4))
    transforms: list[torch.Tensor] = []
    derivatives: list[torch.Tensor] = []

    for link_index, (kind, q_index) in enumerate(
        zip(chain.kinds, chain.q_indices, strict=True)
    ):
        position = (
            configurations.new_zeros((batch,))
            if q_index is None
            else configurations[:, q_index]
        )
        motion, motion_derivative = _motion(
            kind, chain.axes[link_index], position
        )
        origin = chain.origins[link_index].expand(batch, 4, 4)
        local = origin @ motion
        local_derivative = origin @ motion_derivative
        next_derivatives = current_derivatives @ local[:, None]
        if q_index is not None:
            selector = (
                torch.arange(dof, device=configurations.device) == q_index
            ).to(configurations.dtype)
            next_derivatives = next_derivatives + (
                current @ local_derivative
            )[:, None] * selector[None, :, None, None]
        current = current @ local
        current_derivatives = next_derivatives
        transforms.append(current)
        derivatives.append(current_derivatives.permute(0, 2, 3, 1))

    transform_tensor = torch.stack(transforms, dim=1)
    derivative_tensor = torch.stack(derivatives, dim=1)
    rotation = transform_tensor[:, :, :3, :3]
    d_rotation = derivative_tensor[:, :, :3, :3, :]
    omega_matrix = torch.einsum("blikj,blmk->blimj", d_rotation, rotation)
    angular = torch.stack(
        (
            omega_matrix[:, :, 2, 1, :],
            omega_matrix[:, :, 0, 2, :],
            omega_matrix[:, :, 1, 0, :],
        ),
        dim=2,
    )
    geometric = torch.cat(
        (derivative_tensor[:, :, :3, 3, :], angular), dim=2
    )
    return FKResult(
        transforms=transform_tensor,
        transform_jacobian=derivative_tensor,
        geometric_jacobian=geometric,
        link_names=chain.link_names,
        input_was_batched=input_was_batched,
    )
