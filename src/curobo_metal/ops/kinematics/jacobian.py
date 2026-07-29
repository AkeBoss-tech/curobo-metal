"""Public geometric-Jacobian API.

The Jacobian maps joint rates to a link-origin twist expressed in the
world/base frame. Rows ``0:3`` are linear velocity and rows ``3:6`` are
angular velocity. Columns follow the model's active-joint order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from curobo_metal.reference.forward_kinematics import SerialRobot
from curobo_metal.reference.tree_kinematics import TreeRobot
from .forward import KinematicChain, forward_kinematics


@dataclass(frozen=True)
class JacobianResult:
    """Selected world-frame geometric Jacobians, always with a batch axis."""

    jacobian: torch.Tensor
    link_names: tuple[str, ...]
    link_indices: tuple[int, ...]
    input_was_batched: bool

    @property
    def linear(self) -> torch.Tensor:
        return self.jacobian[:, :, :3]

    @property
    def angular(self) -> torch.Tensor:
        return self.jacobian[:, :, 3:]

    @property
    def end_effector(self) -> torch.Tensor:
        if len(self.link_indices) != 1:
            raise ValueError("end_effector requires exactly one selected link")
        return self.jacobian[:, 0]


def _selection(
    names: tuple[str, ...],
    links: str | int | Sequence[str | int] | None,
    default: Sequence[int],
) -> tuple[int, ...]:
    if links is None:
        raw: Sequence[str | int] = tuple(default)
    elif isinstance(links, (str, int)):
        raw = (links,)
    else:
        raw = tuple(links)
    result: list[int] = []
    for value in raw:
        if isinstance(value, str):
            if value not in names:
                raise KeyError(f"unknown link name: {value!r}")
            index = names.index(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            index = value if value >= 0 else len(names) + value
            if not 0 <= index < len(names):
                raise IndexError(f"link index out of range: {value}")
        else:
            raise TypeError("links must contain only link names or integer indices")
        if index in result:
            raise ValueError(f"duplicate link selection: {value!r}")
        result.append(index)
    return tuple(result)


def geometric_jacobian(
    model: KinematicChain | SerialRobot | TreeRobot | object,
    q: torch.Tensor,
    *,
    links: str | int | Sequence[str | int] | None = None,
    end_effectors: bool = False,
) -> JacobianResult:
    """Return differentiable world-frame link-origin geometric Jacobians.

    ``q`` may be ``[J]`` or ``[B,J]``; output is ``[B,L,6,J]``. With
    ``links=None``, every link is selected unless ``end_effectors=True``.
    CPU supports float32/float64; MPS supports float32.
    """
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if links is not None and end_effectors:
        raise ValueError("links and end_effectors are mutually exclusive")
    if isinstance(model, (KinematicChain, SerialRobot)):
        fk = forward_kinematics(model, q)
        default = (len(fk.link_names) - 1,) if end_effectors else range(len(fk.link_names))
    else:
        from curobo_metal.ops.whole_body import WholeBodyModel, tree_forward_kinematics
        if not isinstance(model, (WholeBodyModel, TreeRobot)):
            raise TypeError(
                "model must be a KinematicChain, SerialRobot, WholeBodyModel, or TreeRobot"
            )
        fk = tree_forward_kinematics(model, q)
        default = fk.end_effector_indices if end_effectors else range(len(fk.link_names))
    indices = _selection(fk.link_names, links, default)
    index = torch.tensor(indices, dtype=torch.int64, device=q.device)
    return JacobianResult(
        fk.geometric_jacobian.index_select(1, index),
        tuple(fk.link_names[i] for i in indices),
        indices,
        fk.input_was_batched,
    )


def end_effector_jacobian(
    model: KinematicChain | SerialRobot | TreeRobot | object,
    q: torch.Tensor,
    *,
    link: str | int | None = None,
) -> torch.Tensor:
    """Return one end-effector Jacobian as ``[B,6,J]``."""
    result = (
        geometric_jacobian(model, q, end_effectors=True)
        if link is None
        else geometric_jacobian(model, q, links=link)
    )
    if len(result.link_indices) != 1:
        raise ValueError("model has multiple end effectors; select one with link=")
    return result.end_effector
