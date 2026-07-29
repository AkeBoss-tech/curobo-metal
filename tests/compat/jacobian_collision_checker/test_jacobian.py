from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from curobo_metal.ops.kinematics import (
    KinematicChain, end_effector_jacobian, geometric_jacobian,
)
from curobo_metal.reference.forward_kinematics import SerialRobot
from curobo_metal.reference.tree_kinematics import TreeRobot


ROOT = Path(__file__).parents[3]


def test_serial_public_jacobian_selection_and_gradient() -> None:
    robot = SerialRobot.from_dict(
        json.loads((ROOT / "tests/fixtures/two_link_planar.json").read_text())["robot"]
    )
    q = torch.tensor([[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    result = geometric_jacobian(robot, q, links=-1)
    assert result.jacobian.shape == (2, 1, 6, 2)
    assert torch.equal(result.linear, result.jacobian[:, :, :3])
    assert torch.equal(end_effector_jacobian(robot, q), result.end_effector)
    result.jacobian.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_empty_selection_and_errors() -> None:
    robot = SerialRobot.from_dict(
        json.loads((ROOT / "tests/fixtures/two_link_planar.json").read_text())["robot"]
    )
    q = torch.zeros(robot.dof)
    assert geometric_jacobian(robot, q, links=[]).jacobian.shape == (1, 0, 6, robot.dof)
    with pytest.raises(KeyError, match="unknown link"):
        geometric_jacobian(robot, q, links="missing")
    with pytest.raises(ValueError, match="mutually exclusive"):
        geometric_jacobian(robot, q, links=0, end_effectors=True)


def test_tree_end_effector_jacobian() -> None:
    robot = TreeRobot.from_dict(
        json.loads((ROOT / "tests/fixtures/whole_body/branched_toy.json").read_text())["robot"]
    )
    q = torch.zeros(robot.dof, dtype=torch.float64)
    result = geometric_jacobian(robot, q, end_effectors=True)
    assert result.link_indices == robot.end_effectors
    assert result.jacobian.shape[2:] == (6, robot.dof)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_jacobian() -> None:
    robot = SerialRobot.from_dict(
        json.loads((ROOT / "tests/fixtures/two_link_planar.json").read_text())["robot"]
    )
    q = torch.zeros((2, robot.dof), device="mps")
    assert geometric_jacobian(KinematicChain(robot, device="mps"), q).jacobian.device.type == "mps"
