from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from curobo_metal.ops.whole_body import (
    DynamicsCostConfig,
    WholeBodyModel,
    bias_torque,
    dynamics_cost,
    gravity_torque,
    inverse_dynamics,
    mass_matrix,
    tree_forward_kinematics,
)
from curobo_metal.reference import (
    TreeRobot,
    bias_torque as reference_bias,
    gravity_torque as reference_gravity,
    inverse_dynamics as reference_inverse_dynamics,
    mass_matrix as reference_mass_matrix,
    tree_forward_kinematics as reference_fk,
)

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "whole_body"


def load(name: str) -> tuple[TreeRobot, np.ndarray, np.ndarray, np.ndarray]:
    case = json.loads((FIXTURES / name).read_text())
    inputs = case["inputs"]
    return (
        TreeRobot.from_dict(case["robot"]),
        np.asarray(inputs["q"]),
        np.asarray(inputs["qd"]),
        np.asarray(inputs["qdd"]),
    )


def devices() -> list[str]:
    return ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("name", ["branched_toy.json", "panda_inertial.json"])
def test_float32_matches_numpy_fixture(device: str, name: str) -> None:
    robot, q_np, qd_np, qdd_np = load(name)
    model = WholeBodyModel(robot, device=device, dtype=torch.float32)
    tensor = lambda value: torch.tensor(value, device=device, dtype=torch.float32)
    q, qd, qdd = tensor(q_np), tensor(qd_np), tensor(qdd_np)
    actual_fk = tree_forward_kinematics(model, q)
    expected_fk = reference_fk(robot, q_np)
    torch.testing.assert_close(
        actual_fk.transforms.cpu(),
        torch.tensor(expected_fk.transforms, dtype=torch.float32),
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        actual_fk.transform_jacobian.cpu(),
        torch.tensor(expected_fk.transform_jacobian, dtype=torch.float32),
        rtol=5e-4,
        atol=5e-4,
    )
    torch.testing.assert_close(
        actual_fk.geometric_jacobian.cpu(),
        torch.tensor(expected_fk.geometric_jacobian, dtype=torch.float32),
        rtol=5e-4,
        atol=5e-4,
    )
    torque = inverse_dynamics(model, q, qd, qdd).torque
    expected_torque = reference_inverse_dynamics(robot, q_np, qd_np, qdd_np).torque
    np.testing.assert_allclose(
        torque.detach().cpu(), expected_torque, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        mass_matrix(model, q).detach().cpu(),
        reference_mass_matrix(robot, q_np),
        rtol=8e-4,
        atol=8e-4,
    )
    np.testing.assert_allclose(
        gravity_torque(model, q).detach().cpu(),
        reference_gravity(robot, q_np),
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        bias_torque(model, q, qd).detach().cpu(),
        reference_bias(robot, q_np, qd_np),
        rtol=5e-4,
        atol=5e-4,
    )


@pytest.mark.parametrize("name", ["branched_toy.json", "panda_inertial.json"])
def test_cpu_float64_decomposition_and_positive_definite(name: str) -> None:
    robot, q_np, qd_np, qdd_np = load(name)
    model = WholeBodyModel(robot, dtype=torch.float64)
    q = torch.tensor(q_np, dtype=torch.float64)
    qd = torch.tensor(qd_np, dtype=torch.float64)
    qdd = torch.tensor(qdd_np, dtype=torch.float64)
    torque = inverse_dynamics(model, q, qd, qdd).torque
    matrix = mass_matrix(model, q)
    bias = bias_torque(model, q, qd)
    torch.testing.assert_close(
        torque, torch.einsum("bij,bj->bi", matrix, qdd) + bias,
        rtol=2e-10, atol=2e-10,
    )
    torch.testing.assert_close(matrix, matrix.transpose(-1, -2), atol=2e-12, rtol=0)
    assert torch.linalg.eigvalsh(matrix).min() > 1e-5


def test_tree_effectors_sibling_zeros_and_mimic_projection() -> None:
    robot, q_np, qd_np, qdd_np = load("branched_toy.json")
    model = WholeBodyModel(robot, dtype=torch.float64)
    q = torch.tensor(q_np, dtype=torch.float64)
    result = tree_forward_kinematics(model, q)
    assert result.link_names == tuple(link.name for link in robot.links)
    assert result.end_effector_indices == (2, 3, 4)
    assert result.end_effector_transforms.shape == (2, 3, 4, 4)
    torch.testing.assert_close(
        result.geometric_jacobian[:, 4, :, :2],
        torch.zeros((2, 6, 2), dtype=torch.float64),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        result.geometric_jacobian[:, 2:4, :, 2],
        torch.zeros((2, 2, 6), dtype=torch.float64),
        rtol=0, atol=0,
    )
    # The negative-ratio mimic contributes to its active source effort.
    torque = inverse_dynamics(
        model, q, torch.tensor(qd_np), torch.tensor(qdd_np)
    ).torque
    expected = reference_inverse_dynamics(robot, q_np, qd_np, qdd_np).torque
    np.testing.assert_allclose(torque, expected, atol=2e-10, rtol=2e-10)


def test_autograd_matches_finite_differences_and_analytical_jacobian() -> None:
    robot, q_np, qd_np, qdd_np = load("branched_toy.json")
    model = WholeBodyModel(robot, dtype=torch.float64)
    q = torch.tensor(q_np[1], dtype=torch.float64, requires_grad=True)
    qd = torch.tensor(qd_np[1], dtype=torch.float64, requires_grad=True)
    qdd = torch.tensor(qdd_np[1], dtype=torch.float64, requires_grad=True)
    fk = tree_forward_kinematics(model, q)
    weights = torch.linspace(
        0.1, 1.0, fk.transforms.numel(), dtype=torch.float64
    ).reshape_as(fk.transforms)
    fk_gradient = torch.autograd.grad((fk.transforms * weights).sum(), q, retain_graph=True)[0]
    expected_fk_gradient = torch.einsum(
        "blrc,blrcj->j", weights, fk.transform_jacobian
    )
    torch.testing.assert_close(fk_gradient, expected_fk_gradient, rtol=2e-10, atol=2e-10)

    weights_tau = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
    torque = inverse_dynamics(model, q, qd, qdd).torque[0]
    gradients = torch.autograd.grad((torque * weights_tau).sum(), (q, qd, qdd))
    step = 1e-6
    bases = [q.detach(), qd.detach(), qdd.detach()]
    for argument, analytical in enumerate(gradients):
        numeric = torch.empty_like(analytical)
        for j in range(model.dof):
            plus, minus = [x.clone() for x in bases], [x.clone() for x in bases]
            plus[argument][j] += step
            minus[argument][j] -= step
            numeric[j] = (
                inverse_dynamics(model, *plus).torque[0].dot(weights_tau)
                - inverse_dynamics(model, *minus).torque[0].dot(weights_tau)
            ) / (2 * step)
        torch.testing.assert_close(analytical, numeric, rtol=5e-6, atol=5e-7)


@pytest.mark.parametrize("device", devices())
def test_cost_autograd_and_limit_boundary_subgradient(device: str) -> None:
    robot, _, _, _ = load("branched_toy.json")
    model = WholeBodyModel(robot, device=device, dtype=torch.float32)
    torque = torch.tensor(
        [[6.0, -10.0, 5.0], [13.0, 0.0, -7.0]],
        device=device, requires_grad=True,
    )
    result = dynamics_cost(
        model, torque, DynamicsCostConfig(effort_weight=0.5, limit_weight=3.0)
    )
    torch.testing.assert_close(
        result.limit.cpu(), torch.tensor([12.0, 15.0]), rtol=0, atol=0
    )
    limit_gradient = torch.autograd.grad(result.limit.sum(), torque)[0]
    torch.testing.assert_close(
        limit_gradient[0, 2], torch.tensor(0.0, device=device), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", devices())
def test_unbatched_noncontiguous_empty_and_owning_outputs(device: str) -> None:
    robot, q_np, qd_np, qdd_np = load("branched_toy.json")
    model = WholeBodyModel(robot, device=device, dtype=torch.float32)
    backing = torch.zeros((3, 2), device=device)
    backing[:, 0] = torch.tensor(q_np[1], device=device, dtype=torch.float32)
    q = backing[:, 0]
    assert not q.is_contiguous()
    qd = torch.tensor(qd_np[1], device=device, dtype=torch.float32)
    qdd = torch.tensor(qdd_np[1], device=device, dtype=torch.float32)
    result = inverse_dynamics(model, q, qd, qdd)
    assert result.input_was_batched is False
    assert result.torque.shape == (1, 3)
    assert result.torque.untyped_storage().data_ptr() != q.untyped_storage().data_ptr()
    empty = tree_forward_kinematics(model, torch.empty((0, 3), device=device))
    assert empty.transforms.shape == (0, 6, 4, 4)
    assert empty.transform_jacobian.shape == (0, 6, 4, 4, 3)


def test_validation_and_mps_policy() -> None:
    robot, q_np, qd_np, qdd_np = load("branched_toy.json")
    model = WholeBodyModel(robot)
    with pytest.raises(TypeError):
        tree_forward_kinematics(model, torch.zeros(3, dtype=torch.int64))
    with pytest.raises(ValueError):
        tree_forward_kinematics(model, torch.full((3,), torch.nan))
    with pytest.raises(ValueError, match="batch dimensions"):
        inverse_dynamics(
            model,
            torch.tensor(q_np, dtype=torch.float32),
            torch.tensor(qd_np[:1], dtype=torch.float32),
            torch.tensor(qdd_np, dtype=torch.float32),
        )
    with pytest.raises(ValueError):
        dynamics_cost(model, torch.zeros(3), DynamicsCostConfig(-1, 1))
    if torch.backends.mps.is_available():
        with pytest.raises(TypeError, match="only float32"):
            WholeBodyModel(robot, device="mps", dtype=torch.float64)
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
