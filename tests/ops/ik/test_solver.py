from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from curobo_metal.ops.costs import (
    CollisionModel,
    collision_cost,
    joint_limit_cost,
    pose_cost,
    pose_error,
    smoothness_cost,
)
from curobo_metal.ops.ik import IKProblem, solve_ik
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics
from curobo_metal.reference import SerialRobot

FIXTURES = Path(__file__).parents[2] / "fixtures"


def _case(name: str, device: str = "cpu", dtype: torch.dtype = torch.float64) -> IKProblem:
    value = json.loads((FIXTURES / "ik" / f"{name}.json").read_text())
    inputs, options = value["inputs"], value["options"]
    tensor = lambda data: torch.tensor(data, device=device, dtype=dtype)
    collision = None
    if "collision" in inputs:
        raw = inputs["collision"]
        collision = CollisionModel(
            local_spheres=tensor(raw["local_spheres"]),
            link_indices=torch.tensor(raw["link_indices"], device=device, dtype=torch.int64),
            cuboid_centers=tensor(raw["cuboid_centers"]),
            cuboid_rotations=tensor(raw["cuboid_rotations"]),
            cuboid_half_extents=tensor(raw["cuboid_half_extents"]),
            activation_distance=raw["activation_distance"],
            weight=raw["weight"],
        )
    return IKProblem(
        chain=KinematicChain(SerialRobot.from_dict(value["robot"]), device=device, dtype=dtype),
        target_position=tensor(inputs["target_position"]),
        target_quaternion=tensor(inputs["target_quaternion"]),
        seeds=tensor(inputs["seeds"]),
        lower=tensor(inputs["lower"]),
        upper=tensor(inputs["upper"]),
        pose_weights=tensor(inputs["pose_weights"]),
        position_tolerance=options["position_tolerance"],
        rotation_tolerance=options["rotation_tolerance"],
        max_iterations=options["max_iterations"],
        step_tolerance=options["step_tolerance"],
        collision_model=collision,
    )


def _gradient(function, value: torch.Tensor) -> torch.Tensor:
    candidate = value.detach().clone().requires_grad_(True)
    return torch.autograd.grad(function(candidate).sum(), candidate)[0]


def test_cost_values_and_autograd_match_numpy_oracle() -> None:
    dtype = torch.float64
    error = torch.tensor([.2, -.3, .4, .1, -.2, .3], dtype=dtype)
    weights = torch.arange(1, 7, dtype=dtype)
    assert pose_cost(error, weights).item() == pytest.approx(
        .5 * np.sum(np.arange(1, 7) * np.array([.2, -.3, .4, .1, -.2, .3]) ** 2)
    )
    torch.testing.assert_close(_gradient(lambda x: pose_cost(x, weights), error), weights * error)

    q = torch.tensor([-1.2, .1, 1.4], dtype=dtype)
    lower, upper = torch.full((3,), -1., dtype=dtype), torch.full((3,), 1., dtype=dtype)
    torch.testing.assert_close(
        _gradient(lambda x: joint_limit_cost(x, lower, upper, margin=.1, weight=2), q),
        torch.tensor([-.6, 0., 1.], dtype=dtype),
    )
    trajectory = torch.tensor([[0., .2], [.3, -.1], [.9, .4], [1.1, .8]], dtype=dtype)
    grad = _gradient(
        lambda x: smoothness_cost(x, velocity_weight=.7, acceleration_weight=1.3),
        trajectory,
    )
    assert torch.isfinite(grad).all()
    clearance = torch.tensor([-.2, .1, .8], dtype=dtype)
    torch.testing.assert_close(
        _gradient(lambda x: collision_cost(x, activation_distance=.4, weight=2), clearance),
        torch.tensor([-1.2, -.6, 0.], dtype=dtype),
    )


def test_pose_error_is_batched_and_differentiable_through_production_fk() -> None:
    problem = _case("reachable")
    q = problem.seeds.detach().clone().requires_grad_(True)
    transform = forward_kinematics(problem.chain, q).transforms[:, -1]
    residual = pose_error(
        transform,
        problem.target_position.expand(q.shape[0], 3),
        problem.target_quaternion.expand(q.shape[0], 4),
    )
    assert residual.shape == (1, 6)
    gradient = torch.autograd.grad(pose_cost(residual, problem.pose_weights).sum(), q)[0]
    assert gradient.shape == q.shape and torch.isfinite(gradient).all()
    half_turn = torch.eye(4, dtype=torch.float64)
    half_turn[:3, :3] = torch.diag(torch.tensor([1., -1., -1.], dtype=torch.float64))
    exact = pose_error(
        half_turn, torch.zeros(3, dtype=torch.float64),
        torch.tensor([1., 0., 0., 0.], dtype=torch.float64),
    )
    torch.testing.assert_close(
        exact[3:], torch.tensor([torch.pi, 0., 0.], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("reachable", "success"),
        ("unreachable", "infeasible_or_stationary"),
        ("limit_constrained", "limit_constrained"),
        ("collision_constrained", "collision_constrained"),
    ],
)
def test_oracle_fixture_categories_and_residual_semantics(name: str, status: str) -> None:
    result = solve_ik(_case(name))
    assert result.status == (status,)
    assert bool(result.success[0]) is (status == "success")
    if status == "success":
        assert result.position_error[0] <= 2e-5
        assert result.rotation_error[0] <= 2e-5
        assert result.selected_seed == 0
    if status == "limit_constrained":
        assert torch.isclose(result.solutions.abs(), torch.tensor(.1, dtype=torch.float64)).any()
    if status == "collision_constrained":
        assert result.collision_cost[0] > 0


def test_goal_seed_cartesian_batch_selection_and_deterministic_replay() -> None:
    base = _case("reachable")
    desired = torch.tensor([[.6, -.8], [-.4, .7]], dtype=torch.float64)
    transforms = forward_kinematics(base.chain, desired).transforms[:, -1]
    # These planar goals have scalar-first z-axis quaternions.
    angles = desired.sum(dim=1)
    quaternion = torch.stack(
        (torch.cos(angles / 2), torch.zeros_like(angles), torch.zeros_like(angles),
         torch.sin(angles / 2)),
        dim=1,
    )
    problem = IKProblem(
        base.chain,
        transforms[:, :3, 3],
        quaternion,
        torch.tensor([[0., 0.], [.5, -.5], [-.5, .5]], dtype=torch.float64),
        base.lower,
        base.upper,
        base.pose_weights,
        position_tolerance=3e-5,
        rotation_tolerance=3e-5,
        max_iterations=300,
    )
    first, second = solve_ik(problem), solve_ik(problem)
    assert first.solutions.shape == (2, 3, 2)
    assert first.success.all()
    assert first.selected_seed == second.selected_seed
    torch.testing.assert_close(first.solutions, second.solutions, rtol=0, atol=0)
    assert first.status == second.status


def test_unbatched_seed_and_empty_batch_shapes() -> None:
    problem = _case("reachable")
    single = solve_ik(IKProblem(**{**problem.__dict__, "seeds": problem.seeds[0]}))
    assert single.solutions.shape == (1, 2)
    assert not single.input_seeds_were_batched
    empty = solve_ik(IKProblem(**{**problem.__dict__, "seeds": problem.seeds[:0]}))
    assert empty.solutions.shape == (0, 2)
    assert empty.selected_seed is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_float32_fixed_suite_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    for name in ("reachable", "unreachable", "limit_constrained", "collision_constrained"):
        result = solve_ik(_case(name, "mps", torch.float32))
        assert result.solutions.device.type == "mps"
        assert result.status[0] in {
            "success", "infeasible_or_stationary", "limit_constrained", "collision_constrained"
        }
