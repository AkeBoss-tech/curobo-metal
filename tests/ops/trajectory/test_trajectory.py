from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from curobo_metal.ops.costs import CollisionModel
from curobo_metal.ops.kinematics import KinematicChain
from curobo_metal.ops.trajectory import (
    TrajectoryProblem,
    TrajectoryWeights,
    derivative_cost,
    evaluate_trajectory,
    interpolate_trajectory,
    interpolated_states,
    minimum_jerk_trajectory,
    optimize_trajectory,
    retime_trajectory,
    trajectory_collision_cost,
    trajectory_metrics,
)
from curobo_metal.reference import SerialRobot
from curobo_metal.optim import LBFGSConfig, ParticleConfig

FIXTURES = Path(__file__).parents[2] / "fixtures" / "trajectory"


def _case(name: str, device: str = "cpu", dtype: torch.dtype = torch.float64) -> TrajectoryProblem:
    value = json.loads((FIXTURES / f"{name}.json").read_text())
    inputs, options = value["inputs"], value["options"]
    tensor = lambda data: torch.tensor(data, device=device, dtype=dtype)
    collision = None
    if "collision" in inputs:
        raw = inputs["collision"]
        collision = CollisionModel(
            tensor(raw["local_spheres"]),
            torch.tensor(raw["link_indices"], device=device, dtype=torch.int64),
            cuboid_centers=tensor(raw["cuboid_centers"]),
            cuboid_rotations=tensor(raw["cuboid_rotations"]),
            cuboid_half_extents=tensor(raw["cuboid_half_extents"]),
            padding=raw.get("padding", 0.0),
            activation_distance=raw.get("activation_distance", 0.0),
            weight=raw.get("weight", 1.0),
        )
    return TrajectoryProblem(
        KinematicChain(SerialRobot.from_dict(value["robot"]), device=device, dtype=dtype),
        tensor(inputs["start"]), tensor(inputs["goal"]), tensor(inputs["lower"]),
        tensor(inputs["upper"]), inputs["steps"], inputs["dt"],
        seeds=tensor(inputs["seeds"]) if "seeds" in inputs else None,
        weights=TrajectoryWeights(**options["weights"]),
        collision_model=collision,
        collision_subdivisions=options["collision_subdivisions"],
        endpoint_tolerance=max(options["endpoint_tolerance"], 1e-5 if dtype == torch.float32 else 0),
        collision_tolerance=options["collision_tolerance"],
        max_iterations=options["max_iterations"],
        gradient_tolerance=1e-6 if dtype == torch.float32 else options["gradient_tolerance"],
        learning_rate=0.02,
    )


def test_costs_match_oracle_definition_and_autograd() -> None:
    q = torch.tensor(
        [[0.0, 0.2], [0.1, -0.3], [0.5, 0.4], [1.2, 0.1], [1.4, 0.8]],
        dtype=torch.float64, requires_grad=True,
    )
    total = sum(derivative_cost(q, n, 0.2, weight=w) for n, w in ((1, .7), (2, 1.1), (3, .3)))
    expected = sum(
        .5 * w * .2 ** (1 - 2 * n)
        * np.square(np.diff(q.detach().numpy(), n=n, axis=0)).sum()
        for n, w in ((1, .7), (2, 1.1), (3, .3))
    )
    assert total.item() == pytest.approx(expected)
    gradient = torch.autograd.grad(total, q)[0]
    assert gradient.shape == q.shape and torch.isfinite(gradient).all()


def test_minimum_jerk_interpolation_retime_and_metrics() -> None:
    start = torch.tensor([-1.0, .5], dtype=torch.float64)
    goal = torch.tensor([2.0, -.5], dtype=torch.float64)
    q = minimum_jerk_trajectory(start, goal, 5)
    torch.testing.assert_close(q[[0, -1]], torch.stack((start, goal)), rtol=0, atol=0)
    states, coordinates = interpolated_states(q[::2], 2)
    assert states.shape == (5, 2)
    torch.testing.assert_close(coordinates, torch.arange(5, dtype=torch.float64) / 2)
    sampled = interpolate_trajectory(q, .1, .06)
    torch.testing.assert_close(sampled[[0, -1]], q[[0, -1]], rtol=0, atol=0)
    _, slower_dt = retime_trajectory(
        q, .1, velocity_limit=torch.full((2,), 1.0), acceleration_limit=torch.full((2,), 10.0)
    )
    assert slower_dt >= .1
    metrics = trajectory_metrics(_case("two_link_obstacle_free"), q)
    assert metrics.duration == pytest.approx(.8)
    assert torch.isfinite(metrics.maximum_jerk)


@pytest.mark.parametrize(
    ("name", "status", "success"),
    [
        ("two_link_obstacle_free", "success", True),
        ("two_link_obstacle_detour", "success", True),
        ("panda_obstacle_free", "success", True),
        ("infeasible", "endpoint_infeasible", False),
        ("limit_constrained", "success", True),
    ],
)
def test_oracle_cases_have_correct_production_category(name: str, status: str, success: bool) -> None:
    problem = _case(name)
    result = optimize_trajectory(problem)
    assert result.status == (status,)
    assert result.success.tolist() == [success]
    assert result.selected_seed == (0 if success else None)
    if name != "infeasible":
        assert torch.all(result.maximum_limit_violation == 0)
    else:
        assert result.maximum_limit_violation[0] > 0
    assert torch.all(result.endpoint_error == 0) if name != "infeasible" else True
    if name == "two_link_obstacle_detour":
        assert result.minimum_clearance[0] >= 0
        cost, clearance = trajectory_collision_cost(
            problem.chain, result.trajectories, problem.collision_model, subdivisions=2
        )
        assert clearance[0] >= 0 and cost[0] >= 0


def test_problem_and_seed_batch_are_independent_and_first_tie_wins() -> None:
    base = _case("two_link_obstacle_free")
    start = torch.stack((base.start, base.start + .1))
    goal = torch.stack((base.goal, base.goal - .1))
    seeds = minimum_jerk_trajectory(start, goal, base.steps)
    seeds = torch.stack((seeds, seeds), dim=1)
    problem = TrajectoryProblem(
        base.chain, start, goal, base.lower, base.upper, base.steps, base.dt,
        seeds=seeds, weights=base.weights, max_iterations=4,
    )
    result = optimize_trajectory(problem)
    assert result.trajectories.shape == (2, 2, base.steps, base.chain.dof)
    assert result.success.all()
    assert result.selected_seed == (0, 0)
    torch.testing.assert_close(result.trajectories[:, 0], result.trajectories[:, 1])
    empty = optimize_trajectory(TrajectoryProblem(
        **{**base.__dict__, "seeds": torch.empty(
            (0, base.steps, base.chain.dof), dtype=base.start.dtype
        )}
    ))
    assert empty.trajectories.shape == (0, base.steps, base.chain.dof)
    assert empty.status == () and empty.selected_seed is None


def test_evaluation_and_unrolled_optimizer_are_differentiable() -> None:
    base = _case("two_link_obstacle_free")
    seed = minimum_jerk_trajectory(base.start, base.goal, base.steps).detach().requires_grad_(True)
    problem = TrajectoryProblem(**{**base.__dict__, "seeds": seed[None], "max_iterations": 2})
    result = optimize_trajectory(problem, differentiable=True)
    gradient = torch.autograd.grad(result.trajectories.square().sum(), seed)[0]
    assert gradient.shape == seed.shape and torch.isfinite(gradient).all()
    cost = evaluate_trajectory(problem, result.trajectories)
    assert cost.value.shape == (1, 1)


@pytest.mark.parametrize("optimizer", ["lbfgs", "particle", "es"])
def test_portable_optimizer_choices_on_robotics_fixture(optimizer: str) -> None:
    base = _case("two_link_obstacle_free")
    options = {
        "optimizer": optimizer,
        "max_iterations": 12,
        "lbfgs": LBFGSConfig(iterations=12),
        "particle": ParticleConfig(
            iterations=12, particles=24, elite_count=6, initial_std=.1, seed=9,
        ),
    }
    first = optimize_trajectory(TrajectoryProblem(**{**base.__dict__, **options}))
    second = optimize_trajectory(TrajectoryProblem(**{**base.__dict__, **options}))
    assert first.trajectories.shape == (1, base.steps, base.chain.dof)
    assert torch.isfinite(first.objective).all()
    torch.testing.assert_close(first.trajectories, second.trajectories, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_float32_fixed_suite_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    for name in ("two_link_obstacle_free", "two_link_obstacle_detour", "panda_obstacle_free",
                 "infeasible", "limit_constrained"):
        result = optimize_trajectory(_case(name, "mps", torch.float32))
        assert result.trajectories.device.type == "mps"
        expected = "endpoint_infeasible" if name == "infeasible" else "success"
        assert result.status == (expected,)
