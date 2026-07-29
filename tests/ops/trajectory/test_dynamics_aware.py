from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from curobo_metal.ops.costs import CollisionModel
from curobo_metal.ops.kinematics import KinematicChain
from curobo_metal.ops.trajectory import (
    DynamicsAwareProblem,
    bspline_matrices,
    evaluate_dynamics_aware,
    optimize_dynamics_aware,
    retime_dynamics_aware,
    sample_bspline,
)
from curobo_metal.ops.whole_body import WholeBodyModel, inverse_dynamics
from curobo_metal.reference import SerialRobot, TreeRobot
from curobo_metal.motion_gen import JointState, MotionGen, MotionGenConfig

ROOT = Path(__file__).parents[3]


def panda_model(dtype: torch.dtype = torch.float64, device: str = "cpu") -> WholeBodyModel:
    case = json.loads(
        (ROOT / "tests/fixtures/whole_body/panda_inertial.json").read_text()
    )
    return WholeBodyModel(TreeRobot.from_dict(case["robot"]), dtype=dtype, device=device)


def basic_problem(model: WholeBodyModel, **kwargs: object) -> DynamicsAwareProblem:
    start = torch.tensor(
        [0.0, -0.7, 0.0, -2.0, 0.0, 1.4, 0.7],
        dtype=model.dtype, device=model.device,
    )
    goal = start + torch.tensor(
        [0.1, 0.05, -0.05, 0.1, 0.0, -0.05, 0.05],
        dtype=model.dtype, device=model.device,
    )
    lower = torch.full_like(start, -3.0)
    upper = torch.full_like(start, 3.0)
    defaults = dict(
        control_points=8, samples=21, duration=3.0,
        velocity_limits=torch.full_like(start, 3.0),
        acceleration_limits=torch.full_like(start, 8.0),
        jerk_limits=torch.full_like(start, 30.0),
        max_iterations=3,
    )
    defaults.update(kwargs)
    return DynamicsAwareProblem(model, start, goal, lower, upper, **defaults)


def test_bspline_derivative_matrices_and_partition_of_unity() -> None:
    dtype = torch.float64
    cp = torch.linspace(-0.4, 0.8, 8, dtype=dtype)[:, None]
    duration = 2.3
    matrices = bspline_matrices(8, 101, duration=duration, dtype=dtype)
    q, qd, qdd, jerk = sample_bspline(cp, matrices)
    torch.testing.assert_close(
        matrices.position.sum(-1), torch.ones(101, dtype=dtype), atol=2e-14, rtol=0
    )
    torch.testing.assert_close(
        matrices.velocity.sum(-1), torch.zeros(101, dtype=dtype), atol=2e-14, rtol=0
    )
    dt = duration / 100
    torch.testing.assert_close(
        (q[2:] - q[:-2]) / (2 * dt), qd[1:-1], atol=2e-4, rtol=2e-3
    )
    # Skip stencils crossing the uniform internal knots, where the cubic's
    # third derivative is intentionally discontinuous.
    keep = torch.ones(99, dtype=torch.bool)
    for knot in (20, 40, 60, 80):
        keep[knot - 2 : knot + 1] = False
    torch.testing.assert_close(
        ((qd[2:] - qd[:-2]) / (2 * dt))[keep], qdd[1:-1][keep],
        atol=3e-3, rtol=3e-3,
    )
    torch.testing.assert_close(
        ((qdd[2:] - qdd[:-2]) / (2 * dt))[keep], jerk[1:-1][keep],
        atol=2e-2, rtol=1e-2,
    )
    torch.testing.assert_close(q[0], cp[0], atol=0, rtol=0)
    torch.testing.assert_close(q[-1], cp[-1], atol=0, rtol=0)


def test_rnea_cost_gradients_reach_control_points_and_duration() -> None:
    model = panda_model()
    problem = basic_problem(model)
    phase = torch.linspace(0, 1, problem.control_points, dtype=model.dtype)
    cp = (
        problem.start[None, :] * (1 - phase[:, None])
        + problem.goal[None, :] * phase[:, None]
    ).unsqueeze(0).requires_grad_(True)
    duration = torch.tensor([2.5], dtype=model.dtype, requires_grad=True)
    cost = evaluate_dynamics_aware(problem, cp, duration)
    gradients = torch.autograd.grad(cost.effort.sum(), (cp, duration))
    assert torch.isfinite(gradients[0]).all() and gradients[0].abs().sum() > 0
    assert torch.isfinite(gradients[1]).all() and gradients[1].abs().sum() > 0


def test_branched_tree_rollout_uses_all_generalized_efforts() -> None:
    case = json.loads(
        (ROOT / "tests/fixtures/whole_body/branched_toy.json").read_text()
    )
    model = WholeBodyModel(TreeRobot.from_dict(case["robot"]), dtype=torch.float64)
    start = torch.tensor(case["inputs"]["q"][0], dtype=model.dtype)
    goal = torch.tensor(case["inputs"]["q"][1], dtype=model.dtype)
    phase = torch.linspace(0, 1, 6, dtype=model.dtype)
    cp = (start[None] + phase[:, None] * (goal - start)[None]).unsqueeze(0)
    problem = DynamicsAwareProblem(
        model, start, goal, torch.full_like(start, -3), torch.full_like(start, 3),
        control_points=6, samples=13, duration=2.0,
    )
    result = evaluate_dynamics_aware(problem, cp, torch.tensor([2.0], dtype=model.dtype))
    assert result.torque.shape == (1, 1, 13, model.dof)
    assert (result.torque.abs().amax(dim=(-3, -2)) > 0).all()


def test_feasible_batched_seeds_and_deterministic_statuses() -> None:
    model = panda_model()
    problem = basic_problem(model, torque_limits=torch.full((7,), 200.0, dtype=model.dtype))
    phase = torch.linspace(0, 1, problem.control_points, dtype=model.dtype)
    seed = problem.start[None] + phase[:, None] * (problem.goal - problem.start)[None]
    result = optimize_dynamics_aware(
        DynamicsAwareProblem(**{**problem.__dict__, "seeds": torch.stack((seed, seed))})
    )
    assert result.position.shape == (2, problem.samples, 7)
    assert result.selected_seed == 0
    assert result.status[0] == result.status[1]
    assert result.success.all()
    torch.testing.assert_close(result.position[:, 0], problem.start.expand(2, -1))
    torch.testing.assert_close(result.position[:, -1], problem.goal.expand(2, -1))


def test_torque_violation_and_fixed_control_point_retiming() -> None:
    model = panda_model()
    problem = basic_problem(
        model,
        torque_limits=torch.full((7,), 0.01, dtype=model.dtype),
        min_duration=0.5,
        max_duration=4.0,
    )
    phase = torch.linspace(0, 1, problem.control_points, dtype=model.dtype)
    cp = (problem.start[None] + phase[:, None] * (problem.goal - problem.start)[None]).unsqueeze(0)
    evaluated = evaluate_dynamics_aware(problem, cp, torch.tensor([2.0], dtype=model.dtype))
    assert evaluated.torque_limit.item() > 0
    assert evaluated.maximum_violation.item() > 0
    duration = retime_dynamics_aware(problem, cp, iterations=12)
    assert duration.shape == (1,)
    assert duration.item() == pytest.approx(problem.max_duration, abs=2e-3)


def test_collision_cost_composes_with_dynamics() -> None:
    model = panda_model()
    serial_case = json.loads(
        (ROOT / "tests/fixtures/panda_serial.json").read_text()
    )
    serial = SerialRobot.from_dict(serial_case["robot"])
    chain = KinematicChain(serial, dtype=model.dtype)
    collision = CollisionModel(
        local_spheres=torch.tensor([[0.0, 0.0, 0.0, 0.2]], dtype=model.dtype),
        link_indices=torch.tensor([0]),
        cuboid_centers=torch.tensor([[0.0, 0.0, 0.0]], dtype=model.dtype),
        cuboid_rotations=torch.eye(3, dtype=model.dtype).unsqueeze(0),
        cuboid_half_extents=torch.tensor([[0.3, 0.3, 0.3]], dtype=model.dtype),
    )
    problem = basic_problem(model, chain=chain, collision_model=collision)
    phase = torch.linspace(0, 1, problem.control_points, dtype=model.dtype)
    cp = (problem.start[None] + phase[:, None] * (problem.goal - problem.start)[None]).unsqueeze(0)
    cost = evaluate_dynamics_aware(problem, cp, torch.tensor([problem.duration], dtype=model.dtype))
    assert cost.collision.item() > 0
    assert cost.minimum_clearance.item() < 0


def test_motion_gen_optional_dynamics_aware_mode() -> None:
    model = panda_model()
    serial_case = json.loads(
        (ROOT / "tests/fixtures/panda_serial.json").read_text()
    )
    chain = KinematicChain(SerialRobot.from_dict(serial_case["robot"]), dtype=model.dtype)
    lower, upper = torch.full((7,), -3.0, dtype=model.dtype), torch.full((7,), 3.0, dtype=model.dtype)
    config = MotionGenConfig(
        chain, lower, upper, model.joint_names, steps=16, dt=0.2,
        max_trajectory_iterations=2, dynamics_aware=True, dynamics_model=model,
        dynamics_aware_options={
            "control_points": 8,
            "velocity_limits": torch.full((7,), 5.0, dtype=model.dtype),
            "acceleration_limits": torch.full((7,), 10.0, dtype=model.dtype),
            "jerk_limits": torch.full((7,), 50.0, dtype=model.dtype),
        },
    )
    start = torch.tensor(
        [0.0, -0.7, 0.0, -2.0, 0.0, 1.4, 0.7], dtype=model.dtype
    )
    result = MotionGen(config).plan_single_js(
        JointState(start), JointState(start), enable_graph=False
    )
    assert bool(result.success.item())
    assert result.optimized_plan is not None
    assert result.optimized_plan.position.shape == (16, 7)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_without_cpu_fallback() -> None:
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    model = panda_model(torch.float32, "mps")
    problem = basic_problem(model, max_iterations=1)
    result = optimize_dynamics_aware(problem)
    assert result.position.device.type == "mps"
    assert torch.isfinite(result.objective).all()
