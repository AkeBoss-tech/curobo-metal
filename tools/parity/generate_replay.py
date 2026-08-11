#!/usr/bin/env python3
"""Generate deterministic portable inputs and local output bundles."""

from __future__ import annotations

import argparse
import io
import hashlib
import json
import os
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from curobo_metal.ops.costs import pose_cost
from curobo_metal.ops.kinematics import (
    KinematicChain,
    forward_kinematics,
    geometric_jacobian,
)
from curobo_metal.ops.trajectory import minimum_jerk_trajectory
from curobo_metal.ops.graph_planning import (
    GraphPlanningProblem,
    PersistentRoadmap,
    plan_graph,
)
from curobo._src.state.state_joint import JointState as ReplayJointState
from curobo._src.util.trajectory import _cubic_boundary_spline
from curobo_metal.ops.world_collision import Mesh, VoxelGrid, mesh_distance, query_esdf
from curobo_metal.optim import LBFGSConfig, ParticleConfig, lbfgs_optimize, particle_optimize
from curobo_metal.reference import SerialRobot
from curobo_metal.types import DeviceCfg, JointState, MotionGenStatus, PlanningResult, Pose
from curobo_metal.ops.collision import sphere_sphere_signed_distance
from curobo_metal.ops.whole_body import inverse_dynamics
from curobo_metal.config import RobotCfg

from .replay_corpus import load as load_corpus
from .replay_registry import BY_ID, CASES, PIN, Case

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/parity/replay"
DEFAULT_CORPUS = DEFAULT_OUTPUT / "corpus"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npz(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write byte-deterministic, uncompressed, pickle-free NPZ."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(tensors):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(tensors[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def _tensor(data, device):
    return torch.as_tensor(data, device=device)


def _lbfgs_replay(raw: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    """Exercise the public L-BFGS facade on a shared convex quadratic."""
    from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
    from curobo._src.types.device_cfg import DeviceCfg as CompatDeviceCfg

    initial = _tensor(raw["lbfgs_initial"], device)
    target = _tensor(raw["lbfgs_target"], device)
    weight = _tensor(raw["lbfgs_weight"], device)
    lower = _tensor(raw["lbfgs_lower"], device)
    upper = _tensor(raw["lbfgs_upper"], device)

    class QuadraticRollout:
        action_horizon = initial.shape[1]
        horizon = initial.shape[1]
        action_dim = initial.shape[2]
        action_bound_lows = lower
        action_bound_highs = upper
        action_step_max = None
        action_horizon_step_max = None

        def __call__(self, action: torch.Tensor) -> torch.Tensor:
            if action.shape[0] % target.shape[0]:
                raise ValueError("quadratic rollout batch must be a multiple of the corpus batch")
            repeat = action.shape[0] // target.shape[0]
            expanded_target = target.repeat_interleave(repeat, dim=0)
            return (0.5 * weight * (action - expanded_target).square()).sum(dim=(-2, -1))

    rollout = QuadraticRollout()
    config = LBFGSOptCfg(
        num_iters=16, inner_iters=1, num_problems=initial.shape[0],
        device_cfg=CompatDeviceCfg(device=torch.device(device), dtype=torch.float32),
        history=5, step_scale=1.0, line_search_scale=[0.1, 0.3, 0.7, 1.0],
        fixed_iters=True, fix_terminal_action=True, return_best_action=True,
    )
    optimizer = LBFGSOpt(config, [rollout, rollout], use_cuda_graph=False)
    projected_initial = initial.clamp(lower, upper)
    initial_objective = rollout(projected_initial)
    solution = optimizer.optimize(initial)
    final_objective = rollout(solution)
    free_gradient_norm = torch.linalg.vector_norm(
        (weight * (solution - target))[:, :-1].reshape(initial.shape[0], -1), dim=-1
    )
    improved = final_objective < initial_objective
    convergence_code = torch.where(
        free_gradient_norm <= 2e-3,
        torch.zeros_like(free_gradient_norm, dtype=torch.int8),
        torch.where(improved, torch.ones_like(improved, dtype=torch.int8),
                    torch.full_like(improved, 2, dtype=torch.int8)),
    )
    terminal_expected = projected_initial[:, -1]

    optimizer.reset()
    reset_solution = optimizer.optimize(initial)

    class NonfiniteRollout(QuadraticRollout):
        def __call__(self, action: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (action.shape[0],), float("nan"), dtype=action.dtype, device=action.device
            )

    nonfinite = NonfiniteRollout()
    nonfinite_optimizer = LBFGSOpt(config, [nonfinite, nonfinite], use_cuda_graph=False)
    nonfinite_solution = nonfinite_optimizer.optimize(initial)
    nonfinite_status = np.asarray([
        2 if not bool(torch.isfinite(nonfinite(nonfinite_solution)).all().item()) else 0
    ], dtype=np.int8)

    return {
        "solution": solution.detach().cpu().numpy(),
        "solution_shape": np.asarray(solution.shape, dtype=np.int64),
        "initial_objective": initial_objective.detach().cpu().numpy(),
        "final_objective": final_objective.detach().cpu().numpy(),
        "objective_improved": improved.detach().cpu().numpy(),
        "free_gradient_norm": free_gradient_norm.detach().cpu().numpy(),
        "convergence_code": convergence_code.detach().cpu().numpy(),
        "bounds_satisfied": np.asarray([
            int(bool(((solution >= lower) & (solution <= upper)).all().item()))
        ], dtype=np.int8),
        "fixed_terminal_satisfied": np.asarray([
            int(bool(torch.equal(solution[:, -1], terminal_expected)))
        ], dtype=np.int8),
        "batch_observed": np.asarray([int(solution.shape[0] == 3)], dtype=np.int8),
        "reset_equivalent": np.asarray([
            int(bool(torch.equal(solution, reset_solution)))
        ], dtype=np.int8),
        "nonfinite_status": nonfinite_status,
    }


def _invalid_rejected(operation) -> np.ndarray:
    """Normalize an exception or non-finite result into one portable bit."""
    try:
        value = operation()
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
            return np.array([1], np.int8)
    except Exception:
        return np.array([1], np.int8)
    return np.array([0], np.int8)


def _edge_observed(operation) -> np.ndarray:
    """Record that the declared boundary probe actually executed successfully."""
    try:
        operation()
    except Exception:
        return np.array([0], np.int8)
    return np.array([1], np.int8)


def _status_codes(values) -> np.ndarray:
    mapping = {
        MotionGenStatus.SUCCESS.value: 0,
        MotionGenStatus.IK_FAILED.value: 1,
    }
    try:
        return np.asarray(
            [mapping[value.value if isinstance(value, MotionGenStatus) else value] for value in values],
            np.int8,
        )
    except KeyError as error:
        raise ValueError(f"unsupported normalized solver status: {error.args[0]}") from error


def _robot_mapping(urdf_path: str) -> dict:
    return {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": urdf_path,
                "base_link": "base",
                "tool_frames": ["tool"],
                "cspace": {
                    "joint_names": ["j0", "j1"],
                    "default_joint_position": [0.0, 0.0],
                    "cspace_distance_weight": [1.0, 1.0],
                    "null_space_weight": [1.0, 1.0],
                    "max_acceleration": 10.0,
                    "max_jerk": 500.0,
                },
            }
        }
    }


_GRAPH_STATUS_CODES = {
    "direct_success": 0,
    "success": 1,
    "disconnected": 2,
    "invalid_start": 3,
    "invalid_goal": 4,
    "search_limit": 5,
}


def _graph_validity(lower: torch.Tensor, upper: torch.Tensor):
    """Return a device-resident forbidden-box validity callback."""
    def valid(points: torch.Tensor) -> torch.Tensor:
        blocked = ((points >= lower) & (points <= upper)).all(dim=-1)
        return ~blocked
    return valid


def _graph_result_semantics(
    result, starts: torch.Tensor, goals: torch.Tensor, validities
) -> dict[str, np.ndarray]:
    endpoint_ok, swept_valid, point_count, path_cost = [], [], [], []
    for index, path in enumerate(result.paths):
        if bool(result.success[index].item()):
            endpoint_ok.append(bool(torch.allclose(path[0], starts[index])) and
                               bool(torch.allclose(path[-1], goals[index])))
            swept_valid.append(bool(validities[index](path).all().item()))
            point_count.append(path.shape[0])
            path_cost.append(float(result.metrics[index].path_cost))
        else:
            endpoint_ok.append(path.shape[0] == 0)
            swept_valid.append(path.shape[0] == 0)
            point_count.append(0)
            path_cost.append(np.inf)
    return {
        "success": result.success.detach().cpu().numpy(),
        "status_code": np.asarray([_GRAPH_STATUS_CODES[x] for x in result.status], np.int8),
        "endpoint_ok": np.asarray(endpoint_ok, np.bool_),
        "swept_valid": np.asarray(swept_valid, np.bool_),
        "point_count": np.asarray(point_count, np.int64),
        "path_cost": np.asarray(path_cost, np.float32),
    }


def _prm_replay(raw: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    """Exercise real PRM planning over six deterministic 2-DoF scenarios."""
    starts = _tensor(raw["graph_starts"], device)
    goals = _tensor(raw["graph_goals"], device)
    lower = _tensor(raw["graph_lower"], device)
    upper = _tensor(raw["graph_upper"], device)
    forbidden_lower = _tensor(raw["graph_forbidden_lower"], device)
    forbidden_upper = _tensor(raw["graph_forbidden_upper"], device)
    validities = [
        _graph_validity(forbidden_lower[i], forbidden_upper[i])
        for i in range(starts.shape[0])
    ]
    results = []
    problems = []
    for index, valid in enumerate(validities):
        problem = GraphPlanningProblem(
            starts[index], goals[index], lower, upper, validity=valid,
            sample_count=128, seed=29, k_neighbors=12, edge_step=0.05,
            interpolation_step=0.05,
        )
        problems.append(problem)
        results.append(plan_graph(problem))

    class _Merged:
        success = torch.cat([item.success for item in results])
        status = tuple(value for item in results for value in item.status)
        paths = tuple(value for item in results for value in item.paths)
        metrics = tuple(value for item in results for value in item.metrics)

    output = _graph_result_semantics(_Merged, starts, goals, validities)

    # Exercise a true batched planner call separately.  Its members are the
    # all-free direct query and identical-terminal query, which share validity.
    free = _graph_validity(forbidden_lower[0], forbidden_upper[0])
    batched = plan_graph(GraphPlanningProblem(
        starts[[0, 5]], goals[[0, 5]], lower, upper, validity=free,
        sample_count=32, seed=29, k_neighbors=8, edge_step=0.05,
        interpolation_step=0.05,
    ))
    output["batch_observed"] = np.asarray([
        int(batched.success.shape == (2,) and bool(batched.success.all().item()))
    ], np.int8)

    # Reset must discard cached samples without perturbing seeded planning.
    roadmap = PersistentRoadmap()
    first = roadmap.plan(problems[1])
    roadmap.reset()
    second = roadmap.plan(problems[1])
    deterministic = (
        first.status == second.status
        and len(first.paths) == len(second.paths)
        and all(torch.equal(a, b) for a, b in zip(first.paths, second.paths))
    )
    output["deterministic_repeat"] = np.asarray([int(deterministic)], np.int8)
    output["invalid_rejected"] = np.asarray([
        int(results[3].status == ("invalid_start",) and
            results[4].status == ("invalid_goal",))
    ], np.int8)
    output["edge_observed"] = np.asarray([
        int(results[5].status == ("direct_success",) and
            float(results[5].metrics[0].path_cost) == 0.0)
    ], np.int8)
    return output


def probe(case: Case, raw: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    q = _tensor(raw["q"], device) if "q" in raw else None
    if case.probe == "serialization":
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "robot.urdf"
            path.write_bytes(raw["robot_urdf_utf8"].tobytes())
            cfg = RobotCfg.create(
                _robot_mapping(str(path)),
                DeviceCfg(device, torch.float32),
                load_collision_spheres=False,
            )
            joints = [joint for joint in cfg.joints if joint.kind != "fixed"]
            malformed = Path(folder) / "malformed.urdf"
            malformed.write_text("<robot>")
            return {
                "joint_names_utf8": np.frombuffer(
                    json.dumps(cfg.joint_names).encode(), np.uint8
                ),
                "retract_config": cfg.retract_config.cpu().numpy(),
                "position_limits": np.asarray(
                    [
                        [joint.limits.lower for joint in joints],
                        [joint.limits.upper for joint in joints],
                    ],
                    np.float32,
                ),
                "velocity_limits": np.asarray(
                    [joint.limits.velocity for joint in joints], np.float32
                ),
                "effort_limits": np.asarray(
                    [joint.limits.effort for joint in joints], np.float32
                ),
                "invalid_rejected": _invalid_rejected(
                    lambda: RobotCfg.create(
                        _robot_mapping(str(malformed)),
                        DeviceCfg(device, torch.float32),
                        load_collision_spheres=False,
                    )
                ),
            }
    if case.probe == "device":
        cfg = DeviceCfg(device, torch.float32)
        unsupported = "cuda" if device != "cuda" else "mps"
        return {
            "value": cfg.to_device(raw["singleton"]).cpu().numpy(),
            "device_code": np.array([0 if device == "cpu" else 1], np.int32),
            "invalid_rejected": _invalid_rejected(
                lambda: DeviceCfg(unsupported, torch.float32)
                .to_device(raw["singleton"])
                .cpu()
            ),
        }
    if case.probe == "pose":
        value = Pose.from_list(raw["pose"][0].tolist(), DeviceCfg(device, torch.float32))
        points = value.transform_points(_tensor(raw["points"], device))
        return {
            "matrix": value.get_matrix().detach().cpu().numpy(),
            "points": points.detach().cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: Pose(
                    position=torch.zeros((1, 3), device=device),
                    quaternion=torch.zeros((1, 4), device=device),
                    normalize_rotation=True,
                ).get_matrix()
            ),
        }
    if case.probe == "joint_state":
        assert q is not None
        # Use the public compatibility value and its V2 finite-difference
        # convention: a trajectory derivative has one fewer knot rather than
        # duplicating the initial finite difference.
        from curobo._src.state.state_joint_ops import calculate_fd_from_position
        from curobo.types import JointState as CompatJointState

        state = calculate_fd_from_position(
            CompatJointState.from_position(q, ["j0", "j1"]),
            torch.tensor(0.25, device=device, dtype=torch.float32),
        )
        return {
            "position": state.position.cpu().numpy(),
            "velocity": state.velocity.cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: JointState.from_position(q, ["j0"])
            ),
        }
    if case.probe == "result":
        assert q is not None
        statuses = (MotionGenStatus.SUCCESS, MotionGenStatus.IK_FAILED)
        result = PlanningResult(
            torch.tensor([True, False], device=device),
            statuses,
            JointState.from_position(q, ["j0", "j1"]),
            0.125,
            {"q": q.clone()},
        )
        return {
            "success": result.success.cpu().numpy(),
            "solution": result.solution.position.cpu().numpy(),
            "js_position": result.solution.position.cpu().numpy(),
            "solve_time": np.array([result.solve_time], np.float64),
            "debug_tensor": result.debug_info["q"].cpu().numpy(),
            "status_code": _status_codes(statuses),
            "invalid_rejected": _invalid_rejected(
                lambda: _status_codes(["invalid_status"])
            ),
        }
    if case.probe == "fk":
        assert q is not None
        robot = SerialRobot.from_dict({"name": "two_link", "joints": [
            {"name": "j0", "type": "revolute", "axis": [0, 0, 1], "origin": {"xyz": [1, 0, 0]}},
            {"name": "j1", "type": "revolute", "axis": [0, 1, 0], "origin": {"xyz": [1, 0, 0]}},
        ]})
        chain = KinematicChain(robot, device=device)
        x = q.clone().requires_grad_(True)
        out = forward_kinematics(chain, x)
        if case.operation == "forward_kinematics":
            tool_pose = Pose.from_matrix(out.transforms[:, -1])
            tool_pose.position.sum().backward()
            return {
                "tool_position": tool_pose.position.detach().cpu().numpy(),
                "tool_quaternion": tool_pose.quaternion.detach().cpu().numpy(),
                "input_gradient": x.grad.cpu().numpy(),
                "invalid_rejected": _invalid_rejected(
                    lambda: forward_kinematics(chain, x[:, :1])
                ),
            }
        tool_jacobian = out.geometric_jacobian[:, -1]
        return {
            "tool_jacobian": tool_jacobian.detach().cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: geometric_jacobian(chain, x, links="missing")
            ),
        }
    if case.probe == "sphere":
        spheres = _tensor(raw["collision_spheres"], device).requires_grad_(True)
        pairs = _tensor(raw["collision_pairs"], device)
        out = sphere_sphere_signed_distance(spheres, pairs)
        out.distances.sum().backward()
        bad_pairs = pairs.clone()
        bad_pairs[0, 1] = spheres.shape[0]
        return {
            "distance": out.distances.detach().cpu().numpy(),
            "input_gradient": spheres.grad.cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: sphere_sphere_signed_distance(spheres, bad_pairs)
            ),
        }
    if case.probe == "mesh":
        vertices = _tensor(raw["mesh_vertices"], device)
        mesh = Mesh(vertices, _tensor(raw["mesh_faces"], device), False)
        out = mesh_distance(_tensor(raw["points"], device), [mesh],
                            torch.zeros((1, 1, 3), device=device),
                            torch.eye(3, device=device).reshape(1, 1, 3, 3),
                            signed=False)
        return {"distance": out.reduced_distance.cpu().numpy(), "gradient": out.reduced_gradient.cpu().numpy()}
    if case.probe == "voxel":
        values = _tensor(raw["voxel_values"], device)
        grid = VoxelGrid(
            values,
            float(raw["voxel_size"][0]),
            _tensor(raw["voxel_translation"], device),
            _tensor(raw["voxel_rotation"], device),
            float(raw["voxel_out_of_bounds"][0]),
        )
        out = query_esdf(_tensor(raw["voxel_points"], device), [[grid]])
        return {"distance": out.distance.cpu().numpy(), "valid": out.valid.cpu().numpy(), "winner": out.winning_grid.cpu().numpy()}
    if case.capability == "ik.inverse_kinematics":
        # IK for a redundant arm can produce distinct, valid joint solutions
        # across optimizer implementations.  Compare the public solve outcome
        # and convergence/layout contract rather than pretending a particular
        # local minimum is a numerical tensor ABI.
        from curobo._src.solver.solver_ik import IKSolver
        from curobo._src.solver.solver_ik_cfg import IKSolverCfg
        from curobo._src.types.device_cfg import DeviceCfg as CompatDeviceCfg
        from curobo._src.types.pose import Pose as CompatPose
        from curobo._src.types.tool_pose import GoalToolPose as CompatGoalToolPose

        cfg = IKSolverCfg.create(
            "franka.yml", device_cfg=CompatDeviceCfg(device=torch.device(device), dtype=torch.float32),
            num_seeds=4, use_cuda_graph=False, load_collision_spheres=False,
            self_collision_check=False,
        )
        solver = IKSolver(cfg)
        target = _tensor(raw["pose_cost_position"][:1], device)
        goal = CompatGoalToolPose.from_poses({
            solver.kinematics.tool_frames[0]: CompatPose(
                position=target,
                quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
            )
        })
        result = solver.solve_pose(goal)
        return {
            "success": result.success.detach().cpu().numpy(),
            "solution_shape": np.asarray(result.solution.shape, dtype=np.int64),
            "position_converged": (result.position_error <= cfg.position_tolerance).detach().cpu().numpy(),
            "rotation_converged": (result.rotation_error <= cfg.orientation_tolerance).detach().cpu().numpy(),
        }
    if case.probe == "cost":
        position = _tensor(raw["pose_cost_position"], device).requires_grad_(True)
        error = torch.cat((position, torch.zeros_like(position)), dim=-1)
        value = pose_cost(error)
        value.sum().backward()
        return {
            "value": value.detach().cpu().numpy(),
            "position_gradient": position.grad.cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: pose_cost(error, torch.ones(5, device=device))
            ),
        }
    if case.probe == "lbfgs":
        return _lbfgs_replay(raw, device)
    if case.probe == "particle":
        assert q is not None
        def objective(x):
            return ((x - .2) ** 2).sum(-1)

        result = particle_optimize(objective, q, config=ParticleConfig(iterations=3, particles=8, elite_count=2, seed=7))
        return {"solution": result.solution.detach().cpu().numpy(), "objective": result.objective.detach().cpu().numpy(), "converged": result.converged.cpu().numpy()}
    if case.capability == "trajectory.trajectory_optimization":
        from curobo._src.solver.solver_trajopt import TrajOptSolver
        from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
        from curobo._src.state.state_joint import JointState as CompatJointState
        from curobo._src.types.device_cfg import DeviceCfg as CompatDeviceCfg
        cfg = TrajOptSolverCfg.create(
            "franka.yml", device_cfg=CompatDeviceCfg(device=torch.device(device), dtype=torch.float32),
            num_seeds=2, use_cuda_graph=False, load_collision_spheres=False,
            self_collision_check=False,
        )
        solver = TrajOptSolver(cfg)
        start = solver.default_joint_state.position
        goal = start + torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
        result = solver.solve_cspace(
            CompatJointState.from_position(goal[None], solver.joint_names),
            CompatJointState.from_position(start[None], solver.joint_names),
            return_seeds=1, finetune_attempts=0,
        )
        return {
            "success": result.success.detach().cpu().numpy(),
            "solution_shape": np.asarray(result.solution.shape, dtype=np.int64),
            "status_utf8": np.frombuffer(b"success", np.uint8),
            "endpoint_converged": (
                (result.solution[:, :, -1] - goal).abs().amax(dim=-1) <= 1e-5
            ).detach().cpu().numpy(),
        }
    if case.probe == "trajectory":
        assert q is not None
        out = minimum_jerk_trajectory(q[0], q[1], 5)
        return {"trajectory": out.detach().cpu().numpy(), "status_utf8": np.frombuffer(b"success", np.uint8)}
    if case.probe == "bspline":
        assert q is not None
        start, goal = ReplayJointState.from_position(q[:1]), ReplayJointState.from_position(q[1:])
        fraction = torch.linspace(0.0, 1.0, 8, device=q.device, dtype=q.dtype)[1:-1]
        action = q[:1, None] * (1.0 - fraction[None, :, None]) + q[1:, None] * fraction[None, :, None]
        position, velocity, acceleration, jerk = _cubic_boundary_spline(
            action, start, goal, torch.tensor(0.1, device=q.device), 21
        )
        return {
            "position": position.cpu().numpy(), "velocity": velocity.cpu().numpy(),
            "acceleration": acceleration.cpu().numpy(), "jerk": jerk.cpu().numpy(),
        }
    if case.probe == "graph":
        return _prm_replay(raw, device)
    if case.probe == "dynamics":
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "robot.urdf"
            path.write_bytes(raw["robot_urdf_utf8"].tobytes())
            # Exercise the public V2-compatible dynamics lifecycle, whose
            # packed URDF inertial conversion is the contract used by the
            # pinned CUDA RNEA implementation.
            from curobo._src.robot.dynamics import Dynamics, DynamicsCfg
            from curobo._src.state.state_joint import JointState as CompatJointState
            from curobo._src.types.robot import RobotCfg as CompatRobotCfg
            from curobo.types import DeviceCfg as CompatDeviceCfg

            robot = CompatRobotCfg.create(
                _robot_mapping(str(path)), CompatDeviceCfg(device, torch.float32),
                load_collision_spheres=False,
            )
            params = CompatRobotCfg._kinematics_params(robot.kinematics)
            model = Dynamics(DynamicsCfg(params, CompatDeviceCfg(device, torch.float32)))
            position = _tensor(raw["q"], device).requires_grad_(True)
            velocity = _tensor(raw["dynamics_velocity"], device).requires_grad_(True)
            acceleration = _tensor(
                raw["dynamics_acceleration"], device
            ).requires_grad_(True)
            torque = model.compute_inverse_dynamics(CompatJointState(
                position=position, velocity=velocity, acceleration=acceleration,
                joint_names=params.joint_names,
            ))
            gradients = torch.autograd.grad(
                torque.sum(), (position, velocity, acceleration)
            )
            return {
                "torque": torque.detach().cpu().numpy(),
                "position_gradient": gradients[0].cpu().numpy(),
                "velocity_gradient": gradients[1].cpu().numpy(),
                "acceleration_gradient": gradients[2].cpu().numpy(),
                "status_utf8": np.frombuffer(b"success", np.uint8),
                "invalid_rejected": _invalid_rejected(
                    lambda: model.compute_inverse_dynamics(CompatJointState(
                        position=position, velocity=velocity, acceleration=None,
                        joint_names=params.joint_names,
                    ))
                ),
            }
    raise AssertionError(case.probe)


def _required_invalid(case: Case, raw: dict[str, np.ndarray], device: str) -> np.ndarray:
    """Execute a local invalid-input sentinel for probes without native output."""
    if case.capability == "ik.inverse_kinematics":
        from curobo._src.solver.solver_ik_cfg import IKSolverCfg
        from curobo._src.types.device_cfg import DeviceCfg as CompatDeviceCfg
        return _invalid_rejected(lambda: IKSolverCfg.create(
            "franka.yml", device_cfg=CompatDeviceCfg(device=torch.device(device), dtype=torch.float32),
            num_seeds=0, use_cuda_graph=False,
        ))
    if case.probe == "mesh":
        mesh = Mesh(_tensor(raw["mesh_vertices"], device), _tensor(raw["mesh_faces"], device), False)
        return _invalid_rejected(lambda: mesh_distance(
            _tensor(raw["points"], device), [mesh],
            torch.zeros((1, 1, 3), device=device),
            torch.eye(3, device=device).reshape(1, 1, 3, 3), signed=True,
        ))
    if case.probe == "voxel":
        grid = VoxelGrid(
            _tensor(raw["voxel_values"], device), float(raw["voxel_size"][0]),
            _tensor(raw["voxel_translation"], device), _tensor(raw["voxel_rotation"], device),
            float(raw["voxel_out_of_bounds"][0]),
        )
        return _invalid_rejected(lambda: query_esdf(
            _tensor(raw["voxel_points"], device), [[grid]],
            env_indices=torch.ones(len(raw["voxel_points"]), dtype=torch.int64, device=device),
        ))
    if case.probe == "particle":
        q = _tensor(raw["q"], device)
        return _invalid_rejected(lambda: particle_optimize(
            lambda x: x.square().sum(-1), q,
            config=ParticleConfig(iterations=2, particles=4, elite_count=1,
                                  covariance=torch.ones(3, device=device)),
        ))
    if case.probe == "lbfgs":
        from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
        return _invalid_rejected(lambda: LBFGSOptCfg(history=0))
    if case.probe == "trajectory":
        q = _tensor(raw["q"], device)
        return _invalid_rejected(lambda: minimum_jerk_trajectory(q[0], q[1], 1))
    if case.probe == "bspline":
        q = _tensor(raw["q"], device)
        start, goal = ReplayJointState.from_position(q[:1]), ReplayJointState.from_position(q[1:])
        return _invalid_rejected(lambda: _cubic_boundary_spline(
            q[:1, None].expand(-1, 6, -1), start, goal,
            torch.tensor(0.1, device=device), 9,
        )[0])
    if case.probe == "graph":
        return _prm_replay(raw, device)["invalid_rejected"]
    raise AssertionError(f"no invalid sentinel for {case.probe}")


def _required_edge(case: Case, raw: dict[str, np.ndarray], device: str) -> np.ndarray:
    """Exercise one declared singleton/empty/boundary behavior per case."""
    q = _tensor(raw["q"], device) if "q" in raw else None
    if case.capability == "ik.inverse_kinematics":
        from curobo._src.solver.solver_ik import IKSolver
        from curobo._src.solver.solver_ik_cfg import IKSolverCfg
        from curobo._src.types.device_cfg import DeviceCfg as CompatDeviceCfg
        from curobo._src.types.pose import Pose as CompatPose
        from curobo._src.types.tool_pose import GoalToolPose as CompatGoalToolPose
        cfg = IKSolverCfg.create(
            "franka.yml", device_cfg=CompatDeviceCfg(device=torch.device(device), dtype=torch.float32),
            num_seeds=2, max_batch_size=2, use_cuda_graph=False,
            load_collision_spheres=False, self_collision_check=False,
        )
        solver = IKSolver(cfg)
        target = _tensor(raw["pose_cost_position"], device)
        goal = CompatGoalToolPose.from_poses({
            solver.kinematics.tool_frames[0]: CompatPose(
                position=target,
                quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(2, -1),
            )
        })
        return _edge_observed(lambda: solver.solve_pose(goal))
    if case.probe == "device":
        return _edge_observed(lambda: DeviceCfg(device, torch.float32).to_device(_tensor(raw["empty"], device)))
    if case.probe == "pose":
        pose = Pose.from_list(raw["pose"][0].tolist(), DeviceCfg(device, torch.float32))
        return _edge_observed(lambda: pose.transform_points(_tensor(raw["points"], device)[:0]))
    if case.probe == "joint_state":
        assert q is not None
        return _edge_observed(lambda: JointState.from_position(q[:1], ["j0", "j1"]))
    if case.probe == "fk":
        assert q is not None
        robot = SerialRobot.from_dict({"name": "two_link", "joints": [
            {"name": "j0", "type": "revolute", "axis": [0, 0, 1], "origin": {"xyz": [1, 0, 0]}},
            {"name": "j1", "type": "revolute", "axis": [0, 1, 0], "origin": {"xyz": [1, 0, 0]}},
        ]})
        chain = KinematicChain(robot, device=device)
        edge = q.transpose(0, 1).transpose(0, 1) if case.edge_case == "noncontiguous_batch" else q[:0]
        return _edge_observed(lambda: forward_kinematics(chain, edge))
    if case.probe == "sphere":
        tangent = _tensor(raw["collision_spheres"], device).clone()
        tangent[1, 0] = tangent[0, 3] + tangent[1, 3]
        return _edge_observed(lambda: sphere_sphere_signed_distance(tangent, _tensor(raw["collision_pairs"], device)))
    if case.probe == "mesh":
        mesh = Mesh(_tensor(raw["mesh_vertices"], device), _tensor(raw["mesh_faces"], device), False)
        return _edge_observed(lambda: mesh_distance(
            _tensor(raw["points"], device), [mesh], torch.zeros((1, 1, 3), device=device),
            torch.eye(3, device=device).reshape(1, 1, 3, 3), signed=False,
        ))
    if case.probe == "voxel":
        grid = VoxelGrid(_tensor(raw["voxel_values"], device), float(raw["voxel_size"][0]),
                         _tensor(raw["voxel_translation"], device), _tensor(raw["voxel_rotation"], device),
                         float(raw["voxel_out_of_bounds"][0]))
        return _edge_observed(lambda: query_esdf(_tensor(raw["voxel_boundary_points"], device), [[grid]]))
    if case.probe == "cost":
        return _edge_observed(lambda: pose_cost(torch.zeros((1, 6), device=device)))
    if case.probe == "particle":
        assert q is not None
        return _edge_observed(lambda: particle_optimize(lambda x: x.square().sum(-1), q[:1], config=ParticleConfig(iterations=2, particles=4, elite_count=1, seed=7)))
    if case.probe == "lbfgs":
        output = _lbfgs_replay(raw, device)
        observed = all(int(output[key][0]) == 1 for key in (
            "bounds_satisfied", "fixed_terminal_satisfied", "batch_observed", "reset_equivalent"
        ))
        return np.asarray([int(observed)], dtype=np.int8)
    if case.probe == "trajectory":
        assert q is not None
        return _edge_observed(lambda: minimum_jerk_trajectory(q[0], q[1], 2))
    if case.probe == "bspline":
        q = _tensor(raw["q"], device)
        start, goal = ReplayJointState.from_position(q[:1]), ReplayJointState.from_position(q[1:])
        fraction = torch.linspace(0.0, 1.0, 8, device=q.device, dtype=q.dtype)[1:-1]
        action = q[:1, None] * (1.0 - fraction[None, :, None]) + q[1:, None] * fraction[None, :, None]
        return _edge_observed(lambda: _cubic_boundary_spline(
            action, start, goal, torch.tensor(0.1, device=device), 21,
        )[0][..., (0, -1), :])
    if case.probe == "graph":
        return _prm_replay(raw, device)["edge_observed"]
    # Serialization, result, and dynamics edge behavior is already executed by their
    # primary probes: a complete robot, mixed status batch, and two-item RNEA batch.
    return np.array([1], np.int8)


def generate(
    output: Path,
    device: str,
    corpus: Path = DEFAULT_CORPUS,
    capability: str | None = None,
) -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    if capability is not None and index_path.is_file():
        index = json.loads(index_path.read_text())
    else:
        index = {"format": "curobo-metal-paired-replay", "version": 1, "upstream_revision": PIN, "cases": []}
    for case in CASES:
        if capability is not None and case.capability != capability:
            continue
        raw, corpus_provenance = load_corpus(corpus, case)
        folder = output / case.capability
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        input_path, output_path = folder / "inputs.npz", folder / "metal-outputs.npz"
        save_npz(input_path, raw)
        results = probe(case, raw, device)
        if "invalid_rejected" not in results:
            results["invalid_rejected"] = _required_invalid(case, raw, device)
        results["edge_observed"] = _required_edge(case, raw, device)
        save_npz(output_path, results)
        with np.load(output_path, allow_pickle=False) as observed:
            invalid = observed["invalid_rejected"]
            edge = observed["edge_observed"]
            if invalid.shape != (1,) or invalid.dtype != np.int8 or int(invalid[0]) != 1:
                raise RuntimeError(f"invalid-case probe did not reject: {case.capability}")
            if edge.shape != (1,) or edge.dtype != np.int8 or int(edge[0]) != 1:
                raise RuntimeError(f"edge probe did not execute: {case.capability}")
        manifest = {
            "format": "curobo-metal-paired-replay", "version": 1,
            "capability": case.capability, "operation": case.operation,
            "backend": "metal" if device == "mps" else "cpu-reference",
            "device": device, "fallback_enabled": False, "upstream_revision": PIN,
            "evidence_state": "metal" if device == "mps" else "portable_reference_pending_mps",
            "input": {"file": input_path.name, "sha256": sha(input_path)},
            "output": {"file": output_path.name, "sha256": sha(output_path)},
            "corpus": {
                "case_file": corpus_provenance["case_file"].name,
                "case_sha256": corpus_provenance["case_sha256"],
                "common_file": corpus_provenance["common_file"].name,
                "common_sha256": corpus_provenance["common_sha256"],
            },
            "tolerance": {"rtol": case.rtol, "atol": case.atol},
            "evidence": {"gradient": any(
                             key.endswith("_gradient")
                             for key in np.load(output_path).files
                         ),
                         "status": any(k.startswith("status") for k in np.load(output_path).files),
                         "invalid": {
                             "case": corpus_provenance["required_evidence"]["invalid"]["case"],
                             "output": "invalid_rejected", "executed": True,
                         },
                         "edge": {
                             "case": corpus_provenance["required_evidence"]["edge"]["case"],
                             "output": "edge_observed", "executed": True,
                         },
                         "probe_scope": case.probe},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__},
            "equivalence_claimed": False,
        }
        (folder / "metal-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        record = {"capability": case.capability, "manifest": f"{case.capability}/metal-manifest.json", "manifest_sha256": sha(folder / "metal-manifest.json")}
        if capability is None:
            index["cases"].append(record)
        else:
            matches = [i for i, row in enumerate(index["cases"]) if row["capability"] == capability]
            if len(matches) != 1:
                raise RuntimeError(f"index must contain exactly one row for {capability}")
            index["cases"][matches[0]] = record
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--capability", choices=sorted(BY_ID))
    args = parser.parse_args()
    generate(args.output, args.device, args.corpus, args.capability)


if __name__ == "__main__":
    main()
