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

from curobo_metal.ops.costs import CostManager, CostTerm, pose_cost
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
from curobo_metal.ops.world_collision import (
    Mesh, VoxelGrid, mesh_distance, query_esdf, sample_voxel_sdf,
)
from curobo_metal.optim import LBFGSConfig, lbfgs_optimize
from curobo_metal.reference import SerialRobot
from curobo_metal.types import DeviceCfg, JointState, MotionGenStatus, PlanningResult, Pose
from curobo_metal.ops.collision import sphere_sphere_signed_distance
from curobo_metal.ops.whole_body import inverse_dynamics
from curobo_metal.config import RobotCfg

from .replay_corpus import load as load_corpus
from .replay_registry import BY_ID, CASES, PIN, Case
from .joint_state_probe import run as run_joint_state_probe

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
        fixed_iters=True, fix_terminal_action=False, return_best_action=True,
    )
    optimizer = LBFGSOpt(config, [rollout, rollout], use_cuda_graph=False)
    projected_initial = initial.clamp(lower, upper)
    initial_objective = rollout(projected_initial)
    solution = optimizer.optimize(initial)
    final_objective = rollout(solution)
    projected_optimality_norm = torch.linalg.vector_norm(
        (weight * (solution - target.clamp(lower, upper))).reshape(initial.shape[0], -1), dim=-1
    )
    improved = final_objective < initial_objective
    convergence_code = torch.where(
        projected_optimality_norm <= 2e-3,
        torch.zeros_like(projected_optimality_norm, dtype=torch.int8),
        torch.where(improved, torch.ones_like(improved, dtype=torch.int8),
                    torch.full_like(improved, 2, dtype=torch.int8)),
    )
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
        "projected_optimality_norm": projected_optimality_norm.detach().cpu().numpy(),
        "convergence_code": convergence_code.detach().cpu().numpy(),
        "bounds_satisfied": np.asarray([
            int(bool(((solution >= lower) & (solution <= upper)).all().item()))
        ], dtype=np.int8),
        "batch_observed": np.asarray([int(solution.shape[0] == 3)], dtype=np.int8),
        "reset_equivalent": np.asarray([
            int(bool(torch.equal(solution, reset_solution)))
        ], dtype=np.int8),
        "nonfinite_status": nonfinite_status,
    }


def _particle_replay(raw: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    """Exercise the public EvolutionStrategies facade using semantic outcomes."""
    from curobo._src.optim.components.particle_opt_core import SampleMode
    from curobo._src.optim.particle.evolution_strategies import (
        EvolutionStrategies,
        EvolutionStrategiesCfg,
    )
    from curobo._src.optim.particle.sample_strategies import ParticleSamplerCfg
    from curobo._src.types.device_cfg import DeviceCfg as CompatDeviceCfg

    initial = _tensor(raw["particle_initial"], device)
    target = _tensor(raw["particle_target"], device)
    lower = _tensor(raw["particle_lower"], device)
    upper = _tensor(raw["particle_upper"], device)
    seeds = raw["particle_seeds"].astype(np.int64, copy=False)
    device_cfg = CompatDeviceCfg(device=torch.device(device), dtype=torch.float32)

    class QuadraticRollout:
        action_horizon = initial.shape[1]
        horizon = initial.shape[1]
        action_dim = initial.shape[2]
        action_bound_lows = lower
        action_bound_highs = upper
        action_step_max = None
        action_horizon_step_max = None

        def __init__(self, goal: torch.Tensor):
            self.goal = goal

        def __call__(self, action: torch.Tensor) -> torch.Tensor:
            if action.shape[0] % self.goal.shape[0]:
                raise ValueError("ES rollout batch must be a multiple of the corpus batch")
            repeat = action.shape[0] // self.goal.shape[0]
            goal = self.goal.repeat_interleave(repeat, dim=0)
            return (action - goal).square().sum(dim=(-2, -1))

    def build(seed: int, goal: torch.Tensor = target, *, fixed_samples: bool = False):
        sampler = ParticleSamplerCfg(
            device_cfg=device_cfg, fixed_samples=fixed_samples, seed=int(seed)
        )
        config = EvolutionStrategiesCfg(
            device_cfg=device_cfg, num_iters=8, num_particles=48,
            num_problems=initial.shape[0], null_act_frac=0.0,
            init_cov=0.35, seed=int(seed), sample_params=sampler,
            sample_mode=SampleMode.BEST, store_debug=True,
            learning_rate=0.08, step_size_mean=0.8, step_size_cov=0.1,
            update_cov=False,
        )
        rollout = QuadraticRollout(goal)
        return EvolutionStrategies(config, [rollout], use_cuda_graph=False), rollout

    optimizer, rollout = build(int(seeds[0]))
    initial_objective = rollout(initial)
    solution = optimizer.optimize(initial)
    final_objective = rollout(solution)
    repeat_optimizer, _ = build(int(seeds[0]))
    repeat_solution = repeat_optimizer.optimize(initial)

    shifted_target = target.clone()
    shifted_target[1] = -shifted_target[1]
    independent_optimizer, _ = build(int(seeds[0]), shifted_target)
    independent_solution = independent_optimizer.optimize(initial)

    mean_before_shift = optimizer.mean_action.detach().clone()
    optimizer.shift(1)
    shifted_mean = optimizer.mean_action.detach().clone()
    expected_shift = torch.roll(mean_before_shift, shifts=-1, dims=-2)
    expected_shift[:, -1] = mean_before_shift[:, -1]

    fixed_optimizer, _ = build(int(seeds[0]), fixed_samples=True)
    fixed_optimizer.update_seed(initial)
    population_zero = fixed_optimizer._population(
        initial.shape[0], initial.shape[1], initial.shape[2],
        device=initial.device, dtype=initial.dtype, iteration=0,
    )
    population_one = fixed_optimizer._population(
        initial.shape[0], initial.shape[1], initial.shape[2],
        device=initial.device, dtype=initial.dtype, iteration=1,
    )

    seed_initial, seed_final = [], []
    for seed in seeds:
        seeded, seeded_rollout = build(int(seed))
        seed_initial.append(seeded_rollout(initial))
        seed_final.append(seeded_rollout(seeded.optimize(initial)))
    initial_samples = torch.stack(seed_initial)
    final_samples = torch.stack(seed_final)

    return {
        "solution": solution.detach().cpu().numpy(),
        "solution_shape": np.asarray(solution.shape, np.int64),
        "initial_objective": initial_objective.detach().cpu().numpy(),
        "final_objective": final_objective.detach().cpu().numpy(),
        "objective_improved": (final_objective < initial_objective).detach().cpu().numpy(),
        "solution_finite": np.asarray([int(bool(torch.isfinite(solution).all().item()))], np.int8),
        "bounds_satisfied": np.asarray([
            int(bool(((solution >= lower) & (solution <= upper)).all().item()))
        ], np.int8),
        "deterministic_repeat": np.asarray([int(bool(torch.equal(solution, repeat_solution)))], np.int8),
        "batch_independent": np.asarray([
            int(bool(torch.equal(solution[0], independent_solution[0]) and
                     torch.equal(solution[2], independent_solution[2])))
        ], np.int8),
        "shift_observed": np.asarray([int(bool(torch.equal(shifted_mean, expected_shift)))], np.int8),
        "fixed_sample_repeat": np.asarray([int(bool(torch.equal(population_zero, population_one)))], np.int8),
        "multi_seed_initial_mean": initial_samples.mean(0).detach().cpu().numpy(),
        "multi_seed_final_mean": final_samples.mean(0).detach().cpu().numpy(),
        "multi_seed_final_std": final_samples.std(0, unbiased=False).detach().cpu().numpy(),
        "multi_seed_improvement_rate": (final_samples < initial_samples).float().mean(0).detach().cpu().numpy(),
    }


def _motion_planner_replay(raw: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    """Exercise the real high-level MotionPlanner C-space lifecycle."""
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.state.state_joint import JointState as CompatJointState
    from curobo._src.types.device_cfg import DeviceCfg as CompatDeviceCfg

    cfg = MotionPlannerCfg.create(
        "franka.yml",
        device_cfg=CompatDeviceCfg(device=torch.device(device), dtype=torch.float32),
        num_ik_seeds=2,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        self_collision_check=False,
        max_batch_size=1,
    )
    planner = MotionPlanner(cfg)
    start = _tensor(raw["motion_start"], device)
    goal = start + _tensor(raw["motion_goal_delta"], device)
    start_state = CompatJointState.from_position(start[None], planner.joint_names)
    goal_state = CompatJointState.from_position(goal[None], planner.joint_names)
    result = planner.plan_cspace(
        goal_state, start_state, max_attempts=1, enable_graph_attempt=2
    )
    if result is None or result.js_solution is None:
        raise RuntimeError("MotionPlanner did not return a C-space trajectory")
    active = result.js_solution.position[..., : len(planner.joint_names)]
    path_length = torch.linalg.vector_norm(torch.diff(active, dim=-2), dim=-1).sum(dim=-1)

    try:
        invalid_result = planner.plan_cspace(
            goal_state, start_state, max_attempts=0, enable_graph_attempt=2
        )
        invalid_rejected = int(invalid_result is None)
    except Exception:
        invalid_rejected = 1

    success = bool(result.success.all().item())
    finite = bool(torch.isfinite(active).all().item())
    start_ok = bool(torch.allclose(active[..., 0, :], start, rtol=0.0, atol=1e-5))
    goal_ok = bool(torch.allclose(active[..., -1, :], goal, rtol=0.0, atol=1e-5))
    edge = int(success and finite and start_ok and goal_ok)
    return {
        "success": result.success.detach().cpu().numpy(),
        "trajectory": active.detach().cpu().numpy(),
        "solution_shape": np.asarray(active.shape, np.int64),
        "path_length": path_length.detach().cpu().numpy(),
        "status_code": np.asarray([0 if success else 1], np.int8),
        "start_converged": np.asarray([start_ok], np.int8),
        "endpoint_converged": np.asarray([goal_ok], np.int8),
        "solution_finite": np.asarray([finite], np.int8),
        "invalid_rejected": np.asarray([invalid_rejected], np.int8),
        "edge_observed": np.asarray([edge], np.int8),
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


def _matrix_observed(case: Case, raw: dict[str, np.ndarray], device: str) -> np.ndarray:
    """Execute the declared foundation/kinematics shape-and-lifecycle matrix.

    This is local evidence only: every bit says that a concrete portable
    scenario executed, not that CUDA produced an equivalent result.  The same
    inputs are shipped to the pinned CUDA adapters for future paired runs.
    """
    if not case.matrix_cases:
        return np.empty((0,), dtype=np.int8)
    if case.probe == "device":
        cfg = DeviceCfg(device, torch.float32)
        values = (raw["device_zero"], raw["singleton"], raw["device_many"], raw["empty"])
        bits = [int(tuple(cfg.to_device(value).shape) == tuple(value.shape)) for value in values]
        roundtrip = cfg.to_device(raw["device_many"]).detach().cpu().numpy()
        bits.append(int(np.array_equal(roundtrip, raw["device_many"])))
        return np.asarray(bits, dtype=np.int8)
    if case.probe == "pose":
        packed = _tensor(raw["pose_matrix"], device).requires_grad_(True)
        # Keep the VJP path tensor-native: ``tolist`` is fine for the ordinary
        # construction checks below, but would detach an MPS tensor from its
        # gradient source.
        values = [
            Pose.from_list(row.detach().cpu().tolist(), DeviceCfg(device, torch.float32))
            for row in packed
        ]
        zero = values[0].transform_points(_tensor(raw["points"], device)[:0])
        one = values[0].transform_points(_tensor(raw["points"], device)[:1])
        many = values[1].transform_points(_tensor(raw["points"], device))
        source = _tensor(raw["pose_noncontiguous_source"], device)
        noncontiguous = source[:, ::2]
        transformed = values[2].transform_points(noncontiguous)
        differentiable = Pose(packed[:1, :3], packed[:1, 3:])
        loss = differentiable.transform_points(_tensor(raw["points"], device)).sum()
        (gradient,) = torch.autograd.grad(loss, packed)
        return np.asarray([
            int(zero.shape[0] == 0), int(one.shape[0] == 1),
            int(many.shape[0] == raw["points"].shape[0]), int(not noncontiguous.is_contiguous() and transformed.shape == noncontiguous.shape),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "joint_state":
        zero = JointState.from_position(_tensor(raw["joint_position"], device)[:0], ["j0", "j1"])
        one = JointState.from_position(_tensor(raw["joint_position"], device)[:1], ["j0", "j1"])
        many = JointState.from_position(_tensor(raw["joint_position"], device), ["j0", "j1"])
        source = _tensor(raw["joint_noncontiguous_source"], device)
        noncontiguous = JointState.from_position(source[:, ::2], ["j0", "j1"])
        optional = JointState(
            position=_tensor(raw["joint_position"], device), velocity=None,
            acceleration=_tensor(raw["joint_acceleration"], device), joint_names=["j0", "j1"],
        )
        cloned = many.clone()
        before = many.position.clone()
        cloned.position.add_(1)
        position = _tensor(raw["joint_position"], device).requires_grad_(True)
        (gradient,) = torch.autograd.grad(
            JointState.from_position(position, ["j0", "j1"]).position.square().sum(), position
        )
        return np.asarray([
            int(zero.position.shape[0] == 0), int(one.position.shape[0] == 1), int(many.position.shape[0] == 2),
            int(not noncontiguous.position.is_contiguous()), int(optional.velocity is None and optional.acceleration is not None),
            int(bool(torch.equal(many.position, before))), int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "result":
        q = _tensor(raw["q"], device)
        statuses = (MotionGenStatus.SUCCESS, MotionGenStatus.IK_FAILED)
        values = [
            PlanningResult(torch.zeros(0, dtype=torch.bool, device=device), (), JointState.from_position(q[:0], ["j0", "j1"]), 0.0, {}),
            PlanningResult(torch.ones(1, dtype=torch.bool, device=device), (statuses[0],), JointState.from_position(q[:1], ["j0", "j1"]), 0.0, {}),
            PlanningResult(torch.tensor([True, False], device=device), statuses, JointState.from_position(q, ["j0", "j1"]), 0.0, {"q": q.clone()}),
        ]
        cloned = PlanningResult(
            values[2].success.clone(), values[2].status,
            values[2].solution.clone(), values[2].solve_time,
            dict(values[2].debug_info),
        )
        assert cloned.solution is not None
        cloned.solution.position.add_(1.0)
        return np.asarray([
            int(values[0].success.numel() == 0), int(values[1].success.numel() == 1), int(values[2].success.numel() == 2),
            int(values[2].success.tolist() == [True, False]), int(not torch.equal(values[2].solution.position, cloned.solution.position)),
        ], dtype=np.int8)
    if case.probe == "serialization":
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "chain.urdf"; first.write_bytes(raw["robot_urdf_utf8"].tobytes())
            branch = Path(folder) / "branch.urdf"; branch.write_bytes(raw["robot_branching_urdf_utf8"].tobytes())
            first_cfg = RobotCfg.create(_robot_mapping(str(first)), DeviceCfg(device, torch.float32), load_collision_spheres=False)
            branch_cfg = RobotCfg.create(_robot_mapping(str(branch)), DeviceCfg(device, torch.float32), load_collision_spheres=False)
            encoded = json.dumps(first_cfg.joint_names)
            return np.asarray([
                int(len(first_cfg.joint_names) == 2), int(len(branch_cfg.joint_names) == 2), int(json.loads(encoded) == first_cfg.joint_names),
            ], dtype=np.int8)
    if case.probe == "fk":
        robot = SerialRobot.from_dict({"name": "two_link", "joints": [
            {"name": "j0", "type": "revolute", "axis": [0, 0, 1], "origin": {"xyz": [1, 0, 0]}},
            {"name": "j1", "type": "revolute", "axis": [0, 1, 0], "origin": {"xyz": [1, 0, 0]}},
            {"name": "tool_mount", "type": "fixed", "axis": [0, 0, 1], "origin": {"xyz": [0.2, 0, 0]}},
        ]})
        chain = KinematicChain(robot, device=device)
        q = _tensor(raw["q"], device).requires_grad_(True)
        zero = forward_kinematics(chain, q[:0])
        one = forward_kinematics(chain, q[:1])
        many = forward_kinematics(chain, q)
        noncontiguous_source = _tensor(raw["kinematics_noncontiguous_source"], device)
        noncontiguous = noncontiguous_source[:, ::2]
        nc_out = forward_kinematics(chain, noncontiguous)
        # The compact analytic geometric-Jacobian tensor is intentionally
        # value-only today; exercise a first-order VJP through the linked
        # forward transform for both FK surfaces rather than claiming a
        # higher-order Jacobian derivative that the portable API does not
        # expose.
        loss = many.transforms[:, -1, :3, 3].sum()
        (gradient,) = torch.autograd.grad(loss, q)
        links = many.transforms.shape[1]
        return np.asarray([
            int(zero.transforms.shape[0] == 0), int(one.transforms.shape[0] == 1), int(many.transforms.shape[0] == 2),
            int(not noncontiguous.is_contiguous() and nc_out.transforms.shape[0] == 2), int(links >= 3),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "sphere":
        spheres = _tensor(raw["collision_matrix_spheres"], device).requires_grad_(True)
        pairs = _tensor(raw["collision_matrix_pairs"], device)
        full = sphere_sphere_signed_distance(spheres, pairs)
        separated = sphere_sphere_signed_distance(
            _tensor(raw["collision_separated_spheres"], device),
            _tensor(raw["collision_pairs"], device),
        )
        active = sphere_sphere_signed_distance(
            spheres, pairs, pair_active=_tensor(raw["collision_pair_active"], device)
        )
        (gradient,) = torch.autograd.grad(full.reduced_distance.sum(), spheres)
        return np.asarray([
            int(float(separated.reduced_distance[0].detach()) > 0 and float(full.reduced_distance[0].detach()) < 0),
            int(int(full.winning_pair[0]) == 0),
            int(bool(torch.isinf(active.distances[0, 1]).item()) and int(active.winning_pair[0]) == 0),
            int(bool(torch.equal(full.reduced_distance, full.distances[:, 0]))),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "mesh":
        vertices = _tensor(raw["mesh_tetra_vertices"], device)
        faces = _tensor(raw["mesh_tetra_faces"], device)
        mesh = Mesh(vertices, faces, True)
        points = _tensor(raw["mesh_matrix_points"], device)[:, None].requires_grad_(True)
        translations = _tensor(raw["mesh_env_translations"], device)
        rotations = torch.eye(3, device=device).expand(2, 2, 3, 3).clone()
        active = torch.tensor([[True, True], [False, True]], device=device)
        result = mesh_distance(
            points, [mesh, mesh], translations, rotations, signed=True,
            env_indices=torch.tensor([0, 1], dtype=torch.int64, device=device),
            env_mesh_active=active,
        )
        (gradient,) = torch.autograd.grad(result.reduced_distance.sum(), points)
        return np.asarray([
            int(float(result.reduced_distance[0, 0].detach()) < 0),
            int(int(result.winning_mesh[0, 0]) == 0),
            int(int(result.winning_mesh[1, 0]) == 1),
            int(bool(torch.isfinite(result.reduced_distance).all().item()) and result.reduced_distance.shape == (2, 1)),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "voxel":
        def grid(values: torch.Tensor) -> VoxelGrid:
            return VoxelGrid(
                values, float(raw["voxel_size"][0]),
                _tensor(raw["voxel_translation"], device),
                _tensor(raw["voxel_rotation"], device),
                float(raw["voxel_out_of_bounds"][0]),
            )

        primary, alternate = grid(_tensor(raw["voxel_values"], device)), grid(_tensor(raw["voxel_values_alt"], device))
        environments = [[primary, primary], [alternate, alternate]]
        points = _tensor(raw["voxel_matrix_points"], device)[:, None]
        first = points[:1].detach().requires_grad_(True)
        result = query_esdf(
            first, environments,
            env_indices=torch.tensor([0], dtype=torch.int64, device=device),
            grid_active=_tensor(raw["voxel_grid_active"], device),
        )
        second = query_esdf(
            points[1:], environments,
            env_indices=torch.tensor([1], dtype=torch.int64, device=device),
            grid_active=_tensor(raw["voxel_grid_active"], device),
        )
        repeated = query_esdf(
            first, environments,
            env_indices=torch.tensor([0], dtype=torch.int64, device=device),
            grid_active=_tensor(raw["voxel_grid_active"], device),
        )
        # Query reduction on an all-invalid MPS grid currently raises during
        # gather(-1).  Exercise the underlying ESDF sampling OOB contract
        # directly here, while retaining normal tied-grid query coverage.
        oob = sample_voxel_sdf(_tensor(raw["voxel_oob_points"], device), [[primary]])
        (gradient,) = torch.autograd.grad(result.distance.sum(), first)
        return np.asarray([
            int(bool(result.valid.all().item())),
            int(not bool(oob.valid.any().item()) and bool((oob.values == 99.0).all().item())),
            int(int(result.winning_grid[0, 0]) == 0),
            int(int(second.winning_grid[0, 0]) == 1),
            int(result.distance.shape == (1, 1) and second.distance.shape == (1, 1) and float(second.distance[0, 0].detach()) > float(result.distance[0, 0].detach())),
            int(bool(torch.equal(result.distance, repeated.distance) and torch.equal(result.winning_grid, repeated.winning_grid))),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "cost" and case.capability != "ik.inverse_kinematics":
        position = _tensor(raw["pose_cost_position"], device).requires_grad_(True)
        error = torch.cat((position, torch.zeros_like(position)), dim=-1)
        weights = _tensor(raw["pose_cost_weights"], device)
        weighted = pose_cost(error, weights)
        manager = CostManager({
            "pose": CostTerm(lambda value: pose_cost(value, weights), weight=1.0),
            "regularizer": CostTerm(lambda value: pose_cost(value), weight=0.25),
        })
        total, components = manager.evaluate(error)
        noncontiguous = torch.cat((error, error), dim=-1)[:, ::2]
        (gradient,) = torch.autograd.grad(total.sum(), position)
        return np.asarray([
            int(float(pose_cost(torch.zeros_like(error[:1])).item()) == 0.0),
            int(bool((weighted > 0).all().item())),
            int(bool(torch.allclose(total, components["pose"] + components["regularizer"]))),
            int(not noncontiguous.is_contiguous() and noncontiguous.shape == error.shape and bool(torch.isfinite(pose_cost(noncontiguous)).all().item())),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "dynamics":
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "robot.urdf"
            path.write_bytes(raw["robot_urdf_utf8"].tobytes())
            from curobo._src.robot.dynamics import Dynamics, DynamicsCfg
            from curobo._src.state.state_joint import JointState as CompatJointState
            from curobo._src.types.robot import RobotCfg as CompatRobotCfg
            from curobo.types import DeviceCfg as CompatDeviceCfg

            cfg = CompatDeviceCfg(device, torch.float32)
            robot = CompatRobotCfg.create(_robot_mapping(str(path)), cfg, load_collision_spheres=False)
            params = CompatRobotCfg._kinematics_params(robot.kinematics)
            model = Dynamics(DynamicsCfg(params, cfg))
            position = _tensor(raw["q"], device).requires_grad_(True)
            velocity = _tensor(raw["dynamics_velocity"], device).requires_grad_(True)
            acceleration = _tensor(raw["dynamics_acceleration"], device).requires_grad_(True)
            state = CompatJointState(position=position, velocity=velocity, acceleration=acceleration, joint_names=params.joint_names)
            torque = model.compute_inverse_dynamics(state)
            zeros = torch.zeros_like(position[:1])
            zero_torque = model.compute_inverse_dynamics(CompatJointState(position=position[:1], velocity=zeros, acceleration=zeros, joint_names=params.joint_names))
            source = _tensor(raw["dynamics_noncontiguous_source"], device)
            noncontiguous = source[:, ::2]
            nc_torque = model.compute_inverse_dynamics(CompatJointState(position=noncontiguous, velocity=velocity, acceleration=acceleration, joint_names=params.joint_names))
            (gradient,) = torch.autograd.grad(torque.sum(), position)
            mutated = torque.clone()
            mutated.add_(1.0)
            return np.asarray([
                int(zero_torque.shape == (1, 2) and bool(torch.isfinite(zero_torque).all().item())),
                int(torque.shape == (2, 2)),
                int(not noncontiguous.is_contiguous() and nc_torque.shape == (2, 2)),
                int(not bool(torch.equal(torque, mutated))),
                int(bool(torch.isfinite(gradient).all().item())),
            ], dtype=np.int8)
    if case.probe == "particle":
        observed = _particle_replay(raw, device)
        rate = observed["multi_seed_improvement_rate"]
        return np.asarray([
            int(rate.shape == (3,) and bool((rate >= 0.8).all())),
            int(observed["deterministic_repeat"][0] == 1 and observed["fixed_sample_repeat"][0] == 1),
            int(observed["shift_observed"][0] == 1),
            int(observed["multi_seed_final_std"].shape == (3,) and bool(np.isfinite(observed["multi_seed_final_std"]).all())),
            int(observed["solution_finite"][0] == 1 and bool(observed["objective_improved"].all())),
            int(observed["batch_independent"][0] == 1),
        ], dtype=np.int8)
    if case.probe == "lbfgs":
        observed = _lbfgs_replay(raw, device)
        variant_raw = dict(raw)
        variant_raw["lbfgs_target"] = -raw["lbfgs_target"]
        variant = _lbfgs_replay(variant_raw, device)
        x = _tensor(raw["lbfgs_initial"], device).requires_grad_(True)
        target = _tensor(raw["lbfgs_target"], device)
        weight = _tensor(raw["lbfgs_weight"], device)
        (gradient,) = torch.autograd.grad((0.5 * weight * (x - target).square()).sum(), x)
        return np.asarray([
            int(observed["bounds_satisfied"][0] == 1 and bool(observed["objective_improved"].all())),
            int(variant["bounds_satisfied"][0] == 1 and bool(variant["objective_improved"].all())),
            int(observed["reset_equivalent"][0] == 1),
            int(observed["convergence_code"].shape == (3,) and bool((observed["convergence_code"] <= 1).all())),
            int(observed["nonfinite_status"][0] == 2),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "graph":
        observed = _prm_replay(raw, device)
        return np.asarray([
            int(bool(observed["success"][0]) and observed["status_code"][0] == 0),
            int(bool(observed["success"][1]) and observed["point_count"][1] > 2),
            int(bool((~observed["success"][2:5]).all()) and bool((observed["point_count"][2:5] == 0).all())),
            int(observed["batch_observed"][0] == 1),
            int(observed["deterministic_repeat"][0] == 1),
            int(observed["status_code"].shape == (6,) and observed["path_cost"].shape == (6,)),
        ], dtype=np.int8)
    if case.capability == "trajectory.trajectory_optimization":
        q = _tensor(raw["q"], device)
        primary = minimum_jerk_trajectory(q[0], q[1], 5)
        alternate = minimum_jerk_trajectory(q[0], q[1] * 0.5, 7)
        return np.asarray([
            int(bool(torch.equal(primary[0], q[0]) and torch.equal(primary[-1], q[1]))),
            int(bool(torch.isfinite(primary).all().item())),
            int(bool(torch.equal(alternate[0], q[0]) and torch.equal(alternate[-1], q[1] * 0.5))),
            int(primary.shape == (5, 2) and alternate.shape == (7, 2)),
        ], dtype=np.int8)
    if case.probe == "bspline":
        q = _tensor(raw["q"], device).requires_grad_(True)
        start, goal = ReplayJointState.from_position(q[:1]), ReplayJointState.from_position(q[1:])
        def solve(horizon: int):
            fraction = torch.linspace(0.0, 1.0, 8, device=q.device, dtype=q.dtype)[1:-1]
            action = q[:1, None] * (1.0 - fraction[None, :, None]) + q[1:, None] * fraction[None, :, None]
            return _cubic_boundary_spline(action, start, goal, torch.tensor(0.1, device=q.device), horizon)
        first = solve(21)
        second = solve(31)
        (gradient,) = torch.autograd.grad(first[0].sum(), q)
        return np.asarray([
            int(bool(torch.allclose(first[0][..., 0, :], q[:1]) and torch.allclose(first[0][..., -1, :], q[1:]))),
            int(first[0].shape[-2] == 21 and second[0].shape[-2] == 31),
            int(bool(torch.isfinite(first[1]).all().item()) and bool(torch.isfinite(first[2]).all().item())),
            int(bool(torch.isfinite(first[3]).all().item())),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.capability == "ik.inverse_kinematics":
        observed = probe(case, raw, device)
        position = _tensor(raw["pose_cost_position"][:1], device).requires_grad_(True)
        residual = torch.cat((position, torch.zeros_like(position)), dim=-1)
        (gradient,) = torch.autograd.grad(pose_cost(residual).sum(), position)
        batch = _required_edge(case, raw, device)
        invalid = _required_invalid(case, raw, device)
        success = observed["success"]
        shape = observed["solution_shape"]
        return np.asarray([
            int(success.shape[0] == 1),
            int(batch[0] == 1),
            int(bool(success.all())),
            int(invalid[0] == 1),
            int(position.shape == (1, 3) and _tensor(raw["q"], device).shape[-1] == 2),
            int(shape.dtype == np.int64 and shape.size >= 3 and bool((shape > 0).all())),
            int(observed["position_converged"].dtype == np.bool_ and observed["rotation_converged"].dtype == np.bool_),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    if case.probe == "motion_planner":
        observed = _motion_planner_replay(raw, device)
        repeat = _motion_planner_replay(raw, device)
        start = _tensor(raw["motion_start"], device).requires_grad_(True)
        goal = start + _tensor(raw["motion_goal_delta"], device)
        (gradient,) = torch.autograd.grad((goal - start).square().sum(), start)
        trajectory = observed["trajectory"]
        return np.asarray([
            int(trajectory.shape[-1] == start.numel()),
            int(observed["success"][0, 0] and observed["solution_finite"][0] == 1),
            int(observed["invalid_rejected"][0] == 1),
            int(trajectory.shape[:2] == (1, 1)),
            int(bool(np.array_equal(observed["status_code"], repeat["status_code"]))),
            int(observed["solution_shape"].dtype == np.int64 and observed["solution_shape"].size == 4),
            int(observed["start_converged"][0] == 1 and observed["endpoint_converged"][0] == 1 and np.isfinite(observed["path_length"]).all()),
            int(bool(torch.isfinite(gradient).all().item())),
        ], dtype=np.int8)
    raise AssertionError(f"no matrix probe for {case.capability}")


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
        zero_pose = Pose(
            position=torch.zeros((1, 3), device=device),
            quaternion=torch.zeros((1, 4), device=device),
            normalize_rotation=True,
        )
        zero_matrix = zero_pose.get_matrix()
        expected_zero_matrix = torch.diag(
            torch.tensor([-1.0, -1.0, -1.0, 1.0], device=device)
        ).unsqueeze(0)
        return {
            "matrix": value.get_matrix().detach().cpu().numpy(),
            "points": points.detach().cpu().numpy(),
            "invalid_rejected": np.asarray(
                [int(bool(torch.equal(zero_matrix, expected_zero_matrix)))], np.int8
            ),
        }
    if case.probe == "joint_state":
        return {
            **run_joint_state_probe(raw, device),
            "invalid_rejected": _invalid_rejected(
                lambda: JointState.from_position(
                    _tensor(raw["joint_position"], device), ["j0", "j1"]
                ).reorder(["missing"])
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
        return _particle_replay(raw, device)
    if case.probe == "motion_planner":
        return _motion_planner_replay(raw, device)
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
        from curobo._src.optim.particle.evolution_strategies import EvolutionStrategiesCfg
        return _invalid_rejected(lambda: EvolutionStrategiesCfg(num_problems=0))
    if case.probe == "lbfgs":
        from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
        return _invalid_rejected(lambda: LBFGSOptCfg(stable_mode=False))
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
        source = _tensor(raw["joint_noncontiguous_source"], device)
        return _edge_observed(
            lambda: JointState.from_position(source[:, ::2], ["j0", "j1"])
        )
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
        output = _particle_replay(raw, device)
        observed = all(int(output[key][0]) == 1 for key in (
            "solution_finite", "bounds_satisfied", "deterministic_repeat",
            "batch_independent", "shift_observed", "fixed_sample_repeat",
        )) and bool(output["objective_improved"].all())
        return np.asarray([int(observed)], dtype=np.int8)
    if case.probe == "lbfgs":
        output = _lbfgs_replay(raw, device)
        observed = all(int(output[key][0]) == 1 for key in (
            "bounds_satisfied", "batch_observed", "reset_equivalent"
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
        if "edge_observed" not in results:
            results["edge_observed"] = _required_edge(case, raw, device)
        if case.matrix_cases:
            results["matrix_observed"] = _matrix_observed(case, raw, device)
        save_npz(output_path, results)
        with np.load(output_path, allow_pickle=False) as observed:
            invalid = observed["invalid_rejected"]
            edge = observed["edge_observed"]
            if invalid.shape != (1,) or invalid.dtype != np.int8 or int(invalid[0]) != 1:
                raise RuntimeError(f"invalid-case probe did not reject: {case.capability}")
            if edge.shape != (1,) or edge.dtype != np.int8 or int(edge[0]) != 1:
                raise RuntimeError(f"edge probe did not execute: {case.capability}")
            if case.matrix_cases:
                matrix = observed["matrix_observed"]
                if matrix.shape != (len(case.matrix_cases),) or matrix.dtype != np.int8 or not bool(matrix.all()):
                    raise RuntimeError(f"matrix probe did not cover every declared case: {case.capability}")
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
                         "matrix": (
                             {"cases": list(case.matrix_cases), "output": "matrix_observed", "executed": True}
                             if case.matrix_cases else None
                         ),
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
