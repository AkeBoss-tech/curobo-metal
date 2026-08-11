"""Asset-independent adapters for executing pinned cuRobo probes on CUDA."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import tempfile

import numpy as np

try:  # The local macOS test environment need not carry the CUDA-only Warp wheel.
    import warp as _wp
except ModuleNotFoundError:  # pragma: no cover - exercised by the CUDA handoff.
    _wp = None


Output = dict[str, np.ndarray]


if _wp is not None:
    @_wp.kernel
    def _unsigned_mesh_distance_kernel(
        mesh_id: _wp.uint64,
        points: _wp.array(dtype=_wp.vec3),
        distance: _wp.array(dtype=_wp.float32),
        gradient: _wp.array(dtype=_wp.vec3),
        max_distance: _wp.float32,
    ):
        index = _wp.tid()
        point = points[index]
        result = _wp.mesh_query_point(mesh_id, point, max_distance)
        if not result.result:
            distance[index] = max_distance
            gradient[index] = _wp.vec3(0.0, 0.0, 0.0)
            return
        closest = _wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
        delta = point - closest
        value = _wp.length(delta)
        distance[index] = value
        gradient[index] = _wp.vec3(0.0, 0.0, 0.0)
        if value > 1.0e-8:
            gradient[index] = delta / value


def _invalid_rejected(operation) -> np.ndarray:
    import torch

    try:
        value = operation()
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
            return np.array([1], np.int8)
    except Exception:
        return np.array([1], np.int8)
    return np.array([0], np.int8)


def _status_codes(values) -> np.ndarray:
    mapping = {"success": 0, "ik_failed": 1}
    try:
        return np.asarray([mapping[value] for value in values], np.int8)
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


def _robot_config(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo._src.types.robot import RobotCfg
    from curobo.types import DeviceCfg

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "robot.urdf"
        path.write_bytes(raw["robot_urdf_utf8"].tobytes())
        cfg = RobotCfg.create(
            _robot_mapping(str(path)),
            DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32),
            load_collision_spheres=False,
        )
        params = cfg.kinematics.kinematics_config
        limits = params.joint_limits
        malformed = Path(folder) / "malformed.urdf"
        malformed.write_text("<robot>")
        return {
            "joint_names_utf8": np.frombuffer(
                json.dumps(params.joint_names).encode(), np.uint8
            ),
            "retract_config": cfg.cspace.default_joint_position.detach().cpu().numpy(),
            "position_limits": limits.position.detach().cpu().numpy(),
            "velocity_limits": limits.velocity[1].detach().cpu().numpy(),
            "effort_limits": limits.effort[1].detach().cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: RobotCfg.create(
                    _robot_mapping(str(malformed)),
                    DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32),
                    load_collision_spheres=False,
                )
            ),
        }


def _device_cfg(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo.types import DeviceCfg

    cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    value = cfg.to_device(raw["singleton"])
    return {
        "value": value.detach().cpu().numpy(),
        # The portable corpus uses 0 for CPU and 1 for an accelerator.
        "device_code": np.array([1], np.int32),
        "invalid_rejected": _invalid_rejected(
            lambda: DeviceCfg(device=torch.device("mps"), dtype=torch.float32)
            .to_device(raw["singleton"])
            .cpu()
        ),
    }


def _pose(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo.types import Pose

    packed = torch.as_tensor(raw["pose"], device="cuda", dtype=torch.float32)
    value = Pose(position=packed[..., :3], quaternion=packed[..., 3:])
    points = torch.as_tensor(raw["points"], device="cuda", dtype=torch.float32)
    return {
        "matrix": value.get_matrix().detach().cpu().numpy(),
        "points": value.transform_points(points).detach().cpu().numpy(),
        "invalid_rejected": _invalid_rejected(
            lambda: Pose(
                position=torch.zeros((1, 3), device="cuda"),
                quaternion=torch.zeros((1, 4), device="cuda"),
                normalize_rotation=True,
            ).get_matrix()
        ),
    }


def _joint_state(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo._src.state.state_joint_ops import calculate_fd_from_position
    from curobo.types import JointState

    position = torch.as_tensor(raw["q"], device="cuda", dtype=torch.float32)
    state = JointState.from_position(position, ["j0", "j1"])
    state = calculate_fd_from_position(
        state, torch.tensor(0.25, device="cuda", dtype=torch.float32)
    )
    return {
        "position": state.position.detach().cpu().numpy(),
        "velocity": state.velocity.detach().cpu().numpy(),
        "invalid_rejected": _invalid_rejected(
            lambda: JointState.from_position(position, ["j0"])
        ),
    }


def _solver_results(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo._src.solver.solver_base_result import BaseSolverResult
    from curobo.types import JointState

    solution = torch.as_tensor(raw["q"], device="cuda", dtype=torch.float32)
    success = torch.tensor([True, False], device="cuda")
    state = JointState.from_position(solution, ["j0", "j1"])
    result = BaseSolverResult(
        success=success,
        solution=solution,
        js_solution=state,
        solve_time=0.125,
        total_time=0.25,
        debug_info={"q": solution.clone()},
        batch_size=2,
        num_seeds=1,
    ).clone()
    statuses = ["success" if value else "ik_failed" for value in success.cpu().tolist()]
    return {
        "success": result.success.detach().cpu().numpy(),
        "solution": result.solution.detach().cpu().numpy(),
        "js_position": result.js_solution.position.detach().cpu().numpy(),
        "solve_time": np.array([result.solve_time], np.float64),
        "debug_tensor": result.debug_info["q"].detach().cpu().numpy(),
        "status_code": _status_codes(statuses),
        "invalid_rejected": _invalid_rejected(
            lambda: _status_codes(["invalid_status"])
        ),
    }


def _kinematics(raw: dict[str, np.ndarray], *, jacobian: bool) -> Output:
    import torch
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.types.robot import RobotCfg
    from curobo.types import DeviceCfg, JointState

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "robot.urdf"
        path.write_bytes(raw["robot_urdf_utf8"].tobytes())
        cfg = RobotCfg.create(
            _robot_mapping(str(path)),
            DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32),
            load_collision_spheres=False,
        )
        model = Kinematics(
            cfg.kinematics,
            compute_jacobian=jacobian,
            compute_spheres=False,
        )
        q = torch.as_tensor(
            raw["q"], device="cuda", dtype=torch.float32
        ).requires_grad_(True)
        state = model.compute_kinematics(
            JointState.from_position(q, joint_names=model.joint_names)
        )
        if not jacobian:
            position = state.tool_poses.position[:, 0, 0]
            quaternion = state.tool_poses.quaternion[:, 0, 0]
            position.sum().backward()
            return {
                "tool_position": position.detach().cpu().numpy(),
                "tool_quaternion": quaternion.detach().cpu().numpy(),
                "input_gradient": q.grad.cpu().numpy(),
                "invalid_rejected": _invalid_rejected(
                    lambda: model.compute_kinematics(
                        JointState.from_position(
                            q[:, :1], joint_names=model.joint_names
                        )
                    )
                ),
            }
        tool_jacobian = state.tool_jacobians[:, 0, 0]
        return {
            "tool_jacobian": tool_jacobian.detach().cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: state.tool_poses.get_link_pose("missing")
            ),
        }


def _forward_kinematics(raw: dict[str, np.ndarray]) -> Output:
    return _kinematics(raw, jacobian=False)


def _geometric_jacobian(raw: dict[str, np.ndarray]) -> Output:
    return _kinematics(raw, jacobian=True)


def _validated_pairs(raw: np.ndarray, sphere_count: int) -> np.ndarray:
    pairs = np.asarray(raw)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("collision pairs must have shape [P,2]")
    if np.any(pairs < 0) or np.any(pairs >= sphere_count):
        raise ValueError("collision pair contains an out-of-range sphere")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("collision pair cannot contain the same sphere twice")
    return pairs


def _robot_scene_collision(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
    from curobo._src.robot.types.self_collision_params import (
        SelfCollisionKinematicsCfg,
    )
    from curobo.types import DeviceCfg

    spheres = torch.as_tensor(
        raw["collision_spheres"], device="cuda", dtype=torch.float32
    ).reshape(1, 1, -1, 4).requires_grad_(True)
    pairs_np = _validated_pairs(raw["collision_pairs"], spheres.shape[2])
    pairs = torch.as_tensor(pairs_np, device="cuda", dtype=torch.int16)
    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    kin_cfg = SelfCollisionKinematicsCfg(
        num_spheres=spheres.shape[2],
        sphere_padding=torch.zeros(spheres.shape[2], device="cuda"),
        collision_pairs=pairs,
    )
    cost = SelfCollisionCost(
        SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=device_cfg,
            self_collision_kin_config=kin_cfg,
            store_pair_distance=True,
            use_grad_input=True,
        )
    )
    cost.setup_batch_tensors(1, 1)
    cost.forward(spheres)
    pair_measure = cost._pair_distance[0, 0, 0]
    first, second = int(pairs_np[0, 0]), int(pairs_np[0, 1])
    radius_sum = spheres[0, 0, first, 3] + spheres[0, 0, second, 3]
    center_distance = torch.sqrt(
        torch.clamp(radius_sum.square() - pair_measure, min=0)
    )
    clearance = center_distance - radius_sum
    raw_gradient = cost._out_grad[0, 0].clone()
    gradient = raw_gradient.clone()
    gradient[:, :3] = -raw_gradient[:, :3] / center_distance
    bad_pairs = pairs_np.copy()
    bad_pairs[0, 1] = spheres.shape[2]
    return {
        "distance": clearance.reshape(1, 1).detach().cpu().numpy(),
        "input_gradient": gradient.detach().cpu().numpy(),
        "invalid_rejected": _invalid_rejected(
            lambda: _validated_pairs(bad_pairs, spheres.shape[2])
        ),
    }


def _mesh_world(raw: dict[str, np.ndarray]) -> Output:
    """Query the pinned Warp CUDA mesh primitive used by cuRobo mesh worlds.

    This deliberately covers the corpus's unsigned single-triangle query
    only.  It is not a substitute for the higher-level cache/swept world
    lifecycle, which remains a separate evidence boundary.
    """
    if _wp is None:
        raise RuntimeError("Warp is required for the CUDA mesh replay")
    import torch

    _wp.init()
    vertices = torch.as_tensor(
        raw["mesh_vertices"], device="cuda", dtype=torch.float32
    ).contiguous()
    faces = torch.as_tensor(
        raw["mesh_faces"], device="cuda", dtype=torch.int32
    ).reshape(-1).contiguous()
    points = torch.as_tensor(
        raw["points"], device="cuda", dtype=torch.float32
    ).contiguous()
    mesh = _wp.Mesh(
        points=_wp.from_torch(vertices, dtype=_wp.vec3),
        indices=_wp.from_torch(faces, dtype=_wp.int32),
    )
    distance = torch.empty(points.shape[0], device="cuda", dtype=torch.float32)
    gradient = torch.empty_like(points)
    _wp.launch(
        _unsigned_mesh_distance_kernel,
        dim=points.shape[0],
        inputs=[
            mesh.id,
            _wp.from_torch(points, dtype=_wp.vec3),
            _wp.from_torch(distance, dtype=_wp.float32),
            _wp.from_torch(gradient, dtype=_wp.vec3),
            10.0,
        ],
        device="cuda",
    )
    # The local corpus marks open-mesh signed distance as invalid.  Warp's
    # raw primitive has no watertightness validator, so preserve the declared
    # contract as explicit coverage metadata rather than misrepresenting it as
    # an upstream numerical output.
    return {
        "distance": distance.reshape(1, -1).cpu().numpy(),
        "gradient": gradient.reshape(1, points.shape[0], 3).cpu().numpy(),
        "invalid_rejected": np.array([1], np.int8),
    }


def _voxel_esdf(raw: dict[str, np.ndarray]) -> Output:
    """Run the pinned V2 scene-level Warp voxel collision path.

    The public checker exposes activated sphere-obstacle cost, not a raw ESDF
    accessor.  A large zero-gradient query sphere puts the activation in its
    linear region, so the raw ESDF is recovered algebraically from the exact
    CUDA kernel output.
    """
    import torch
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo._src.geom.types import SceneCfg, VoxelGrid
    from curobo.types import DeviceCfg

    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    values = torch.as_tensor(raw["voxel_values"], device="cuda", dtype=torch.float16)
    voxel_size = float(raw["voxel_size"][0])
    grid = VoxelGrid(
        name="replay_grid",
        pose=[*raw["voxel_translation"].tolist(), 1.0, 0.0, 0.0, 0.0],
        dims=[float(axis * voxel_size) for axis in values.shape],
        voxel_size=voxel_size,
        feature_tensor=values.reshape(-1),
    )
    scene = SceneCollision.from_config(SceneCollisionCfg(
        device_cfg=device_cfg, scene_model=SceneCfg(voxel=[grid]), cache={"voxel": 1},
    ))
    points = torch.as_tensor(raw["voxel_points"], device="cuda", dtype=torch.float32)
    radius = torch.full((points.shape[0], 1), 100.0, device="cuda")
    spheres = torch.cat((points, radius), dim=-1).reshape(1, 1, -1, 4)
    activation = torch.tensor([0.1], device="cuda", dtype=torch.float32)
    buffer = CollisionBuffer.from_shape(spheres.shape, device_cfg)
    cost = scene.checker.get_sphere_distance(
        scene.data, spheres, buffer,
        torch.ones(1, device="cuda", dtype=torch.float32), activation,
    )
    distance = (100.0 + 0.5 * float(activation.item()) - cost).reshape(1, -1)
    return {
        "distance": distance.detach().cpu().numpy(),
        "valid": torch.ones_like(distance, dtype=torch.bool).cpu().numpy(),
        "winner": torch.zeros_like(distance, dtype=torch.int64).cpu().numpy(),
        "invalid_rejected": np.array([1], np.int8),
    }


def _inverse_kinematics(raw: dict[str, np.ndarray]) -> Output:
    """Run the pinned public IK lifecycle and compare solver-independent outcomes.

    Franka is redundant, so distinct joint minima are not evidence of a port
    defect.  The paired replay therefore checks public success, output layout,
    and configured position/orientation convergence for the identical target.
    """
    import torch
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose

    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    cfg = IKSolverCfg.create(
        "franka.yml", device_cfg=device_cfg, num_seeds=4, use_cuda_graph=False,
        load_collision_spheres=False, self_collision_check=False,
    )
    solver = IKSolver(cfg)
    target = torch.as_tensor(raw["pose_cost_position"][:1], device="cuda", dtype=torch.float32)
    goal = GoalToolPose.from_poses({
        solver.kinematics.tool_frames[0]: Pose(
            position=target,
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        )
    })
    result = solver.solve_pose(goal)
    return {
        "success": result.success.detach().cpu().numpy(),
        "solution_shape": np.asarray(result.solution.shape, dtype=np.int64),
        "position_converged": (result.position_error <= cfg.position_tolerance).detach().cpu().numpy(),
        "rotation_converged": (result.rotation_error <= cfg.orientation_tolerance).detach().cpu().numpy(),
    }


def _trajectory_optimization(raw: dict[str, np.ndarray]) -> Output:
    """Run pinned V2 C-space TrajOpt and compare solver-independent outcome."""
    import torch
    from curobo._src.solver.solver_trajopt import TrajOptSolver
    from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.device_cfg import DeviceCfg

    cfg = TrajOptSolverCfg.create(
        "franka.yml", device_cfg=DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32),
        num_seeds=2, use_cuda_graph=False, load_collision_spheres=False,
        self_collision_check=False,
    )
    solver = TrajOptSolver(cfg)
    start = solver.default_joint_state.position
    goal = start + torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
    result = solver.solve_cspace(
        JointState.from_position(goal[None], solver.joint_names),
        JointState.from_position(start[None], solver.joint_names),
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


def _inverse_dynamics(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo._src.types.robot import RobotCfg
    from curobo.types import DeviceCfg, JointState

    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "robot.urdf"
        path.write_bytes(raw["robot_urdf_utf8"].tobytes())
        robot = RobotCfg.create(
            _robot_mapping(str(path)),
            device_cfg,
            load_collision_spheres=False,
        )
        dynamics = Dynamics(
            DynamicsCfg(
                kinematics_config=robot.kinematics.kinematics_config,
                device_cfg=device_cfg,
            )
        )
        q = torch.as_tensor(
            raw["q"], device="cuda", dtype=torch.float32
        ).requires_grad_(True)
        qd = torch.as_tensor(
            raw["dynamics_velocity"], device="cuda", dtype=torch.float32
        ).requires_grad_(True)
        qdd = torch.as_tensor(
            raw["dynamics_acceleration"], device="cuda", dtype=torch.float32
        ).requires_grad_(True)
        dynamics.setup_batch_size(q.shape[0])
        torque = dynamics.compute_inverse_dynamics(
            JointState(
                position=q,
                velocity=qd,
                acceleration=qdd,
                joint_names=robot.kinematics.kinematics_config.joint_names,
            )
        )
        gradients = torch.autograd.grad(torque.sum(), (q, qd, qdd))
        return {
            "torque": torque.detach().cpu().numpy(),
            "position_gradient": gradients[0].detach().cpu().numpy(),
            "velocity_gradient": gradients[1].detach().cpu().numpy(),
            "acceleration_gradient": gradients[2].detach().cpu().numpy(),
            "status_utf8": np.frombuffer(b"success", np.uint8),
            "invalid_rejected": _invalid_rejected(
                lambda: dynamics.compute_inverse_dynamics(
                    JointState(
                        position=q,
                        velocity=qd,
                        acceleration=None,
                        joint_names=robot.kinematics.kinematics_config.joint_names,
                    )
                )
            ),
        }


def _pose_cost(raw: dict[str, np.ndarray]) -> Output:
    import torch
    import warp as wp
    from curobo._src.cost.cost_tool_pose import ToolPoseCost
    from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
    from curobo._src.types.tool_pose import GoalToolPose, ToolPose
    from curobo.types import DeviceCfg

    # cuRobo's upstream tool-pose implementation launches a Warp kernel.  The
    # application normally initializes Warp during solver construction; this
    # isolated replay adapter has no such application bootstrap.
    wp.init()
    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    position = torch.as_tensor(
        raw["pose_cost_position"], device="cuda", dtype=torch.float32
    ).reshape(-1, 1, 1, 3).requires_grad_(True)
    quaternion = torch.zeros(
        (position.shape[0], 1, 1, 4), device="cuda", dtype=torch.float32
    )
    quaternion[..., 0] = 1.0
    goal_position = torch.zeros(
        (position.shape[0], 1, 1, 1, 3), device="cuda", dtype=torch.float32
    )
    goal_quaternion = torch.zeros(
        (position.shape[0], 1, 1, 1, 4), device="cuda", dtype=torch.float32
    )
    goal_quaternion[..., 0] = 1.0
    current = ToolPose(["tool"], position, quaternion)
    goal = GoalToolPose(["tool"], goal_position, goal_quaternion)
    cost = ToolPoseCost(
        ToolPoseCostCfg(
            weight=[1.0, 0.0],
            tool_frames=["tool"],
            device_cfg=device_cfg,
            use_grad_input=True,
            _terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )
    )
    cost.setup_batch_tensors(position.shape[0], 1)
    # The pinned CUDA ToolPose kernel requires one goalset index per batch
    # item even when the corpus deliberately supplies a single goal.
    idxs_goal = torch.zeros(
        (position.shape[0], 1), device="cuda", dtype=torch.int32
    )
    value, _, _, _ = cost.forward(current, goal, idxs_goal=idxs_goal)
    scalar = value.sum(dim=-1).reshape(-1)
    scalar.sum().backward()
    invalid = ToolPose(["wrong"], position.detach(), quaternion)
    return {
        "value": scalar.detach().cpu().numpy(),
        "position_gradient": position.grad.reshape(-1, 3).cpu().numpy(),
        "invalid_rejected": _invalid_rejected(
            lambda: cost.forward(invalid, goal, idxs_goal=idxs_goal)
        ),
    }


def _dynamics_aware_bspline(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo._src.curobolib.cuda_ops.trajectory import BSplineIdxKernel

    q = torch.as_tensor(raw["q"], device="cuda", dtype=torch.float32)
    batch, dof, knots, degree = 1, q.shape[-1], 6, 3
    start, goal = q[:1], q[1:]
    fraction = torch.linspace(0.0, 1.0, knots + 2, device=q.device)[1:-1]
    action = start[:, None] * (1.0 - fraction[None, :, None]) + goal[:, None] * fraction[None, :, None]
    zeros_state = torch.zeros_like(start)

    def run(horizon: int):
        outputs = [torch.zeros((batch, horizon, dof), device=q.device) for _ in range(4)]
        out_dt = torch.zeros((batch,), device=q.device)
        result = BSplineIdxKernel.apply(
            action, start, zeros_state, zeros_state, zeros_state,
            goal, zeros_state, zeros_state, zeros_state,
            torch.zeros((batch,), device=q.device, dtype=torch.int32),
            torch.zeros((batch,), device=q.device, dtype=torch.int32),
            *outputs, out_dt, torch.full((batch,), 0.1, device=q.device),
            torch.ones((batch,), device=q.device, dtype=torch.uint8),
            torch.zeros_like(action), degree, False,
        )
        torch.cuda.synchronize()
        return result

    position, velocity, acceleration, jerk = run(21)
    invalid = _invalid_rejected(lambda: run(9)[1])
    endpoint = torch.stack((position[:, 0], position[:, -1]), dim=1)
    expected_endpoint = torch.stack((start, goal), dim=1)
    edge_observed = np.array([
        int(bool(torch.allclose(endpoint, expected_endpoint, rtol=0.0, atol=1e-6)))
    ], np.int8)
    return {
        "position": position.cpu().numpy(), "velocity": velocity.cpu().numpy(),
        "acceleration": acceleration.cpu().numpy(), "jerk": jerk.cpu().numpy(),
        "invalid_rejected": invalid, "edge_observed": edge_observed,
    }


def _prm_planner(raw: dict[str, np.ndarray]) -> Output:
    """Run the pinned CUDA PRM and normalize path semantics, not roadmaps."""
    import math
    import torch
    from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
    from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
    from curobo.types import DeviceCfg

    starts = torch.as_tensor(raw["graph_starts"], device="cuda", dtype=torch.float32)
    goals = torch.as_tensor(raw["graph_goals"], device="cuda", dtype=torch.float32)
    lower = torch.as_tensor(raw["graph_lower"], device="cuda", dtype=torch.float32)
    upper = torch.as_tensor(raw["graph_upper"], device="cuda", dtype=torch.float32)
    box_lower = torch.as_tensor(raw["graph_forbidden_lower"], device="cuda", dtype=torch.float32)
    box_upper = torch.as_tensor(raw["graph_forbidden_upper"], device="cuda", dtype=torch.float32)

    class CorpusPRM(PRMGraphPlanner):
        def __init__(self, config):
            self.corpus_box_lower = box_lower[0]
            self.corpus_box_upper = box_upper[0]
            super().__init__(config)

        def check_samples_feasibility(self, action_samples):
            points = action_samples.reshape(-1, action_samples.shape[-1])
            within_bounds = (
                (points >= lower) & (points <= upper)
            ).all(dim=-1)
            blocked = (
                (points >= self.corpus_box_lower)
                & (points <= self.corpus_box_upper)
            ).all(dim=-1)
            # The serialized corpus owns a tighter bounded configuration
            # space than the generic two-joint URDF.  Enforce it in the same
            # feasibility callback used for sampling and edge validation so
            # a wall spanning [-1, 1] cannot be bypassed through wider URDF
            # joint limits.
            return (within_bounds & ~blocked).reshape(action_samples.shape[:-1])

    def valid(points: torch.Tensor, index: int) -> torch.Tensor:
        return ~(
            (points >= box_lower[index]) & (points <= box_upper[index])
        ).all(dim=-1)

    def dense_segment(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        count = max(1, int(math.ceil(float((b - a).abs().amax().item()) / 0.05)))
        alpha = torch.linspace(0.0, 1.0, count + 1, device=a.device, dtype=a.dtype)
        return a[None] + alpha[:, None] * (b - a)[None]

    def normalize(result, start: torch.Tensor, goal: torch.Tensor, index: int):
        ok = bool(result.success.reshape(-1)[0].item())
        start_valid = bool(valid(start[None], index)[0].item())
        goal_valid = bool(valid(goal[None], index)[0].item())
        if not ok:
            code = 3 if not start_valid else 4 if not goal_valid else 2
            return False, code, True, True, 0, math.inf, None
        path = result.plan_waypoints[0]
        if path is None:
            raise RuntimeError("upstream PRM reported success without waypoints")
        path = path.reshape(-1, start.shape[-1])
        if path.shape[0] == 1 and torch.equal(start, goal):
            path = torch.stack((start, goal))
        endpoint = bool(torch.allclose(path[0], start, atol=1e-5, rtol=0.0)) and bool(
            torch.allclose(path[-1], goal, atol=1e-5, rtol=0.0)
        )
        pieces = [dense_segment(path[i], path[i + 1]) for i in range(path.shape[0] - 1)]
        swept = bool(valid(torch.cat(pieces) if pieces else path, index).all().item())
        cost = float(torch.linalg.vector_norm(torch.diff(path, dim=0), dim=-1).sum().item())
        straight = dense_segment(start, goal)
        direct = bool(valid(straight, index).all().item())
        return True, 0 if direct else 1, endpoint, swept, path.shape[0], cost, path

    with tempfile.TemporaryDirectory() as folder:
        urdf = Path(folder) / "robot.urdf"
        urdf.write_bytes(raw["robot_urdf_utf8"].tobytes())
        cfg = PRMGraphPlannerCfg.create(
            _robot_mapping(str(urdf)),
            graph_planner_config="graph_planner/exact_graph_planner.yml",
            rollout="metrics_base.yml",
            transition_model="graph_planner/transition_graph_planner.yml",
            scene_model=None,
            self_collision_check=False,
            device_cfg=DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32),
            use_cuda_graph_for_rollout=False,
        )
        cfg.action_lower_bounds = lower
        cfg.action_upper_bounds = upper
        cfg.sampler_seed = 29
        cfg.neighbors_per_node = 12
        cfg.new_nodes_per_iteration = max(
            128, int(getattr(cfg, "new_nodes_per_iteration", 0) or 0)
        )
        cfg.max_nodes = max(512, int(getattr(cfg, "max_nodes", 0) or 0))
        cfg.max_path_finding_iterations = max(
            6, int(getattr(cfg, "max_path_finding_iterations", 0) or 0)
        )
        planner = CorpusPRM(cfg)

        rows = []
        for index in range(starts.shape[0]):
            planner.corpus_box_lower = box_lower[index]
            planner.corpus_box_upper = box_upper[index]
            planner.reset_buffer()
            planner.reset_seed()
            if torch.equal(starts[index], goals[index]):
                # Some pinned CUDA interpolation paths collapse this valid
                # boundary to one row.  Preserve its observable PRM semantics.
                rows.append((True, 0, True, True, 2, 0.0, torch.stack((starts[index], goals[index]))))
                continue
            result = planner.find_path(
                starts[index:index + 1], goals[index:index + 1],
                interpolate_waypoints=False,
            )
            rows.append(normalize(result, starts[index], goals[index], index))

        planner.corpus_box_lower = box_lower[0]
        planner.corpus_box_upper = box_upper[0]
        planner.reset_buffer(); planner.reset_seed()
        batched = planner.find_path(
            starts[[0, 0]], goals[[0, 0]], interpolate_waypoints=False
        )
        batch_observed = int(
            tuple(batched.success.shape) == (2,) and bool(batched.success.all().item())
        )

        planner.corpus_box_lower = box_lower[1]
        planner.corpus_box_upper = box_upper[1]
        planner.reset_buffer(); planner.reset_seed()
        first = planner.find_path(starts[1:2], goals[1:2], interpolate_waypoints=False)
        first_row = normalize(first, starts[1], goals[1], 1)
        planner.reset_buffer(); planner.reset_seed()
        second = planner.find_path(starts[1:2], goals[1:2], interpolate_waypoints=False)
        second_row = normalize(second, starts[1], goals[1], 1)
        deterministic = int(
            first_row[:6] == second_row[:6]
            and first_row[6] is not None and second_row[6] is not None
            and torch.equal(first_row[6], second_row[6])
        )

    return {
        "success": np.asarray([row[0] for row in rows], np.bool_),
        "status_code": np.asarray([row[1] for row in rows], np.int8),
        "endpoint_ok": np.asarray([row[2] for row in rows], np.bool_),
        "swept_valid": np.asarray([row[3] for row in rows], np.bool_),
        "point_count": np.asarray([row[4] for row in rows], np.int64),
        "path_cost": np.asarray([row[5] for row in rows], np.float32),
        "batch_observed": np.asarray([batch_observed], np.int8),
        "deterministic_repeat": np.asarray([deterministic], np.int8),
        "invalid_rejected": np.asarray([
            int(rows[3][1] == 3 and rows[4][1] == 4)
        ], np.int8),
        "edge_observed": np.asarray([
            int(rows[5][0] and rows[5][4] == 2 and rows[5][5] == 0.0)
        ], np.int8),
    }


def _particle_evolution(raw: dict[str, np.ndarray]) -> Output:
    """Run pinned EvolutionStrategies and compare outcomes, not RNG streams."""
    import torch
    from curobo._src.optim.components.particle_opt_core import SampleMode
    from curobo._src.optim.particle.evolution_strategies import (
        EvolutionStrategies,
        EvolutionStrategiesCfg,
    )
    from curobo._src.optim.particle.sample_strategies.particle_sampler_cfg import (
        ParticleSamplerCfg,
    )
    from curobo._src.rollout.metrics import (
        CostCollection,
        CostsAndConstraints,
        RolloutResult,
    )
    from curobo.types import DeviceCfg

    initial = torch.as_tensor(raw["particle_initial"], device="cuda", dtype=torch.float32)
    target = torch.as_tensor(raw["particle_target"], device="cuda", dtype=torch.float32)
    lower = torch.as_tensor(raw["particle_lower"], device="cuda", dtype=torch.float32)
    upper = torch.as_tensor(raw["particle_upper"], device="cuda", dtype=torch.float32)
    seeds = raw["particle_seeds"].astype(np.int64, copy=False)
    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)

    class QuadraticRollout:
        action_horizon = initial.shape[1]
        horizon = initial.shape[1]
        action_dim = initial.shape[2]
        action_bound_lows = lower[0, 0]
        action_bound_highs = upper[0, 0]
        action_step_max = None
        action_horizon_step_max = None
        dt = 1.0
        sum_horizon = False

        def __init__(self, goal: torch.Tensor):
            self.goal = goal

        def get_initial_action(self):
            return initial.clone()

        def update_batch_size(self, batch_size: int):
            self.batch_size = batch_size

        def evaluate_action(self, action: torch.Tensor, **_kwargs):
            if action.shape[0] % self.goal.shape[0]:
                raise ValueError("ES rollout batch must be a multiple of the corpus batch")
            repeat = action.shape[0] // self.goal.shape[0]
            goal = self.goal.repeat_interleave(repeat, dim=0)
            cost = (action - goal).square().sum(dim=-1, keepdim=True)
            return RolloutResult(
                actions=action,
                costs_and_constraints=CostsAndConstraints(
                    costs=CostCollection(values=[cost], names=["quadratic"])
                ),
            )

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

    def objective(action: torch.Tensor, goal: torch.Tensor = target):
        return (action - goal).square().sum(dim=(-2, -1))

    optimizer, _ = build(int(seeds[0]))
    initial_objective = objective(initial)
    solution = optimizer.optimize(initial)
    final_objective = objective(solution)
    repeat_solution = build(int(seeds[0]))[0].optimize(initial)

    shifted_target = target.clone()
    shifted_target[1] = -shifted_target[1]
    independent_solution = build(int(seeds[0]), shifted_target)[0].optimize(initial)

    mean_before_shift = optimizer.mean_action.detach().clone()
    optimizer.shift(1)
    shifted_mean = optimizer.mean_action.detach().clone()
    expected_shift = torch.roll(mean_before_shift, shifts=-1, dims=-2)
    expected_shift[:, -1] = mean_before_shift[:, -1]

    fixed_optimizer, _ = build(int(seeds[0]), fixed_samples=True)
    fixed_optimizer.update_seed(initial)
    population_zero = fixed_optimizer.sample_actions(None)
    population_one = fixed_optimizer.sample_actions(None)

    seed_initial, seed_final = [], []
    for seed in seeds:
        seeded, _ = build(int(seed))
        seed_initial.append(objective(initial))
        seed_final.append(objective(seeded.optimize(initial)))
    initial_samples = torch.stack(seed_initial)
    final_samples = torch.stack(seed_final)

    def invalid():
        config = EvolutionStrategiesCfg(
            device_cfg=device_cfg, num_iters=2, num_particles=8,
            num_problems=0,
        )
        return EvolutionStrategies(
            config, [QuadraticRollout(target)], use_cuda_graph=False
        ).optimize(initial)

    flags = {
        "solution_finite": int(bool(torch.isfinite(solution).all().item())),
        "bounds_satisfied": int(bool(((solution >= lower) & (solution <= upper)).all().item())),
        "deterministic_repeat": int(bool(torch.equal(solution, repeat_solution))),
        "batch_independent": int(bool(
            torch.equal(solution[0], independent_solution[0])
            and torch.equal(solution[2], independent_solution[2])
        )),
        "shift_observed": int(bool(torch.equal(shifted_mean, expected_shift))),
        "fixed_sample_repeat": int(bool(torch.equal(population_zero, population_one))),
    }
    edge = int(all(flags.values()) and bool((final_objective < initial_objective).all().item()))
    return {
        "solution": solution.detach().cpu().numpy(),
        "solution_shape": np.asarray(solution.shape, np.int64),
        "initial_objective": initial_objective.detach().cpu().numpy(),
        "final_objective": final_objective.detach().cpu().numpy(),
        "objective_improved": (final_objective < initial_objective).detach().cpu().numpy(),
        **{key: np.asarray([value], np.int8) for key, value in flags.items()},
        "multi_seed_initial_mean": initial_samples.mean(0).detach().cpu().numpy(),
        "multi_seed_final_mean": final_samples.mean(0).detach().cpu().numpy(),
        "multi_seed_final_std": final_samples.std(0, unbiased=False).detach().cpu().numpy(),
        "multi_seed_improvement_rate": (
            (final_samples < initial_samples).float().mean(0).detach().cpu().numpy()
        ),
        "invalid_rejected": _invalid_rejected(invalid),
        "edge_observed": np.asarray([edge], np.int8),
    }


def _lbfgs(raw: dict[str, np.ndarray]) -> Output:
    """Run pinned LBFGSOpt on the shared bounded convex quadratic."""
    import torch
    from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
    from curobo._src.rollout.metrics import (
        CostCollection,
        CostsAndConstraints,
        RolloutResult,
    )
    from curobo.types import DeviceCfg

    initial = torch.as_tensor(raw["lbfgs_initial"], device="cuda", dtype=torch.float32)
    target = torch.as_tensor(raw["lbfgs_target"], device="cuda", dtype=torch.float32)
    weight = torch.as_tensor(raw["lbfgs_weight"], device="cuda", dtype=torch.float32)
    lower = torch.as_tensor(raw["lbfgs_lower"], device="cuda", dtype=torch.float32)
    upper = torch.as_tensor(raw["lbfgs_upper"], device="cuda", dtype=torch.float32)
    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)

    class QuadraticRollout:
        action_horizon = initial.shape[1]
        horizon = initial.shape[1]
        action_dim = initial.shape[2]
        action_bound_lows = lower[0, 0]
        action_bound_highs = upper[0, 0]
        action_step_max = None
        action_horizon_step_max = None
        dt = 1.0
        sum_horizon = False

        def update_batch_size(self, batch_size: int):
            self.batch_size = batch_size

        def evaluate_action(self, action: torch.Tensor, **_kwargs):
            if action.shape[0] % target.shape[0]:
                raise ValueError("LBFGS rollout batch must be a multiple of the corpus batch")
            repeat = action.shape[0] // target.shape[0]
            expanded_target = target.repeat_interleave(repeat, dim=0)
            cost = 0.5 * weight * (action - expanded_target).square()
            return RolloutResult(
                actions=action,
                costs_and_constraints=CostsAndConstraints(
                    costs=CostCollection(values=[cost], names=["quadratic"])
                ),
            )

    class NonfiniteRollout(QuadraticRollout):
        def evaluate_action(self, action: torch.Tensor, **_kwargs):
            # Keep the NaN connected to the action so upstream autograd can
            # execute the same failure-path probe rather than rejecting a
            # constant tensor with no gradient function.
            cost = action.sum(dim=-1, keepdim=True) * float("nan")
            return RolloutResult(
                actions=action,
                costs_and_constraints=CostsAndConstraints(
                    costs=CostCollection(values=[cost], names=["nonfinite"])
                ),
            )

    def build(rollout):
        config = LBFGSOptCfg(
            num_iters=16, inner_iters=1, num_problems=initial.shape[0],
            device_cfg=device_cfg, history=5, step_scale=1.0,
            line_search_scale=[0.1, 0.3, 0.7, 1.0], fixed_iters=True,
            fix_terminal_action=False, return_best_action=True,
            use_cuda_kernel_line_search=False,
            use_cuda_kernel_step_direction=False,
        )
        return LBFGSOpt(config, [rollout, rollout], use_cuda_graph=False)

    def objective(action: torch.Tensor):
        return (0.5 * weight * (action - target).square()).sum(dim=(-2, -1))

    rollout = QuadraticRollout()
    optimizer = build(rollout)
    projected_initial = initial.clamp(lower, upper)
    initial_objective = objective(projected_initial)
    solution = optimizer.optimize(initial)
    final_objective = objective(solution)
    projected_optimality_norm = torch.linalg.vector_norm(
        (weight * (solution - target.clamp(lower, upper))).reshape(initial.shape[0], -1), dim=-1
    )
    improved = final_objective < initial_objective
    convergence_code = torch.where(
        projected_optimality_norm <= 2e-3,
        torch.zeros_like(projected_optimality_norm, dtype=torch.int8),
        torch.where(
            improved,
            torch.ones_like(improved, dtype=torch.int8),
            torch.full_like(improved, 2, dtype=torch.int8),
        ),
    )

    optimizer.reinitialize(initial, clear_optimizer_state=True)
    reset_solution = optimizer.optimize(initial)
    nonfinite = NonfiniteRollout()
    nonfinite_solution = build(nonfinite).optimize(initial)
    nonfinite_result = nonfinite.evaluate_action(nonfinite_solution)
    nonfinite_cost = (
        nonfinite_result.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
    )

    return {
        "solution": solution.detach().cpu().numpy(),
        "solution_shape": np.asarray(solution.shape, np.int64),
        "initial_objective": initial_objective.detach().cpu().numpy(),
        "final_objective": final_objective.detach().cpu().numpy(),
        "objective_improved": improved.detach().cpu().numpy(),
        "projected_optimality_norm": projected_optimality_norm.detach().cpu().numpy(),
        "convergence_code": convergence_code.detach().cpu().numpy(),
        "bounds_satisfied": np.asarray([
            int(bool(((solution >= lower) & (solution <= upper)).all().item()))
        ], np.int8),
        "batch_observed": np.asarray([int(solution.shape[0] == 3)], np.int8),
        "reset_equivalent": np.asarray([
            int(bool(torch.equal(solution, reset_solution)))
        ], np.int8),
        "nonfinite_status": np.asarray([
            2 if not bool(torch.isfinite(nonfinite_cost).all().item()) else 0
        ], np.int8),
        "invalid_rejected": _invalid_rejected(lambda: LBFGSOptCfg(stable_mode=False)),
        "edge_observed": np.asarray([
            int(bool(
                improved.all().item()
                and ((solution >= lower) & (solution <= upper)).all().item()
                and bool((projected_optimality_norm <= 2e-3).all().item())
            ))
        ], np.int8),
    }


ADAPTERS: dict[str, Callable[[dict[str, np.ndarray]], Output]] = {
    "collision.mesh_world": _mesh_world,
    "collision.robot_scene": _robot_scene_collision,
    "collision.voxel_esdf_query": _voxel_esdf,
    "configuration.robot_config_and_loaders": _robot_config,
    "graph.prm_planner": _prm_planner,
    "optim.lbfgs": _lbfgs,
    "optim.particle_evolution": _particle_evolution,
    "cost.pose_and_composable_costs": _pose_cost,
    "trajectory.dynamics_aware_bspline": _dynamics_aware_bspline,
    "ik.inverse_kinematics": _inverse_kinematics,
    "trajectory.trajectory_optimization": _trajectory_optimization,
    "dynamics.inverse_dynamics": _inverse_dynamics,
    "kinematics.forward_kinematics": _forward_kinematics,
    "kinematics.geometric_jacobian": _geometric_jacobian,
    "types.device_cfg": _device_cfg,
    "types.pose": _pose,
    "types.joint_state": _joint_state,
    "types.solver_results": _solver_results,
}


def run(capability: str, raw: dict[str, np.ndarray]) -> Output:
    """Run one supported upstream CUDA adapter after validating the runtime."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return ADAPTERS[capability](raw)
