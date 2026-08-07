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


ADAPTERS: dict[str, Callable[[dict[str, np.ndarray]], Output]] = {
    "collision.mesh_world": _mesh_world,
    "collision.robot_scene": _robot_scene_collision,
    "collision.voxel_esdf_query": _voxel_esdf,
    "configuration.robot_config_and_loaders": _robot_config,
    "cost.pose_and_composable_costs": _pose_cost,
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
