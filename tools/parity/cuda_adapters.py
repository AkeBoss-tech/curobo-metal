"""Asset-independent adapters for executing pinned cuRobo probes on CUDA."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import tempfile

import numpy as np


Output = dict[str, np.ndarray]


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
    from curobo._src.cost.cost_tool_pose import ToolPoseCost
    from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
    from curobo._src.types.tool_pose import GoalToolPose, ToolPose
    from curobo.types import DeviceCfg

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
    value, _, _, _ = cost.forward(current, goal)
    scalar = value.sum(dim=-1).reshape(-1)
    scalar.sum().backward()
    invalid = ToolPose(["wrong"], position.detach(), quaternion)
    return {
        "value": scalar.detach().cpu().numpy(),
        "position_gradient": position.grad.reshape(-1, 3).cpu().numpy(),
        "invalid_rejected": _invalid_rejected(
            lambda: cost.forward(invalid, goal)
        ),
    }


ADAPTERS: dict[str, Callable[[dict[str, np.ndarray]], Output]] = {
    "collision.robot_scene": _robot_scene_collision,
    "configuration.robot_config_and_loaders": _robot_config,
    "cost.pose_and_composable_costs": _pose_cost,
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
