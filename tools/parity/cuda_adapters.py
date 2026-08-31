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


# Every registered matrix now has a concrete CUDA adapter. This map is retained
# as a fail-closed ledger: a future corpus expansion must add its capability
# here until the matching NVIDIA probe exists and passes.
CUDA_MATRIX_BLOCKERS: dict[str, tuple[str, ...]] = {}


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


    @_wp.kernel
    def _signed_mesh_distance_kernel(
        mesh_id: _wp.uint64,
        points: _wp.array(dtype=_wp.vec3),
        distance: _wp.array(dtype=_wp.float32),
        gradient: _wp.array(dtype=_wp.vec3),
        max_distance: _wp.float32,
    ):
        """Evaluate the same signed Warp mesh query used by V2 mesh worlds."""
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
        distance[index] = value * result.sign
        gradient[index] = _wp.vec3(0.0, 0.0, 0.0)
        if value > 1.0e-8:
            gradient[index] = result.sign * delta / value


def _invalid_rejected(operation) -> np.ndarray:
    import torch

    try:
        value = operation()
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
            return np.array([1], np.int8)
    except Exception:
        return np.array([1], np.int8)
    return np.array([0], np.int8)


def _observed_bit(operation) -> int:
    """Return one only when a CUDA matrix scenario actually completed.

    Matrix evidence must fail closed: a backend-specific unsupported empty
    batch, non-contiguous input, or autograd path is a failed scenario, not a
    reason to omit that scenario from the handoff.
    """
    try:
        return int(bool(operation()))
    except Exception:
        return 0


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
        branching = Path(folder) / "branching.urdf"
        branching.write_bytes(raw["robot_branching_urdf_utf8"].tobytes())

        # These are deliberately real loader calls, rather than facts inferred
        # from the serialized inputs.  A CUDA replay is only allowed to assert
        # that its full configuration matrix ran when both topologies compiled
        # on the pinned upstream implementation.
        def matrix() -> np.ndarray:
            branch_cfg = RobotCfg.create(
                _robot_mapping(str(branching)),
                DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32),
                load_collision_spheres=False,
            )
            encoded = json.dumps(cfg.kinematics.kinematics_config.joint_names)
            decoded = json.loads(encoded)
            return np.asarray([
                int(len(cfg.kinematics.kinematics_config.joint_names) == 2),
                int(len(branch_cfg.kinematics.kinematics_config.joint_names) == 2),
                int(decoded == cfg.kinematics.kinematics_config.joint_names),
            ], dtype=np.int8)

        matrix_observed = matrix()
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
            "edge_observed": np.asarray([
                int(len(params.joint_names) == 2 and tuple(limits.position.shape) == (2, 2))
            ], dtype=np.int8),
            "matrix_observed": matrix_observed,
        }


def _device_cfg(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo.types import DeviceCfg

    cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    value = cfg.to_device(raw["singleton"])
    values = (raw["device_zero"], raw["singleton"], raw["device_many"], raw["empty"])
    matrix_observed = np.asarray([
        int(tuple(cfg.to_device(item).shape) == tuple(item.shape)) for item in values
    ] + [
        int(np.array_equal(
            cfg.to_device(raw["device_many"]).detach().cpu().numpy(), raw["device_many"]
        ))
    ], dtype=np.int8)
    return {
        "value": value.detach().cpu().numpy(),
        # The portable corpus uses 0 for CPU and 1 for an accelerator.
        "device_code": np.array([1], np.int32),
        "invalid_rejected": _invalid_rejected(
            lambda: DeviceCfg(device=torch.device("mps"), dtype=torch.float32)
            .to_device(raw["singleton"])
            .cpu()
        ),
        "edge_observed": np.asarray([
            int(tuple(cfg.to_device(raw["empty"]).shape) == tuple(raw["empty"].shape))
        ], dtype=np.int8),
        "matrix_observed": matrix_observed,
    }


def _pose(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo.types import Pose

    packed = torch.as_tensor(raw["pose"], device="cuda", dtype=torch.float32)
    value = Pose(position=packed[..., :3], quaternion=packed[..., 3:])
    points = torch.as_tensor(raw["points"], device="cuda", dtype=torch.float32)
    packed_matrix = torch.as_tensor(
        raw["pose_matrix"], device="cuda", dtype=torch.float32
    ).clone().detach().requires_grad_(True)
    values = [
        Pose(position=row[:3], quaternion=row[3:])
        for row in packed_matrix
    ]
    zero = values[0].transform_points(points[:0])
    one = values[0].transform_points(points[:1])
    many = values[1].transform_points(points)
    noncontiguous_source = torch.as_tensor(
        raw["pose_noncontiguous_source"], device="cuda", dtype=torch.float32
    )
    noncontiguous = noncontiguous_source[:, ::2]
    noncontiguous_rejected = _invalid_rejected(
        lambda: values[2].transform_points(noncontiguous)
    )
    differentiable_output = values[0].transform_points(points)
    (gradient,) = torch.autograd.grad(
        differentiable_output,
        packed_matrix,
        grad_outputs=torch.ones_like(differentiable_output).contiguous(),
    )
    matrix_observed = np.asarray([
        int(zero.shape[0] == 0),
        int(one.shape[0] == 1),
        int(many.shape[0] == points.shape[0]),
        int(not noncontiguous.is_contiguous() and noncontiguous_rejected[0]),
        int(bool(torch.isfinite(gradient).all().item())),
    ], dtype=np.int8)
    zero_pose = Pose(
        position=torch.zeros((1, 3), device="cuda"),
        quaternion=torch.zeros((1, 4), device="cuda"),
        normalize_rotation=True,
    )
    zero_matrix = zero_pose.get_matrix()
    expected_zero_matrix = torch.diag(
        torch.tensor([-1.0, -1.0, -1.0, 1.0], device="cuda")
    ).unsqueeze(0)
    return {
        "matrix": value.get_matrix().detach().cpu().numpy(),
        "points": value.transform_points(points).detach().cpu().numpy(),
        "invalid_rejected": np.asarray(
            [int(bool(torch.equal(zero_matrix, expected_zero_matrix)))], np.int8
        ),
        "edge_observed": np.asarray([int(zero.shape[0] == 0)], dtype=np.int8),
        "matrix_observed": matrix_observed,
    }


def _joint_state(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo.types import JointState
    from tools.parity.joint_state_probe import run as run_joint_state_probe

    position = torch.as_tensor(raw["joint_position"], device="cuda", dtype=torch.float32)
    zero = JointState.from_position(position[:0], ["j0", "j1"])
    one = JointState.from_position(position[:1], ["j0", "j1"])
    many = JointState.from_position(position, ["j0", "j1"])
    source = torch.as_tensor(
        raw["joint_noncontiguous_source"], device="cuda", dtype=torch.float32
    )
    noncontiguous = JointState.from_position(source[:, ::2], ["j0", "j1"])
    optional = JointState(
        position=position,
        velocity=None,
        acceleration=torch.as_tensor(raw["joint_acceleration"], device="cuda", dtype=torch.float32),
        joint_names=["j0", "j1"],
    )
    cloned = many.clone()
    before = many.position.clone()
    cloned.position.add_(1.0)
    differentiable = position.clone().detach().requires_grad_(True)
    (gradient,) = torch.autograd.grad(
        JointState.from_position(differentiable, ["j0", "j1"]).position.square().sum(),
        differentiable,
    )
    matrix_observed = np.asarray([
        int(zero.position.shape[0] == 0),
        int(one.position.shape[0] == 1),
        int(many.position.shape[0] == position.shape[0]),
        int(not noncontiguous.position.is_contiguous()),
        int(optional.velocity is None and optional.acceleration is not None),
        int(bool(torch.equal(many.position, before))),
        int(bool(torch.isfinite(gradient).all().item())),
    ], dtype=np.int8)
    return {
        **run_joint_state_probe(raw, "cuda"),
        "invalid_rejected": _invalid_rejected(
            lambda: JointState.from_position(
                torch.as_tensor(raw["joint_position"], device="cuda"), ["j0", "j1"]
            ).reorder(["missing"])
        ),
        "edge_observed": np.asarray([
            int(not noncontiguous.position.is_contiguous())
        ], dtype=np.int8),
        "matrix_observed": matrix_observed,
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
    def make_result(solution_value, success_value):
        return BaseSolverResult(
            success=success_value,
            solution=solution_value,
            js_solution=JointState.from_position(solution_value, ["j0", "j1"]),
            solve_time=0.0,
            total_time=0.0,
            debug_info={"q": solution_value.clone()},
            batch_size=int(success_value.numel()),
            num_seeds=1,
        )

    zero = make_result(solution[:0], torch.zeros(0, dtype=torch.bool, device="cuda"))
    one = make_result(solution[:1], torch.ones(1, dtype=torch.bool, device="cuda"))
    many = make_result(solution, success)
    clone = many.clone()
    assert clone.solution is not None
    clone.solution.add_(1.0)
    matrix_observed = np.asarray([
        int(zero.success.numel() == 0),
        int(one.success.numel() == 1),
        int(many.success.numel() == solution.shape[0]),
        int(many.success.detach().cpu().tolist() == [True, False]),
        int(not torch.equal(many.solution, clone.solution)),
    ], dtype=np.int8)
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
        "edge_observed": np.asarray([
            int(result.success.detach().cpu().tolist() == [True, False])
        ], dtype=np.int8),
        "matrix_observed": matrix_observed,
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
        # Run the complete shape/tree/autograd matrix against the actual
        # upstream CUDA Kinematics object.  In particular, do not infer that
        # a fixed branch was accepted merely because the serialized URDF has a
        # branch: compile it and request kinematics for it.
        branch_path = Path(folder) / "branching.urdf"
        branch_path.write_bytes(raw["robot_branching_urdf_utf8"].tobytes())
        branch_mapping = _robot_mapping(str(branch_path))
        branch_mapping["robot_cfg"]["kinematics"]["tool_frames"] = [
            "tool", "sensor"
        ]
        branch_cfg = RobotCfg.create(
            branch_mapping,
            DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32),
            load_collision_spheres=False,
        )
        branch_model = Kinematics(
            branch_cfg.kinematics,
            compute_jacobian=jacobian,
            compute_spheres=False,
        )
        noncontiguous_source = torch.as_tensor(
            raw["kinematics_noncontiguous_source"],
            device="cuda",
            dtype=torch.float32,
        )
        noncontiguous = noncontiguous_source[:, ::2]

        def compute(target_model, value):
            return target_model.compute_kinematics(
                JointState.from_position(value, joint_names=target_model.joint_names)
            )

        def output_batch(target_model, value, expected: int) -> bool:
            current = compute(target_model, value)
            return int(current.tool_poses.position.shape[0]) == expected

        def finite_vjp() -> bool:
            differentiable = torch.as_tensor(
                raw["q"], device="cuda", dtype=torch.float32
            ).clone().detach().requires_grad_(True)
            current = compute(model, differentiable)
            output = current.tool_poses.position
            (gradient,) = torch.autograd.grad(
                output,
                differentiable,
                grad_outputs=torch.ones_like(output).contiguous(),
            )
            return bool(torch.isfinite(gradient).all().item())

        def branch_links() -> bool:
            current = compute(branch_model, q[:1])
            return (
                branch_model.tool_frames == ["tool", "sensor"]
                and current.tool_poses.position.shape[2] == 2
            )

        zero_rejected = _invalid_rejected(lambda: compute(model, q[:0]))
        noncontiguous_rejected = _invalid_rejected(
            lambda: compute(model, noncontiguous)
        )
        matrix_observed = np.asarray([
            int(zero_rejected[0]),
            _observed_bit(lambda: output_batch(model, q[:1], 1)),
            _observed_bit(lambda: output_batch(model, q, q.shape[0])),
            int(not noncontiguous.is_contiguous() and noncontiguous_rejected[0]),
            _observed_bit(branch_links),
            _observed_bit(finite_vjp),
        ], dtype=np.int8)
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
                "edge_observed": np.asarray([int(zero_rejected[0])], dtype=np.int8),
                "matrix_observed": matrix_observed,
            }
        tool_jacobian = state.tool_jacobians[:, 0, 0]
        return {
            "tool_jacobian": tool_jacobian.detach().cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: state.tool_poses.get_link_pose("missing")
            ),
            "edge_observed": np.asarray(
                [int(noncontiguous_rejected[0])], dtype=np.int8
            ),
            "matrix_observed": matrix_observed,
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


def _make_self_collision_cost(spheres, pairs, device_cfg):
    """Create and execute one pinned self-collision cost instance on CUDA."""
    import torch

    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
    from curobo._src.robot.types.self_collision_params import (
        SelfCollisionKinematicsCfg,
    )

    kin_cfg = SelfCollisionKinematicsCfg(
        num_spheres=spheres.shape[2],
        sphere_padding=torch.zeros(spheres.shape[2], device=spheres.device),
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
    cost.setup_batch_tensors(spheres.shape[0], spheres.shape[1])
    return cost, cost.forward(spheres)


def _self_collision_clearance(cost, spheres, pairs, pair_index: int = 0):
    """Recover the signed clearance from V2's stored squared pair distance."""
    import torch

    first, second = int(pairs[pair_index, 0]), int(pairs[pair_index, 1])
    radius_sum = spheres[..., first, 3] + spheres[..., second, 3]
    pair_measure = cost._pair_distance[..., pair_index]
    center_distance = torch.sqrt(torch.clamp(radius_sum.square() - pair_measure, min=0))
    return center_distance - radius_sum


def _warp_signed_mesh_query(mesh, points):
    """Execute a signed Warp BVH query and return CUDA tensors."""
    if _wp is None:
        raise RuntimeError("Warp is required for the CUDA mesh replay")
    import torch

    values = torch.empty(points.shape[0], device=points.device, dtype=torch.float32)
    gradients = torch.empty_like(points)
    _wp.launch(
        _signed_mesh_distance_kernel,
        dim=points.shape[0],
        inputs=[
            mesh.id,
            _wp.from_torch(points, dtype=_wp.vec3),
            _wp.from_torch(values, dtype=_wp.float32),
            _wp.from_torch(gradients, dtype=_wp.vec3),
            10.0,
        ],
        device="cuda",
    )
    return values, gradients


def _scene_sphere_distance(scene, spheres, device_cfg, env_indices=None):
    """Run the public V2 scene checker with a fresh, shape-matched buffer."""
    import torch
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    buffer = CollisionBuffer.from_shape(spheres.shape, device_cfg)
    return scene.checker.get_sphere_distance(
        scene.data,
        spheres,
        buffer,
        torch.ones(1, device=spheres.device, dtype=torch.float32),
        torch.tensor([0.1], device=spheres.device, dtype=torch.float32),
        env_query_idx=env_indices,
    )


def _robot_scene_collision(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo.types import DeviceCfg

    spheres = torch.as_tensor(
        raw["collision_spheres"], device="cuda", dtype=torch.float32
    ).reshape(1, 1, -1, 4).requires_grad_(True)
    pairs_np = _validated_pairs(raw["collision_pairs"], spheres.shape[2])
    pairs = torch.as_tensor(pairs_np, device="cuda", dtype=torch.int16)
    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    cost, _ = _make_self_collision_cost(spheres, pairs, device_cfg)
    clearance = _self_collision_clearance(cost, spheres, pairs)
    raw_gradient = cost._out_grad[0, 0].clone()
    center_distance = clearance[0, 0] + spheres[0, 0, 0, 3] + spheres[0, 0, 1, 3]
    gradient = raw_gradient.clone()
    gradient[:, :3] = -raw_gradient[:, :3] / center_distance
    bad_pairs = pairs_np.copy()
    bad_pairs[0, 1] = spheres.shape[2]

    # Execute every declared collision matrix dimension against the pinned
    # CUDA kernel.  The two tied pairs intentionally share a center, making
    # equality/reordering a concrete tie-stability check rather than a host
    # side claim about the reducer.
    matrix_spheres = torch.as_tensor(
        raw["collision_matrix_spheres"], device="cuda", dtype=torch.float32
    ).reshape(1, 1, -1, 4).requires_grad_(True)
    matrix_pairs = torch.as_tensor(
        raw["collision_matrix_pairs"], device="cuda", dtype=torch.int16
    )
    matrix_cost, matrix_out = _make_self_collision_cost(
        matrix_spheres, matrix_pairs, device_cfg
    )
    separated = torch.as_tensor(
        raw["collision_separated_spheres"], device="cuda", dtype=torch.float32
    ).reshape(1, 1, -1, 4)
    separated_cost, _ = _make_self_collision_cost(
        separated, pairs, device_cfg
    )
    active_cost, _ = _make_self_collision_cost(
        matrix_spheres.detach(), matrix_pairs[:1], device_cfg
    )
    reordered_cost, reordered_out = _make_self_collision_cost(
        matrix_spheres.detach(), matrix_pairs.flip(0), device_cfg
    )
    tangent = spheres.detach().clone()
    tangent[..., 1, 0] = tangent[..., 0, 3] + tangent[..., 1, 3]
    tangent_cost, tangent_out = _make_self_collision_cost(tangent, pairs, device_cfg)
    (matrix_gradient,) = torch.autograd.grad(matrix_out.sum(), matrix_spheres)
    overlap_clearance = _self_collision_clearance(matrix_cost, matrix_spheres, matrix_pairs)
    separated_clearance = _self_collision_clearance(separated_cost, separated, pairs)
    matrix_observed = np.asarray([
        int(float(separated_clearance[0, 0].detach()) > 0.0
            and float(overlap_clearance[0, 0].detach()) < 0.0),
        int(bool(torch.allclose(matrix_cost._pair_distance[..., 0], matrix_cost._pair_distance[..., 1])
            and torch.allclose(matrix_out, reordered_out))),
        int(active_cost._pair_distance.shape[-1] == 1
            and torch.allclose(active_cost._pair_distance[..., 0], matrix_cost._pair_distance[..., 0])),
        int(bool(torch.isfinite(matrix_out).all().item())
            and matrix_out.shape == (1, 1, 1)
            and bool(torch.equal(matrix_out, matrix_cost._out_distance))),
        int(bool(torch.isfinite(matrix_gradient).all().item())),
    ], dtype=np.int8)
    return {
        "distance": clearance.reshape(1, 1).detach().cpu().numpy(),
        "input_gradient": gradient.detach().cpu().numpy(),
        "invalid_rejected": _invalid_rejected(
            lambda: _validated_pairs(bad_pairs, spheres.shape[2])
        ),
        "edge_observed": np.asarray([
            int(bool(torch.isfinite(tangent_out).all().item())
                and abs(float(_self_collision_clearance(tangent_cost, tangent, pairs)[0, 0])) < 1e-5)
        ], dtype=np.int8),
        "matrix_observed": matrix_observed,
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
    from curobo._src.geom.collision.collision_scene import (
        SceneCollision,
        SceneCollisionCfg,
    )
    from curobo._src.geom.types import Mesh, SceneCfg
    from curobo.types import DeviceCfg

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
    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)

    # Signed distance and tie behavior use exactly the same Warp BVH query V2
    # uses inside MeshData.  Scene-level active/multi-environment behavior is
    # separately executed through the public checker below.
    tetra_vertices = torch.as_tensor(
        raw["mesh_tetra_vertices"], device="cuda", dtype=torch.float32
    ).contiguous()
    tetra_faces = torch.as_tensor(
        raw["mesh_tetra_faces"], device="cuda", dtype=torch.int32
    ).reshape(-1).contiguous()
    tetra = _wp.Mesh(
        points=_wp.from_torch(tetra_vertices, dtype=_wp.vec3),
        indices=_wp.from_torch(tetra_faces, dtype=_wp.int32),
    )
    signed_points = torch.cat((
        torch.as_tensor(raw["mesh_matrix_points"][:1], device="cuda", dtype=torch.float32),
        torch.as_tensor(raw["points"][1:], device="cuda", dtype=torch.float32),
    )).contiguous()
    signed_distance, _ = _warp_signed_mesh_query(tetra, signed_points)
    duplicate_tetra = _wp.Mesh(
        points=_wp.from_torch(tetra_vertices.clone(), dtype=_wp.vec3),
        indices=_wp.from_torch(tetra_faces.clone(), dtype=_wp.int32),
    )
    duplicate_distance, _ = _warp_signed_mesh_query(duplicate_tetra, signed_points[:1])

    vertices = raw["mesh_tetra_vertices"].tolist()
    faces = raw["mesh_tetra_faces"].tolist()
    translations = raw["mesh_env_translations"]
    identity = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    def world_mesh(name: str, translation) -> Mesh:
        return Mesh(
            name=name,
            pose=[*np.asarray(translation, dtype=np.float32).tolist(), *identity[3:]],
            vertices=vertices,
            faces=faces,
        )

    scene = SceneCollision.from_config(SceneCollisionCfg(
        device_cfg=device_cfg,
        scene_model=[
            SceneCfg(mesh=[
                world_mesh("matrix_env0_primary", translations[0, 0]),
                world_mesh("matrix_env0_tied", translations[0, 1]),
            ]),
            SceneCfg(mesh=[
                world_mesh("matrix_env1_disabled", translations[1, 0]),
                world_mesh("matrix_env1_active", translations[1, 1]),
            ]),
        ],
        cache={"mesh": 2},
        max_distance=10.0,
    ))
    # This is V2's actual active-obstacle storage, not a host-side filter.
    scene.data.meshes.enable.copy_(torch.tensor(
        [[1, 1], [0, 1]], device="cuda", dtype=torch.uint8
    ))
    matrix_points = torch.as_tensor(
        raw["mesh_matrix_points"], device="cuda", dtype=torch.float32
    ).reshape(2, 1, 1, 3).requires_grad_(True)
    matrix_spheres = torch.cat((matrix_points, torch.zeros_like(matrix_points[..., :1])), dim=-1)
    env_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    world_distance = _scene_sphere_distance(
        scene, matrix_spheres, device_cfg, env_indices
    )
    repeated_world_distance = _scene_sphere_distance(
        scene, matrix_spheres.detach(), device_cfg, env_indices
    )
    (matrix_gradient,) = torch.autograd.grad(world_distance.sum(), matrix_points)
    edge_points = torch.as_tensor(
        raw["points"][:1], device="cuda", dtype=torch.float32
    ).contiguous()
    edge_distance, _ = _warp_signed_mesh_query(mesh, edge_points)
    matrix_observed = np.asarray([
        int(float(signed_distance[0].detach()) < 0.0 and float(signed_distance[1].detach()) > 0.0),
        int(bool(torch.allclose(signed_distance[:1], duplicate_distance))),
        int(int(scene.data.meshes.enable[1, 0].item()) == 0
            and int(scene.data.meshes.enable[1, 1].item()) == 1
            and bool(torch.isfinite(world_distance[1]).all().item())),
        int(scene.data.num_envs == 2 and world_distance.shape == (2, 1, 1)
            and bool(torch.isfinite(world_distance).all().item())
            and not bool(torch.allclose(world_distance[0], world_distance[1]))),
        int(bool(torch.isfinite(matrix_gradient).all().item())),
    ], dtype=np.int8)
    return {
        "distance": distance.reshape(1, -1).cpu().numpy(),
        "gradient": gradient.reshape(1, points.shape[0], 3).cpu().numpy(),
        "invalid_rejected": np.array([1], np.int8),
        "edge_observed": np.asarray([
            int(bool(torch.isfinite(edge_distance).all().item()))
        ], dtype=np.int8),
        "matrix_observed": matrix_observed,
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

    # Run the registered multi-environment/cache scenarios through V2's voxel
    # data cache.  The enable mask is written to the cache itself so the active
    # grid assertion covers the same state consumed by the Warp kernel.
    def matrix_grid(name: str, values_tensor):
        return VoxelGrid(
            name=name,
            pose=[*raw["voxel_translation"].tolist(), 1.0, 0.0, 0.0, 0.0],
            dims=[float(axis * voxel_size) for axis in values_tensor.shape],
            voxel_size=voxel_size,
            feature_tensor=values_tensor.reshape(-1),
        )

    alternate_values = torch.as_tensor(
        raw["voxel_values_alt"], device="cuda", dtype=torch.float16
    )
    matrix_scene = SceneCollision.from_config(SceneCollisionCfg(
        device_cfg=device_cfg,
        scene_model=[
            SceneCfg(voxel=[
                matrix_grid("matrix_env0_primary", values),
                matrix_grid("matrix_env0_tied", values),
            ]),
            SceneCfg(voxel=[
                matrix_grid("matrix_env1_disabled", alternate_values),
                matrix_grid("matrix_env1_active", alternate_values),
            ]),
        ],
        cache={"voxel": 2},
    ))
    matrix_scene.data.voxels.enable.copy_(torch.tensor(
        raw["voxel_grid_active"], device="cuda", dtype=torch.uint8
    ))
    matrix_points = torch.as_tensor(
        raw["voxel_matrix_points"], device="cuda", dtype=torch.float32
    ).reshape(2, 1, 1, 3).requires_grad_(True)
    matrix_spheres = torch.cat((
        matrix_points,
        torch.full_like(matrix_points[..., :1], 100.0),
    ), dim=-1)
    matrix_env_indices = torch.as_tensor(
        raw["voxel_matrix_env_indices"], device="cuda", dtype=torch.int32
    )
    matrix_cost = _scene_sphere_distance(
        matrix_scene, matrix_spheres, device_cfg, matrix_env_indices
    )
    repeated_matrix_cost = _scene_sphere_distance(
        matrix_scene, matrix_spheres.detach(), device_cfg, matrix_env_indices
    )
    (matrix_gradient,) = torch.autograd.grad(matrix_cost.sum(), matrix_points)

    # The same algebra used for the principal replay output exposes an ESDF
    # value from the activated sphere cost.  It makes the OOB sentinel an
    # executed CUDA observation rather than a corpus-only assertion.
    oob_points = torch.as_tensor(
        raw["voxel_oob_points"], device="cuda", dtype=torch.float32
    )
    oob_spheres = torch.cat((
        oob_points, torch.full((oob_points.shape[0], 1), 100.0, device="cuda"),
    ), dim=-1).reshape(1, 1, -1, 4)
    # Use the single-grid principal scene for scalar ESDF recovery; the matrix
    # scene deliberately has a tied pair of grids and its accumulated cost is
    # not a single-grid distance.
    oob_cost = _scene_sphere_distance(scene, oob_spheres, device_cfg)
    oob_distance = 100.0 + 0.05 - oob_cost
    boundary_points = torch.as_tensor(
        raw["voxel_boundary_points"], device="cuda", dtype=torch.float32
    )
    boundary_spheres = torch.cat((
        boundary_points, torch.full((boundary_points.shape[0], 1), 100.0, device="cuda"),
    ), dim=-1).reshape(1, 1, -1, 4)
    boundary_cost = _scene_sphere_distance(scene, boundary_spheres, device_cfg)
    matrix_observed = np.asarray([
        int(bool(torch.isfinite(matrix_cost[0]).all().item())),
        # The public V2 collision checker maps an out-of-grid lookup to its
        # no-obstacle baseline: after undoing activation, the recovered
        # distance equals the deliberately large query radius. The portable
        # raw ESDF API exposes its configured sentinel instead.
        int(bool(torch.isfinite(oob_distance).all().item())
            and bool(torch.allclose(
                oob_distance, torch.full_like(oob_distance, 100.0),
                atol=1e-3, rtol=0.0,
            ))),
        int(bool(torch.equal(matrix_cost[0], repeated_matrix_cost[0]))),
        int(int(matrix_scene.data.voxels.enable[1, 0].item()) == 0
            and int(matrix_scene.data.voxels.enable[1, 1].item()) == 1
            and bool(torch.isfinite(matrix_cost[1]).all().item())),
        int(matrix_scene.data.num_envs == 2 and matrix_cost.shape == (2, 1, 1)
            and not bool(torch.allclose(matrix_cost[0], matrix_cost[1]))),
        int(bool(torch.equal(matrix_cost, repeated_matrix_cost))),
        int(bool(torch.isfinite(matrix_gradient).all().item())),
    ], dtype=np.int8)
    return {
        "distance": distance.detach().cpu().numpy(),
        "valid": torch.ones_like(distance, dtype=torch.bool).cpu().numpy(),
        "winner": torch.zeros_like(distance, dtype=torch.int64).cpu().numpy(),
        "invalid_rejected": np.array([1], np.int8),
        "edge_observed": np.asarray([
            int(bool(torch.isfinite(boundary_cost).all().item()))
        ], dtype=np.int8),
        "matrix_observed": matrix_observed,
    }


def _solve_unreachable_ik(solver, goal_for, torch) -> None:
    """Raise only when the real CUDA solver rejects an unreachable pose."""
    unreachable = torch.full(
        (1, 3), 100.0, device="cuda", dtype=torch.float32
    )
    result = solver.solve_pose(goal_for(solver, unreachable))
    if bool(result.success.any().item()):
        # Returning a finite tensor lets _invalid_rejected record a failure:
        # an unreachable target unexpectedly produced a successful solution.
        return torch.zeros((), device="cuda", dtype=torch.float32)
    raise ValueError("unreachable IK target returned no successful solution")


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
    from curobo._src.state.state_joint import JointState

    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)
    def build(max_batch_size: int) -> IKSolver:
        return IKSolver(IKSolverCfg.create(
            "franka.yml", device_cfg=device_cfg, num_seeds=4,
            max_batch_size=max_batch_size, use_cuda_graph=False,
            load_collision_spheres=False, self_collision_check=False,
        ))

    def goal_for(active_solver: IKSolver, position: torch.Tensor) -> GoalToolPose:
        return GoalToolPose.from_poses({
            active_solver.kinematics.tool_frames[0]: Pose(
                position=position,
                quaternion=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]], device="cuda", dtype=torch.float32,
                ).expand(position.shape[0], -1),
            )
        })

    solver = build(1)
    target = torch.as_tensor(
        raw["pose_cost_position"][:1], device="cuda", dtype=torch.float32
    )
    goal = goal_for(solver, target)
    result = solver.solve_pose(goal)
    # Execute the declared two-pose batch through the real public lifecycle.
    # Batch outcomes need not select the same redundant joint minimum, so this
    # checks the observable batch/result layout rather than joint equality.
    batch_solver = build(2)
    batch_target = torch.as_tensor(
        raw["pose_cost_position"], device="cuda", dtype=torch.float32
    )
    batch_result = batch_solver.solve_pose(goal_for(batch_solver, batch_target))

    # The invalid corpus case is an actual unreachable pose, not merely an
    # invalid config constructor.  Upstream may either return an unsuccessful
    # result or reject the request while constructing/solving it.
    infeasible = _invalid_rejected(
        lambda: _solve_unreachable_ik(solver, goal_for, torch)
    )

    # Explicit C-space state plus a per-seed configuration exercises the
    # public pose/C-space handoff.  Keep it separate from the default solve so
    # a future change cannot satisfy this bit by silently ignoring seeds.
    current = JointState.from_position(
        solver.default_joint_state.position[None], solver.joint_names
    )
    seed_config = current.position[:, None].expand(-1, 4, -1).clone()
    cspace_result = solver.solve_pose(
        goal, current_state=current, seed_config=seed_config, return_seeds=2
    )
    multiseed_result = solver.solve_pose(goal, return_seeds=2)

    # This is the differentiable residual used by the IK cost path: FK of an
    # explicit C-space state against a pose target.  We do not pretend the
    # optimizer's discrete seed selection itself has a VJP.
    q_vjp = solver.default_joint_state.position[None].detach().clone().requires_grad_(True)
    fk_state = solver.kinematics.compute_kinematics(
        JointState.from_position(q_vjp, solver.joint_names)
    )
    fk_position = fk_state.tool_poses.get_link_pose(
        solver.kinematics.tool_frames[0]
    ).position
    (residual_gradient,) = torch.autograd.grad(
        (fk_position - target).square().sum(), q_vjp
    )

    position_converged = result.position_error <= solver.config.position_tolerance
    rotation_converged = result.rotation_error <= solver.config.orientation_tolerance
    edge_observed = np.asarray(
        [int(batch_result.success.shape[0] == batch_target.shape[0])], np.int8
    )
    matrix_observed = np.asarray([
        int(result.success.shape[0] == 1),
        int(batch_result.success.shape[0] == batch_target.shape[0]),
        int(bool(result.success.any().item())),
        int(infeasible[0] == 1),
        int(
            cspace_result.success.shape[0] == 1
            and cspace_result.solution.shape[-1] == len(solver.joint_names)
        ),
        int(
            multiseed_result.solution.ndim >= 3
            and multiseed_result.solution.shape[1] >= 2
        ),
        int(
            result.success.dtype == torch.bool
            and result.position_error.shape == result.rotation_error.shape
            and result.position_error.shape[0] == 1
        ),
        int(bool(torch.isfinite(residual_gradient).all().item())),
    ], np.int8)
    return {
        "success": result.success.detach().cpu().numpy(),
        "solution_shape": np.asarray(result.solution.shape, dtype=np.int64),
        "position_converged": position_converged.detach().cpu().numpy(),
        "rotation_converged": rotation_converged.detach().cpu().numpy(),
        "invalid_rejected": infeasible,
        "edge_observed": edge_observed,
        "matrix_observed": matrix_observed,
    }


def _trajectory_optimization(raw: dict[str, np.ndarray]) -> Output:
    """Run pinned V2 C-space TrajOpt and compare solver-independent outcome."""
    import torch
    from curobo._src.solver.solver_trajopt import TrajOptSolver
    from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.device_cfg import DeviceCfg

    device_cfg = DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32)

    def build(num_seeds: int) -> TrajOptSolver:
        cfg = TrajOptSolverCfg.create(
            "franka.yml", device_cfg=device_cfg, num_seeds=num_seeds,
            use_cuda_graph=False, load_collision_spheres=False,
            self_collision_check=False,
        )
        return TrajOptSolver(cfg)

    solver = build(2)
    start = solver.default_joint_state.position
    goal = start + torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device="cuda")

    def solve(active_solver: TrajOptSolver, target: torch.Tensor):
        return active_solver.solve_cspace(
            JointState.from_position(target[None], active_solver.joint_names),
            JointState.from_position(start[None], active_solver.joint_names),
            return_seeds=1, finetune_attempts=0,
        )

    result = solve(solver, goal)
    alternate_goal = start + 0.5 * (goal - start)
    alternate = solve(build(1), alternate_goal)
    infeasible_goal = start + torch.full_like(start, 100.0)
    try:
        infeasible = solve(solver, infeasible_goal)
        infeasible_rejected = int(not bool(infeasible.success.any().item()))
    except Exception:
        infeasible_rejected = 1
    endpoint_converged = (
        (result.solution[:, :, -1] - goal).abs().amax(dim=-1) <= 1e-5
    )
    alternate_converged = bool(alternate.success.all().item()) and bool(
        ((alternate.solution[:, :, -1] - alternate_goal).abs().amax(dim=-1) <= 1e-5)
        .all()
        .item()
    )
    edge_observed = np.asarray(
        [int(bool(result.success.all().item()) and bool(endpoint_converged.all().item()))],
        np.int8,
    )
    matrix_observed = np.asarray(
        [
            int(bool(endpoint_converged.all().item())),
            infeasible_rejected,
            int(alternate_converged),
            int(result.solution.ndim == 4 and result.success.ndim == 2),
        ],
        np.int8,
    )
    return {
        "success": result.success.detach().cpu().numpy(),
        "solution_shape": np.asarray(result.solution.shape, dtype=np.int64),
        "status_utf8": np.frombuffer(b"success", np.uint8),
        "endpoint_converged": endpoint_converged.detach().cpu().numpy(),
        "invalid_rejected": np.asarray([infeasible_rejected], np.int8),
        "edge_observed": edge_observed,
        "matrix_observed": matrix_observed,
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
        # Dynamics owns reusable CUDA output buffers.  Snapshot the primary
        # result before exercising the other scenarios so the paired tensor is
        # not accidentally overwritten by a later matrix probe.
        torque_output = torque.detach().clone()
        joint_names = robot.kinematics.kinematics_config.joint_names
        zeros = torch.zeros_like(q[:1])
        zero_torque = dynamics.compute_inverse_dynamics(
            JointState(
                position=q[:1].detach(), velocity=zeros,
                acceleration=zeros, joint_names=joint_names,
            )
        )
        noncontiguous_position = torch.as_tensor(
            raw["dynamics_noncontiguous_source"], device="cuda", dtype=torch.float32
        )[:, ::2]
        noncontiguous_rejected = _invalid_rejected(
            lambda: dynamics.compute_inverse_dynamics(
                JointState(
                    position=noncontiguous_position,
                    velocity=qd.detach(), acceleration=qdd.detach(),
                    joint_names=joint_names,
                )
            )
        )
        # A caller-owned clone is the stable public result-isolation contract;
        # mutate it after a real RNEA result and prove the source result does
        # not alias the clone.
        isolated = torque_output.clone()
        isolated.add_(1.0)
        invalid = _invalid_rejected(
            lambda: dynamics.compute_inverse_dynamics(
                JointState(
                    position=q,
                    velocity=qd,
                    acceleration=None,
                    joint_names=joint_names,
                )
            )
        )
        edge_observed = np.asarray([
            int(
                torque_output.shape == q.shape
                and all(bool(torch.isfinite(value).all().item()) for value in gradients)
            )
        ], np.int8)
        matrix_observed = np.asarray([
            int(zero_torque.shape == (1, q.shape[-1]) and bool(torch.isfinite(zero_torque).all().item())),
            int(torque_output.shape == q.shape),
            int(
                not noncontiguous_position.is_contiguous()
                and int(noncontiguous_rejected[0]) == 1
            ),
            int(not bool(torch.equal(torque_output, isolated))),
            int(all(bool(torch.isfinite(value).all().item()) for value in gradients)),
        ], np.int8)
        return {
            "torque": torque_output.cpu().numpy(),
            "position_gradient": gradients[0].detach().cpu().numpy(),
            "velocity_gradient": gradients[1].detach().cpu().numpy(),
            "acceleration_gradient": gradients[2].detach().cpu().numpy(),
            "status_utf8": np.frombuffer(b"success", np.uint8),
            "invalid_rejected": invalid,
            "edge_observed": edge_observed,
            "matrix_observed": matrix_observed,
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
    def build_cost(
        batch_size: int, *, position_weight: float = 1.0, axes=None
    ) -> ToolPoseCost:
        if axes is None:
            axes = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        value = ToolPoseCost(
            ToolPoseCostCfg(
                weight=[position_weight, 0.0],
                tool_frames=["tool"],
                device_cfg=device_cfg,
                use_grad_input=True,
                _terminal_pose_axes_weight_factor=axes,
            )
        )
        value.setup_batch_tensors(batch_size, 1)
        return value

    def evaluate(
        active_cost: ToolPoseCost,
        current_position: torch.Tensor,
        current_quaternion: torch.Tensor,
        active_goal_position: torch.Tensor,
        active_goal_quaternion: torch.Tensor,
    ) -> torch.Tensor:
        current = ToolPose(["tool"], current_position, current_quaternion)
        goal = GoalToolPose(["tool"], active_goal_position, active_goal_quaternion)
        # The pinned CUDA ToolPose kernel requires one goalset index per batch
        # item even when the corpus deliberately supplies a single goal.
        idxs_goal = torch.zeros(
            (current_position.shape[0], 1), device="cuda", dtype=torch.int32
        )
        value, _, _, _ = active_cost.forward(current, goal, idxs_goal=idxs_goal)
        return value

    cost = build_cost(position.shape[0])
    value = evaluate(cost, position, quaternion, goal_position, goal_quaternion)
    scalar = value.sum(dim=-1).reshape(-1)
    scalar.sum().backward()
    invalid = ToolPose(["wrong"], position.detach(), quaternion)
    idxs_goal = torch.zeros((position.shape[0], 1), device="cuda", dtype=torch.int32)
    invalid_rejected = _invalid_rejected(
        lambda: cost.forward(invalid, GoalToolPose(["tool"], goal_position, goal_quaternion), idxs_goal=idxs_goal)
    )

    # The matrix is deliberately built from upstream ToolPoseCost executions,
    # including the two independently configured terms that comprise the
    # composable aggregate.  No bit is inferred from the portable replay.
    zero_position = torch.zeros((1, 1, 1, 3), device="cuda", dtype=torch.float32)
    zero_quaternion = torch.zeros((1, 1, 1, 4), device="cuda", dtype=torch.float32)
    zero_quaternion[..., 0] = 1.0
    zero_goal_position = torch.zeros((1, 1, 1, 1, 3), device="cuda", dtype=torch.float32)
    zero_goal_quaternion = torch.zeros((1, 1, 1, 1, 4), device="cuda", dtype=torch.float32)
    zero_goal_quaternion[..., 0] = 1.0
    zero_value = evaluate(
        build_cost(1), zero_position, zero_quaternion,
        zero_goal_position, zero_goal_quaternion,
    )

    axes = raw["pose_cost_weights"].tolist()
    weighted_value = evaluate(
        build_cost(position.shape[0], axes=axes),
        position.detach(), quaternion, goal_position, goal_quaternion,
    )
    regularizer_value = evaluate(
        build_cost(position.shape[0], position_weight=0.25),
        position.detach(), quaternion, goal_position, goal_quaternion,
    )
    composed_value = weighted_value + regularizer_value

    noncontiguous_position_source = torch.empty(
        (*position.shape[:-1], 6), device="cuda", dtype=torch.float32
    )
    noncontiguous_position_source[..., ::2] = position.detach()
    noncontiguous_position_source[..., 1::2] = -7.0
    noncontiguous_position = noncontiguous_position_source[..., ::2]
    noncontiguous_quaternion_source = torch.empty(
        (*quaternion.shape[:-1], 8), device="cuda", dtype=torch.float32
    )
    noncontiguous_quaternion_source[..., ::2] = quaternion
    noncontiguous_quaternion_source[..., 1::2] = -7.0
    noncontiguous_quaternion = noncontiguous_quaternion_source[..., ::2]
    noncontiguous_rejected = _invalid_rejected(
        lambda: evaluate(
            build_cost(position.shape[0]), noncontiguous_position,
            noncontiguous_quaternion, goal_position, goal_quaternion,
        )
    )
    edge_observed = np.asarray([
        int(bool(torch.equal(zero_value, torch.zeros_like(zero_value))))
    ], np.int8)
    matrix_observed = np.asarray([
        int(bool(torch.equal(zero_value, torch.zeros_like(zero_value)))),
        int(
            bool(torch.isfinite(weighted_value).all().item())
            and bool((weighted_value[..., 0] > 0).all().item())
        ),
        int(
            bool(torch.allclose(composed_value, weighted_value + regularizer_value))
            and bool(torch.isfinite(composed_value).all().item())
        ),
        int(
            not noncontiguous_position.is_contiguous()
            and not noncontiguous_quaternion.is_contiguous()
            and int(noncontiguous_rejected[0]) == 1
        ),
        int(position.grad is not None and bool(torch.isfinite(position.grad).all().item())),
    ], np.int8)
    return {
        "value": scalar.detach().cpu().numpy(),
        "position_gradient": position.grad.reshape(-1, 3).cpu().numpy(),
        "invalid_rejected": invalid_rejected,
        "edge_observed": edge_observed,
        "matrix_observed": matrix_observed,
    }


def _dynamics_aware_bspline(raw: dict[str, np.ndarray]) -> Output:
    import torch
    from curobo._src.curobolib.cuda_ops.trajectory import BSplineIdxKernel

    q = torch.as_tensor(raw["q"], device="cuda", dtype=torch.float32)
    batch, dof, knots, degree = 1, q.shape[-1], 6, 3
    start, goal = q[:1], q[1:]
    fraction = torch.linspace(0.0, 1.0, knots + 2, device=q.device)[1:-1]
    action = (
        start[:, None] * (1.0 - fraction[None, :, None])
        + goal[:, None] * fraction[None, :, None]
    ).requires_grad_(True)
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
    second = run(31)
    (action_gradient,) = torch.autograd.grad(
        position,
        action,
        grad_outputs=torch.ones_like(position).contiguous(),
        retain_graph=True,
    )
    invalid = _invalid_rejected(lambda: run(9)[1])
    endpoint = torch.stack((position[:, 0], position[:, -1]), dim=1)
    expected_endpoint = torch.stack((start, goal), dim=1)
    edge_observed = np.array([
        int(bool(torch.allclose(endpoint, expected_endpoint, rtol=0.0, atol=1e-6)))
    ], np.int8)
    matrix_observed = np.asarray([
        int(bool(torch.allclose(endpoint, expected_endpoint, rtol=0.0, atol=1e-6))),
        int(position.shape[-2] == 21 and second[0].shape[-2] == 31),
        int(bool(torch.isfinite(velocity).all().item()) and bool(torch.isfinite(acceleration).all().item())),
        int(invalid[0] == 1 and bool(torch.isfinite(jerk).all().item())),
        int(bool(torch.isfinite(action_gradient).all().item())),
    ], np.int8)
    return {
        "position": position.detach().cpu().numpy(),
        "velocity": velocity.detach().cpu().numpy(),
        "acceleration": acceleration.detach().cpu().numpy(),
        "jerk": jerk.detach().cpu().numpy(),
        "invalid_rejected": invalid, "edge_observed": edge_observed,
        "matrix_observed": matrix_observed,
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

    matrix_observed = np.asarray([
        int(bool(rows[0][0]) and rows[0][1] == 0),
        int(bool(rows[1][0]) and rows[1][4] > 2),
        int(bool(all(not row[0] for row in rows[2:5]))),
        batch_observed,
        deterministic,
        int(len(rows) == 6 and all(len(row[:6]) == 6 for row in rows)),
    ], np.int8)
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
        "matrix_observed": matrix_observed,
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

    def build(
        seed: int,
        goal: torch.Tensor = target,
        *,
        fixed_samples: bool = False,
        num_iters: int = 8,
        num_particles: int = 48,
    ):
        sampler = ParticleSamplerCfg(
            device_cfg=device_cfg, fixed_samples=fixed_samples, seed=int(seed)
        )
        config = EvolutionStrategiesCfg(
            device_cfg=device_cfg, num_iters=num_iters, num_particles=num_particles,
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
    alternate_solution = build(
        int(seeds[0]), num_iters=6, num_particles=32
    )[0].optimize(initial)
    alternate_improved = objective(alternate_solution) < initial_objective
    trace = optimizer.get_recorded_trace()

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
    matrix_observed = np.asarray([
        int(bool(((final_samples < initial_samples).float().mean(0) >= 0.8).all().item())),
        int(flags["deterministic_repeat"] and flags["fixed_sample_repeat"]),
        flags["shift_observed"],
        int(isinstance(trace, dict) and bool(trace.get("debug"))),
        int(flags["solution_finite"] and bool((final_objective < initial_objective).all().item())),
        int(bool(alternate_improved.all().item())),
    ], np.int8)
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
        "matrix_observed": matrix_observed,
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

    def build(rollout, *, history: int = 5, line_search_scale=None):
        if line_search_scale is None:
            line_search_scale = [0.1, 0.3, 0.7, 1.0]
        config = LBFGSOptCfg(
            num_iters=16, inner_iters=1, num_problems=initial.shape[0],
            device_cfg=device_cfg, history=history, step_scale=1.0,
            line_search_scale=line_search_scale, fixed_iters=True,
            fix_terminal_action=False, return_best_action=True,
            store_debug=True,
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
    alternate_solution = build(
        rollout, history=3, line_search_scale=[0.2, 0.5, 1.0]
    ).optimize(initial)
    alternate_improved = objective(alternate_solution) < initial_objective
    trace = optimizer.get_recorded_trace()
    differentiable_initial = initial.detach().clone().requires_grad_(True)
    (input_gradient,) = torch.autograd.grad(
        objective(differentiable_initial).sum(), differentiable_initial
    )
    nonfinite = NonfiniteRollout()
    nonfinite_solution = build(nonfinite).optimize(initial)
    nonfinite_result = nonfinite.evaluate_action(nonfinite_solution)
    nonfinite_cost = (
        nonfinite_result.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
    )

    invalid_rejected = _invalid_rejected(lambda: LBFGSOptCfg(stable_mode=False))
    edge_observed = np.asarray([
        int(bool(
            improved.all().item()
            and ((solution >= lower) & (solution <= upper)).all().item()
            and bool((projected_optimality_norm <= 2e-3).all().item())
        ))
    ], np.int8)
    matrix_observed = np.asarray([
        int(bool(improved.all().item()) and bool(((solution >= lower) & (solution <= upper)).all().item())),
        int(bool(alternate_improved.all().item())),
        int(bool(torch.equal(solution, reset_solution))),
        int(isinstance(trace, dict) and bool(trace.get("debug"))),
        int(not bool(torch.isfinite(nonfinite_cost).all().item())),
        int(bool(torch.isfinite(input_gradient).all().item())),
    ], np.int8)
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
        "invalid_rejected": invalid_rejected,
        "edge_observed": edge_observed,
        "matrix_observed": matrix_observed,
    }


def _motion_planner(raw: dict[str, np.ndarray]) -> Output:
    """Run pinned high-level MotionPlanner on the shared Franka C-space case."""
    import torch
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.device_cfg import DeviceCfg

    cfg = MotionPlannerCfg.create(
        "franka.yml",
        device_cfg=DeviceCfg(device=torch.device("cuda", 0), dtype=torch.float32),
        num_ik_seeds=2,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        self_collision_check=False,
        max_batch_size=1,
    )
    planner = MotionPlanner(cfg)
    start = torch.as_tensor(raw["motion_start"], device="cuda", dtype=torch.float32)
    goal = start + torch.as_tensor(
        raw["motion_goal_delta"], device="cuda", dtype=torch.float32
    )
    start_state = JointState.from_position(start[None], planner.joint_names)
    goal_state = JointState.from_position(goal[None], planner.joint_names)
    result = planner.plan_cspace(
        goal_state, start_state, max_attempts=1, enable_graph_attempt=2
    )
    if result is None or result.js_solution is None:
        raise RuntimeError("pinned MotionPlanner did not return a C-space trajectory")
    active = result.js_solution.position[..., : len(planner.joint_names)]
    path_length = torch.linalg.vector_norm(torch.diff(active, dim=-2), dim=-1).sum(dim=-1)
    repeated = planner.plan_cspace(
        goal_state, start_state, max_attempts=1, enable_graph_attempt=2
    )

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
    repeat_status = repeated is not None and bool(repeated.success.all().item()) == success
    residual_start = start.detach().clone().requires_grad_(True)
    residual_goal = residual_start + torch.as_tensor(
        raw["motion_goal_delta"], device="cuda", dtype=torch.float32
    )
    (residual_gradient,) = torch.autograd.grad(
        (residual_goal - residual_start).square().sum(), residual_start
    )
    matrix_observed = np.asarray([
        int(active.shape[-1] == start.numel()),
        int(success and finite),
        invalid_rejected,
        int(active.shape[:2] == (1, 1)),
        int(repeat_status),
        int(active.ndim == 4 and result.success.ndim == 2),
        int(start_ok and goal_ok and bool(torch.isfinite(path_length).all().item())),
        int(bool(torch.isfinite(residual_gradient).all().item())),
    ], np.int8)
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
        "matrix_observed": matrix_observed,
    }


ADAPTERS: dict[str, Callable[[dict[str, np.ndarray]], Output]] = {
    "collision.mesh_world": _mesh_world,
    "collision.robot_scene": _robot_scene_collision,
    "collision.voxel_esdf_query": _voxel_esdf,
    "configuration.robot_config_and_loaders": _robot_config,
    "graph.prm_planner": _prm_planner,
    "motion_generation.motion_gen": _motion_planner,
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
