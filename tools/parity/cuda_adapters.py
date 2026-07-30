"""Asset-independent adapters for executing pinned cuRobo probes on CUDA."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Output = dict[str, np.ndarray]


def _invalid_rejected(operation) -> np.ndarray:
    import torch

    try:
        value = operation()
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
            return np.array([1], np.int8)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return np.array([1], np.int8)
    return np.array([0], np.int8)


def _status_codes(values) -> np.ndarray:
    mapping = {"success": 0, "ik_failed": 1}
    try:
        return np.asarray([mapping[value] for value in values], np.int8)
    except KeyError as error:
        raise ValueError(f"unsupported normalized solver status: {error.args[0]}") from error


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


ADAPTERS: dict[str, Callable[[dict[str, np.ndarray]], Output]] = {
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
