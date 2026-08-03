"""Portable sequential optimizer composition compatible with cuRobo V2.

The CUDA implementation uses graph-captured rollout buffers to chain stages.
This version preserves the useful public contract with ordinary PyTorch tensors:
every enabled stage receives the preceding stage's best action, lifecycle
changes are broadcast to all stages, and timing is device-synchronised on MPS.
It deliberately does *not* claim CUDA graph or Warp-buffer compatibility.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from curobo._src.optim.optimization_iteration_state import OptimizationIterationState


class MultiStageOptimizer:
    """Chain optimizer stages, using each stage's output as the next seed.

    Stages may have different horizon/dimension factorizations, but they must
    describe the same per-problem action size.  This is the practical portion
    of V2's stage-local ``view`` contract and prevents accidental silent data
    reinterpretation in eager CPU/MPS execution.
    """

    def __init__(self, optimizers: List[Any], rollout_list: Optional[List[Any]] = None):
        if not optimizers:
            raise ValueError("optimizers must not be empty")
        self.optimizers = list(optimizers)
        if rollout_list is None:
            rollout_list = getattr(self.optimizers[-1], "_rollout_list", None)
        if not rollout_list:
            raise ValueError("rollout_list must contain at least one rollout")
        self._rollout_list = list(rollout_list)
        self.rollout_fn = self._rollout_list[0]
        self.config = self.optimizers[-1].config
        self.device_cfg = self.config.device_cfg
        self._validate_stage_layouts()
        self._enabled = True
        self.opt_dt = 0.0
        self._last_iteration_state: Optional[OptimizationIterationState] = None
        self._last_stage_outputs: tuple[torch.Tensor, ...] = ()

    def _validate_stage_layouts(self) -> None:
        final_size = self.action_horizon * self.action_dim
        final_problems = int(self.config.num_problems)
        if final_problems <= 0:
            raise ValueError("final optimizer config.num_problems must be positive")
        for index, optimizer in enumerate(self.optimizers):
            try:
                stage_size = int(optimizer.action_horizon) * int(optimizer.action_dim)
                stage_problems = int(optimizer.config.num_problems)
            except AttributeError as error:
                raise TypeError(
                    f"optimizer stage {index} does not expose config/action_horizon/action_dim"
                ) from error
            if stage_size != final_size:
                raise ValueError(
                    "all multi-stage optimizers must have the same action size per problem; "
                    f"stage {index} has {stage_size}, final stage has {final_size}"
                )
            if stage_problems != final_problems:
                raise ValueError(
                    "all multi-stage optimizers must have the same num_problems; "
                    f"stage {index} has {stage_problems}, final stage has {final_problems}"
                )

    # -- Properties ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def action_horizon(self) -> int:
        return int(self.optimizers[-1].action_horizon)

    @property
    def action_dim(self) -> int:
        return int(self.optimizers[-1].action_dim)

    @property
    def opt_dim(self) -> int:
        return self.action_horizon * self.action_dim

    @property
    def outer_iters(self) -> int:
        return 1

    @property
    def solver_names(self) -> list[str]:
        return [str(stage.config.solver_name) for stage in self.optimizers]

    @property
    def solve_time(self) -> float:
        return self.opt_dt

    @property
    def last_stage_outputs(self) -> tuple[torch.Tensor, ...]:
        """Immutable view of outputs from the previous successful optimize."""

        return self._last_stage_outputs

    # -- Core ---------------------------------------------------------------

    def _canonical_seed(self, seed_action: torch.Tensor) -> torch.Tensor:
        if not isinstance(seed_action, torch.Tensor):
            raise TypeError("seed_action must be a torch.Tensor")
        problems = int(self.config.num_problems)
        if problems <= 0:
            raise ValueError("config.num_problems must be positive")
        shape = (problems, self.action_horizon, self.action_dim)
        if tuple(seed_action.shape) == shape:
            return seed_action
        if tuple(seed_action.shape) == (problems, self.opt_dim):
            return seed_action.reshape(shape)
        if problems == 1 and tuple(seed_action.shape) == shape[1:]:
            return seed_action.unsqueeze(0)
        # Lightweight V2 callers commonly leave ``num_problems`` at its
        # default of one and let the first batched seed establish the runtime
        # batch.  Preserve that convenience, but only for an otherwise
        # unambiguous [B,H,D] or [B,H*D] seed.
        if problems == 1 and seed_action.ndim == 3 and tuple(seed_action.shape[1:]) == shape[1:]:
            self.update_num_problems(int(seed_action.shape[0]))
            return seed_action
        if problems == 1 and seed_action.ndim == 2 and seed_action.shape[1] == self.opt_dim:
            self.update_num_problems(int(seed_action.shape[0]))
            return seed_action.reshape(int(seed_action.shape[0]), self.action_horizon, self.action_dim)
        raise ValueError(
            "seed_action must have shape "
            f"[{problems}, {self.action_horizon}, {self.action_dim}] or "
            f"[{problems}, {self.opt_dim}], got {tuple(seed_action.shape)}"
        )

    @staticmethod
    def _stage_seed(action: torch.Tensor, optimizer: Any) -> torch.Tensor:
        problems = int(optimizer.config.num_problems)
        horizon, action_dim = int(optimizer.action_horizon), int(optimizer.action_dim)
        if action.numel() != problems * horizon * action_dim:
            raise ValueError(
                "stage action layout is incompatible with the previous stage: "
                f"cannot reshape {tuple(action.shape)} to [{problems}, {horizon}, {action_dim}]"
            )
        return action.reshape(problems, horizon, action_dim)

    @staticmethod
    def _validate_stage_output(
        output: Any, seed: torch.Tensor, optimizer: Any
    ) -> torch.Tensor:
        """Reject accidental host/device or dtype transitions between stages.

        CUDA graphs keep each stage's action buffer on a single device.  The
        eager backend has no graph buffer to provide that safety, so make the
        invariant explicit instead of silently accepting an MPS-to-CPU copy
        (or a precision change) between otherwise composable stages.
        """

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"optimizer {type(optimizer).__name__}.optimize must return a torch.Tensor"
            )
        if output.device != seed.device:
            raise ValueError(
                f"optimizer {type(optimizer).__name__} changed action device "
                f"from {seed.device} to {output.device}"
            )
        if output.dtype != seed.dtype:
            raise ValueError(
                f"optimizer {type(optimizer).__name__} changed action dtype "
                f"from {seed.dtype} to {output.dtype}"
            )
        if output.numel() != seed.numel():
            raise ValueError(
                f"optimizer {type(optimizer).__name__} returned {output.numel()} action values, "
                f"expected {seed.numel()}"
            )
        return output

    @staticmethod
    def _synchronize(tensor: torch.Tensor) -> None:
        if tensor.device.type == "mps":
            torch.mps.synchronize()

    def _run_stages(
        self, iteration_state: OptimizationIterationState
    ) -> OptimizationIterationState:
        """Execute stage composition without recursively entering ``optimize``."""

        current = iteration_state.best_action
        if current is None:
            current = iteration_state.action
        current = self._canonical_seed(current)
        outputs: list[torch.Tensor] = []
        if self.enabled:
            for optimizer in self.optimizers:
                if not bool(getattr(optimizer, "enabled", True)):
                    continue
                stage_seed = self._stage_seed(current, optimizer)
                optimized = self._validate_stage_output(
                    optimizer.optimize(stage_seed), stage_seed, optimizer
                )
                current = self._stage_seed(optimized, optimizer)
                outputs.append(current.detach().clone())

        # Convert the last stage's layout back to the public final-stage view.
        result = current.reshape(
            int(self.config.num_problems), self.action_horizon, self.action_dim
        )
        result_state = OptimizationIterationState(
            action=result,
            exploration_action=result,
            best_action=result,
        )
        self._last_stage_outputs = tuple(outputs)
        self._last_iteration_state = result_state
        return result_state

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        """Run enabled stages in order and return a final ``[B,H,D]`` tensor."""

        current = self._canonical_seed(seed_action)
        start = time.perf_counter()
        result_state = self._run_stages(
            OptimizationIterationState(action=current, exploration_action=current)
        )
        result = result_state.best_action
        assert result is not None  # Internal invariant established by _run_stages.
        self._synchronize(result)
        self.opt_dt = time.perf_counter() - start
        return result

    def _opt_iters(self, iteration_state: OptimizationIterationState) -> OptimizationIterationState:
        """Compatibility helper for code that drives stage state directly."""

        if not isinstance(iteration_state, OptimizationIterationState):
            raise TypeError("iteration_state must be an OptimizationIterationState")
        return self._run_stages(iteration_state)

    # -- Lifecycle ----------------------------------------------------------

    @staticmethod
    def _call_lifecycle(optimizer: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        callback = getattr(optimizer, name, None)
        return callback(*args, **kwargs) if callable(callback) else None

    def reinitialize(
        self,
        action: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        clear_optimizer_state: bool = True,
        reset_num_iters: bool = False,
    ) -> None:
        canonical_action = self._canonical_seed(action)
        for optimizer in self.optimizers:
            if bool(getattr(optimizer, "enabled", True)):
                stage_action = self._stage_seed(canonical_action, optimizer)
                self._call_lifecycle(
                    optimizer,
                    "reinitialize",
                    stage_action,
                    mask=mask,
                    clear_optimizer_state=clear_optimizer_state,
                    reset_num_iters=reset_num_iters,
                )
        self._last_iteration_state = None
        self._last_stage_outputs = ()

    def shift(self, shift_steps: int = 0) -> bool:
        if shift_steps < 0:
            raise ValueError("shift_steps must be nonnegative")
        result = True
        for optimizer in self.optimizers:
            callback = getattr(optimizer, "_shift", None) or getattr(optimizer, "shift", None)
            if callable(callback):
                result = bool(callback(shift_steps)) and result
        return result

    _shift = shift

    def update_num_problems(self, num_problems: int) -> None:
        if num_problems <= 0:
            raise ValueError("num_problems must be positive")
        for optimizer in self.optimizers:
            self._call_lifecycle(optimizer, "update_num_problems", int(num_problems))
        # Optimizers generally share config references only within a stage;
        # assert the wrapper's exposed final config follows the update.
        self.config.num_problems = int(num_problems)
        self._last_iteration_state = None
        self._last_stage_outputs = ()

    def update_rollout_params(self, goal: Any) -> None:
        for optimizer in self.optimizers:
            if bool(getattr(optimizer, "enabled", True)):
                self._call_lifecycle(optimizer, "update_rollout_params", goal)

    def update_goal_dt(self, goal_dt: Any) -> None:
        for optimizer in self.optimizers:
            self._call_lifecycle(optimizer, "update_goal_dt", goal_dt)

    def reset(self) -> None:
        for optimizer in self.optimizers:
            self._call_lifecycle(optimizer, "reset")
        self._last_iteration_state = None
        self._last_stage_outputs = ()

    def get_all_rollout_instances(self) -> list[Any]:
        instances: list[Any] = []
        for optimizer in self.optimizers:
            found = self._call_lifecycle(optimizer, "get_all_rollout_instances")
            if found is not None:
                instances.extend(found)
        return instances

    def compute_metrics(self, action: torch.Tensor) -> Any:
        # V2 expressly rejects this ambiguous request: stages can have
        # different rollout configurations.  Keep the error rather than
        # silently choosing a stage and reporting misleading metrics.
        del action
        raise RuntimeError(
            "compute_metrics is ambiguous for MultiStageOptimizer; call it on a specific stage"
        )

    def reset_shape(self) -> None:
        for optimizer in self.optimizers:
            self._call_lifecycle(optimizer, "reset_shape")
        self._call_lifecycle(self.rollout_fn, "reset_shape")
        self._last_iteration_state = None
        self._last_stage_outputs = ()

    def reset_seed(self) -> None:
        for optimizer in self.optimizers:
            self._call_lifecycle(optimizer, "reset_seed")
        self._call_lifecycle(self.rollout_fn, "reset_seed")

    def reset_cuda_graph(self) -> None:
        """Reset portable execution state; no CUDA graph is captured on MPS."""

        for optimizer in self.optimizers:
            try:
                self._call_lifecycle(optimizer, "reset_cuda_graph")
            except NotImplementedError:
                # Some portable stages deliberately reject direct CUDA graph
                # control.  The composite has no graph either, so reset their
                # ordinary persistent state instead.
                self._call_lifecycle(optimizer, "reset")

    def get_recorded_trace(self) -> Dict[str, list[Any]]:
        trace: Dict[str, list[Any]] = {"debug": [], "debug_cost": []}
        for optimizer in self.optimizers:
            stage_trace = self._call_lifecycle(optimizer, "get_recorded_trace")
            if not isinstance(stage_trace, dict):
                continue
            for key in trace:
                value = stage_trace.get(key, [])
                if value is None:
                    continue
                trace[key].extend(value if isinstance(value, (tuple, list)) else [value])
        return trace

    def update_niters(self, niters: int) -> None:
        for optimizer in self.optimizers:
            self._call_lifecycle(optimizer, "update_niters", niters)

    def update_solver_params(self, solver_params: Dict[str, Dict[str, Any]]) -> bool:
        if not isinstance(solver_params, dict) or not solver_params:
            raise ValueError("solver_params must be a non-empty dictionary of dictionaries")
        for name, values in solver_params.items():
            if not isinstance(values, dict):
                raise ValueError("solver_params must be a dictionary of dictionaries")
            if name not in self.solver_names:
                raise ValueError(f"Optimizer {name} not found in {self.solver_names}")
            optimizer = self.optimizers[self.solver_names.index(name)]
            result = self._call_lifecycle(optimizer, "update_solver_params", {name: values})
            if result is False:
                return False
        return True

    def debug_dump(self, file_path: str = "") -> list[Any]:
        """Delegate portable debug dumps to every stage and return their values."""

        return [self._call_lifecycle(optimizer, "debug_dump", file_path) for optimizer in self.optimizers]


__all__ = ["MultiStageOptimizer"]
