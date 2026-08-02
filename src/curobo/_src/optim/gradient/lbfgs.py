"""Pinned L-BFGS surface backed by portable batched L-BFGS."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from curobo._src.optim._portable import PortableOptCfg, PortableOptimizer
from curobo._src.types.device_cfg import DeviceCfg
from .line_search_strategy import LineSearchType


@dataclass
class LBFGSOptCfg(PortableOptCfg):
    solver_type: str = "lbfgs"
    solver_name: str = "lbfgs"
    inner_iters: int = 25
    _num_rollout_instances: int = 2
    cost_convergence: float = 1.0e-11
    cost_delta_threshold: float = 0.0
    cost_relative_threshold: float = 0.0
    converged_ratio: float = 0.8
    fixed_iters: bool = True
    convergence_iteration: int = 0
    minimum_iters: Optional[int] = None
    return_best_action: bool = True
    line_search_scale: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.7, 1.0])
    line_search_type: Any = "APPROX_WOLFE"
    use_cuda_kernel_line_search: bool = False
    fix_terminal_action: bool = False
    line_search_wolfe_c_1: float = 1e-5
    line_search_wolfe_c_2: float = 0.9
    history: int = 7
    epsilon: float = 0.01
    use_cuda_kernel_step_direction: bool = False
    stable_mode: bool = True
    use_cuda_kernel_shared_buffers: bool = False
    initial_step_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.inner_iters <= 0:
            raise ValueError("inner_iters must be positive")
        if self.num_particles is None:
            self.num_particles = len(self.line_search_scale)
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if not self.line_search_scale:
            raise ValueError("line_search_scale must not be empty")
        if any(scale < 0 for scale in self.line_search_scale):
            raise ValueError("line_search_scale must be nonnegative")
        self.line_search_type = LineSearchType(self.line_search_type)
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0
            self.cost_relative_threshold = 0.0
        if self.cost_relative_threshold >= 1.0:
            raise ValueError("cost_relative_threshold must be less than 1.0")
        if not self.stable_mode:
            raise ValueError("LBFGS stable_mode must be true")
        if self._num_rollout_instances != 2:
            raise ValueError("LBFGS _num_rollout_instances must be 2")
        # These toggles select CUDA-only implementation details upstream.  A
        # regular PyTorch line search/two-loop solve remains available here.
        self.use_cuda_kernel_line_search = False
        self.use_cuda_kernel_step_direction = False
        self.use_cuda_kernel_shared_buffers = False

    @property
    def portable_line_search(self):
        name = LineSearchType(self.line_search_type).value
        # The production portable L-BFGS core exposes Armijo and strong
        # Wolfe.  V2's weak-Wolfe candidate strategy is available from this
        # package directly; L-BFGS maps it to the stronger safe condition.
        if "wolfe" in name:
            return "strong_wolfe"
        return "armijo"

    def update_niters(self, niters: int):
        if niters <= 0:
            raise ValueError("niters must be positive")
        if niters < self.inner_iters or niters % self.inner_iters:
            raise ValueError("num_iters must be a positive multiple of inner_iters")
        self.num_iters = niters


class LBFGSOpt(PortableOptimizer):
    strategy = "lbfgs"


__all__ = ["LBFGSOptCfg", "LBFGSOpt"]
