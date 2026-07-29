"""cuRoboV2-shaped solver and per-call configuration values."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from .cost import CostSet, UnsupportedCompatOption


class OptimizerType(str, Enum):
    LBFGS = "lbfgs"
    PARTICLE = "particle"
    ES = "es"


class InterpolationType(str, Enum):
    LINEAR = "linear"
    CUBIC = "cubic"
    BSPLINE = "bspline"


@dataclass(frozen=True)
class IKSolverConfig:
    num_seeds: int = 16
    max_iterations: int = 300
    position_tolerance: float = 5e-3
    rotation_tolerance: float = 5e-2
    random_seed: int = 0
    retract_config: tuple[float, ...] | None = None
    success_requires_convergence: bool = True
    optimizer: OptimizerType = OptimizerType.LBFGS


@dataclass(frozen=True)
class TrajOptSolverConfig:
    steps: int = 32
    dt: float = 0.05
    max_iterations: int = 100
    num_seeds: int = 1
    interpolation_dt: float = 0.025
    interpolation_type: InterpolationType = InterpolationType.LINEAR
    minimum_trajectory_dt: float | None = None
    maximum_trajectory_dt: float | None = None
    optimizer: OptimizerType = OptimizerType.LBFGS
    costs: CostSet = field(default_factory=CostSet)


@dataclass(frozen=True)
class GraphSolverConfig:
    sample_count: int = 128
    seed: int = 0
    k_neighbors: int = 12
    edge_step: float = 0.05
    cache_size: int = 1

    def __post_init__(self) -> None:
        if self.sample_count < 0 or self.seed < 0 or self.k_neighbors <= 0:
            raise ValueError("graph counts/seed are invalid")
        if self.edge_step <= 0 or self.cache_size < 0:
            raise ValueError("edge_step must be positive and cache_size nonnegative")


@dataclass(frozen=True)
class MotionGenPlanConfig:
    enable_graph: bool = True
    enable_opt: bool = True
    max_attempts: int = 1
    timeout: float | None = None
    enable_graph_attempt: int = 1
    partial_ik_opt: bool = False
    finetune_trajopt: bool = False
    parallel_finetune: bool = False
    time_dilation_factor: float = 1.0
    finetune_dt_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.enable_graph_attempt < 1:
            raise ValueError("enable_graph_attempt must be >= 1")
        if self.time_dilation_factor <= 0 or self.finetune_dt_scale <= 0:
            raise ValueError("retiming scales must be positive")
        unsupported = []
        if self.partial_ik_opt:
            unsupported.append("partial_ik_opt")
        if self.finetune_trajopt:
            unsupported.append("finetune_trajopt")
        if self.parallel_finetune:
            unsupported.append("parallel_finetune")
        if unsupported:
            raise UnsupportedCompatOption(f"unsupported MotionGenPlanConfig options: {unsupported}")

    def clone(self) -> "MotionGenPlanConfig":
        return type(self)(**asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MotionGenPlanConfig":
        return cls(**dict(value))
