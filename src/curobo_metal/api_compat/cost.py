"""Serializable cuRobo-style cost configuration adapters.

These classes deliberately contain configuration only.  ``to_production`` is
the explicit boundary to the smaller set of costs implemented by curobo-metal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from curobo_metal.ops.trajectory import TrajectoryWeights


class UnsupportedCompatOption(ValueError):
    """A valid upstream concept which has no portable implementation."""


class ProjectedDistType(str, Enum):
    DISABLED = "disabled"
    EUCLIDEAN = "euclidean"
    SQUARED = "squared"


@dataclass(frozen=True)
class CostConfig:
    weight: float | Sequence[float] = 1.0
    terminal: bool = False
    run_weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CostConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class PoseCostConfig(CostConfig):
    position_weight: float = 1.0
    rotation_weight: float = 1.0
    position_tolerance: float = 5e-3
    rotation_tolerance: float = 5e-2
    project_distance: ProjectedDistType = ProjectedDistType.DISABLED


@dataclass(frozen=True)
class BoundCostConfig(CostConfig):
    margin: float = 0.0
    null_space_weight: float = 0.0


@dataclass(frozen=True)
class SmoothnessCostConfig(CostConfig):
    velocity: float = 0.05
    acceleration: float = 1.0
    jerk: float = 0.05

    def to_production(self, *, endpoint: float = 1000.0, joint_limit: float = 10.0
                      ) -> TrajectoryWeights:
        return TrajectoryWeights(endpoint, joint_limit, self.velocity, self.acceleration, self.jerk)


@dataclass(frozen=True)
class CollisionCostConfig(CostConfig):
    activation_distance: float = 0.0
    padding: float = 0.0
    use_sweep: bool = False
    sum_distance: bool = True

    def validate_production(self) -> None:
        if self.use_sweep:
            raise UnsupportedCompatOption("collision.use_sweep requires an upstream sweep kernel")
        if not self.sum_distance:
            raise UnsupportedCompatOption("collision.sum_distance=False is not implemented")


PrimitiveCollisionCostConfig = CollisionCostConfig
SelfCollisionCostConfig = CollisionCostConfig


@dataclass(frozen=True)
class RunWeight:
    weight: float = 1.0
    horizon: int | None = None


@dataclass(frozen=True)
class OffsetWaypoint:
    timestep: int
    offset: tuple[float, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.timestep < 0:
            raise ValueError("offset waypoint timestep must be nonnegative")


@dataclass(frozen=True)
class CostSet:
    pose: PoseCostConfig = field(default_factory=PoseCostConfig)
    bounds: BoundCostConfig = field(default_factory=BoundCostConfig)
    smoothness: SmoothnessCostConfig = field(default_factory=SmoothnessCostConfig)
    collision: CollisionCostConfig = field(default_factory=CollisionCostConfig)
    run_weight: RunWeight | None = None
    offset_waypoints: tuple[OffsetWaypoint, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_production(self) -> None:
        self.collision.validate_production()
        if self.run_weight is not None:
            raise UnsupportedCompatOption("time-varying run weights are not implemented")
        if self.offset_waypoints:
            raise UnsupportedCompatOption("offset waypoint costs are not implemented")
