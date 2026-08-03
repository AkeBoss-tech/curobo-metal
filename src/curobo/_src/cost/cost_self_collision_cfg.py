"""Configuration for the portable V2 self-collision trajectory cost.

The upstream record is intentionally small, but it is an important factory
boundary: a rollout uses ``class_type`` to instantiate the concrete cost.
Keep that pointer aimed at the production portable implementation rather than
at the legacy generic cost shim in :mod:`.portable`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type

from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg

from .cost_self_collision import SelfCollisionCost
from .portable import BaseCostCfg


@dataclass
class SelfCollisionCostCfg(BaseCostCfg):
    """Compile self-collision parameters for the configured CPU/MPS device.

    ``self_collision_kin_config`` accepts the pinned
    :class:`SelfCollisionKinematicsCfg` as well as compatible robot-builder
    records.  The latter keeps existing serialized robot configurations
    usable while the cost validates their actual tensor topology before it
    runs.  CUDA/Warp launch buffers are deliberately not part of this record.
    """

    class_type: Type[SelfCollisionCost] = SelfCollisionCost
    self_collision_kin_config: Optional[Any] = None
    store_pair_distance: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.store_pair_distance, bool):
            raise TypeError("store_pair_distance must be a bool")
        config = self.self_collision_kin_config
        if config is not None:
            count = getattr(config, "num_spheres", None)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    "self_collision_kin_config.num_spheres must be a non-negative integer"
                )


__all__ = ["BaseCostCfg", "SelfCollisionCost", "SelfCollisionCostCfg", "SelfCollisionKinematicsCfg"]
