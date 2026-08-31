"""Portable configuration record for :class:`~.rollout_robot.RobotRollout`.

The pinned V2 record is deliberately small, but it owns an important boundary:
YAML-shaped dictionaries become typed transition and cost-manager configs before
the rollout creates device-resident components.  This implementation keeps that
boundary on CPU/MPS without importing CUDA, Warp, or a scene backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.logging import log_and_raise


_MANAGER_FIELDS = (
    "cost_cfg",
    "constraint_cfg",
    "hybrid_cost_constraint_cfg",
    "convergence_cfg",
)


@dataclass
class RobotRolloutCfg:
    """Configuration required to build a portable robot rollout.

    ``object`` remains a supported component-type sentinel for callers that
    deliberately pass already-compiled portable records (notably generic
    solver assembly).  With a concrete type, V2's exact component-type
    contract is retained so configuration errors are reported before rollout
    execution rather than in an unrelated cost or transition call.
    """

    device_cfg: DeviceCfg
    sum_horizon: bool = False
    sampler_seed: int = 1312
    cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg
    transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg
    transition_model_cfg: Optional[Any] = None
    cost_cfg: Optional[RobotCostManagerCfg] = None
    constraint_cfg: Optional[RobotCostManagerCfg] = None
    hybrid_cost_constraint_cfg: Optional[RobotCostManagerCfg] = None
    convergence_cfg: Optional[RobotCostManagerCfg] = None
    scene_collision_cfg: Optional[SceneCollisionCfg] = None

    def __post_init__(self) -> None:
        if not isinstance(self.device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if not isinstance(self.sum_horizon, bool):
            raise TypeError("sum_horizon must be a bool")
        if isinstance(self.sampler_seed, bool) or not isinstance(self.sampler_seed, int):
            raise TypeError("sampler_seed must be an integer")
        if self.sampler_seed < 0:
            raise ValueError("sampler_seed must be nonnegative")
        self._validate_component_type(
            "transition_model_config_instance_type", self.transition_model_config_instance_type,
            self.transition_model_cfg,
        )
        for field_name in _MANAGER_FIELDS:
            self._validate_component_type(
                "cost_manager_config_instance_type", self.cost_manager_config_instance_type,
                getattr(self, field_name), field_name,
            )

    @staticmethod
    def _validate_component_type(
        declared_name: str,
        declared_type: type,
        value: Any,
        field_name: str = "transition_model_cfg",
    ) -> None:
        if not isinstance(declared_type, type):
            raise TypeError(f"{declared_name} must be a type")
        # ``object`` is an explicit portable escape hatch, not a claim that
        # arbitrary values have V2 CUDA semantics.  It lets generic solver
        # configuration retain a precompiled record until its owning layer
        # supplies a concrete component implementation.
        if value is None or declared_type is object:
            return
        # The portable rollout also accepts an already-compiled transition
        # record exposing its runtime ``class_type``.  This is how lightweight
        # CPU/MPS transition adapters participate without pretending to be the
        # CUDA-only ``RobotStateTransitionCfg`` dataclass.  A plain arbitrary
        # object is still rejected below.
        if field_name == "transition_model_cfg" and isinstance(getattr(value, "class_type", None), type):
            return
        if type(value) is not declared_type:
            raise TypeError(
                f"{declared_name} {declared_type} must match type of "
                f"{field_name} {type(value)}"
            )

    @staticmethod
    def _create_component(
        value: Any,
        component_type: type,
        robot_cfg: Any,
        device_cfg: DeviceCfg,
        *,
        transition: bool,
    ) -> Any:
        if value is None or not isinstance(value, Mapping):
            return value
        # Generic assembly uses ``object`` as a documented sentinel.  It
        # intentionally preserves this dictionary; calling ``object.create``
        # would turn a valid portable assembly path into an attribute error.
        if component_type is object:
            return deepcopy(dict(value))
        create = getattr(component_type, "create", None)
        if not callable(create):
            label = "transition_model_config_instance_type" if transition else "cost_manager_config_instance_type"
            raise TypeError(f"{label} must provide a callable create method")
        payload = deepcopy(dict(value))
        if transition:
            return create(payload, robot_cfg, device_cfg)
        return create(payload, device_cfg=device_cfg)

    @classmethod
    def create_with_component_types(
        cls,
        data_dict: Dict,
        robot_cfg: Union[Dict, RobotCfg],
        device_cfg: DeviceCfg = DeviceCfg(),
        transition_model_config_instance_type: Type[RobotStateTransitionCfg] = RobotStateTransitionCfg,
        cost_manager_config_instance_type: Type[RobotCostManagerCfg] = RobotCostManagerCfg,
    ):
        """Compile a YAML-style rollout mapping without mutating the caller's data.

        The created config retains every V2 field and uses the caller-selected
        component classes.  Scene-collision construction remains owned by the
        solver/collision layer; this record only transports its typed config.
        """
        if not isinstance(data_dict, Mapping):
            raise TypeError("data_dict must be a mapping")
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        data = deepcopy(dict(data_dict))
        data["transition_model_cfg"] = cls._create_component(
            data.get("transition_model_cfg"), transition_model_config_instance_type,
            robot_cfg, device_cfg, transition=True,
        )
        for field_name in _MANAGER_FIELDS:
            data[field_name] = cls._create_component(
                data.get(field_name), cost_manager_config_instance_type,
                robot_cfg, device_cfg, transition=False,
            )
        return cls(
            **data,
            device_cfg=device_cfg,
            transition_model_config_instance_type=transition_model_config_instance_type,
            cost_manager_config_instance_type=cost_manager_config_instance_type,
        )

    def get_cost_manager_configs(self, include_constraint_cfg: bool = True) -> List[RobotCostManagerCfg]:
        """Return configured managers in the pinned cost/constraint order."""
        values = [self.cost_cfg]
        if include_constraint_cfg:
            values.append(self.constraint_cfg)
        values.extend((self.hybrid_cost_constraint_cfg, self.convergence_cfg))
        return [value for value in values if value is not None]


__all__ = [
    "Dict", "List", "RobotCfg", "RobotCostManagerCfg", "RobotRolloutCfg",
    "RobotStateTransitionCfg", "SceneCollisionCfg", "Type", "Union", "DeviceCfg",
    "log_and_raise",
]
