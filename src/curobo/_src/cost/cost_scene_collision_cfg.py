"""Configuration for the portable scene-collision cost.

The pinned V2 record is deliberately small, but it is used as the hand-off
between solver configuration, the collision checker, and the cost factory.
This implementation keeps that public record while compiling scalar options
onto the configured CPU/MPS device and validating the checker protocol before
an optimization loop starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type

import torch

from curobo._src.geom.collision.collision_scene import SceneCollision

from .cost_scene_collision import SceneCollisionCost
from .portable import BaseCostCfg, SceneCollisionCostCfg as _PortableSceneCollisionCostCfg


@dataclass
class SceneCollisionCostCfg(_PortableSceneCollisionCostCfg):
    """Compile a V2 scene-collision cost configuration for CPU or Metal.

    ``use_sweep_kernel`` remains a retained V2 option: it selects the
    checker's portable swept-query implementation, rather than exposing a raw
    Warp kernel.  Custom checkers are supported when they provide the same
    discrete/swept query protocol used by :class:`SceneCollisionCost`; this
    preserves lightweight user checkers without accepting arbitrary objects
    that would fail later inside an optimizer.
    """

    class_type: Type[SceneCollisionCost] = SceneCollisionCost

    def __post_init__(self) -> None:
        if isinstance(self.num_spheres, bool) or not isinstance(self.num_spheres, int):
            raise TypeError("num_spheres must be an integer")
        if self.num_spheres < 0:
            raise ValueError("num_spheres must be non-negative")
        if isinstance(self._num_scene_collision_checkers, bool) or not isinstance(
            self._num_scene_collision_checkers, int
        ):
            raise TypeError("_num_scene_collision_checkers must be an integer")
        if self._num_scene_collision_checkers < 0:
            raise ValueError("_num_scene_collision_checkers must be non-negative")
        for name in ("use_sweep", "use_sweep_kernel", "use_speed_metric", "sum_distance"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

        # The portable base converts floats, Python sequences, and tensors to
        # the configured dtype/device.  A scene cost has one scalar activation
        # radius; accepting a vector here would silently ignore all but its
        # first entry in the production query path.
        super().__post_init__()
        activation = self.activation_distance
        if activation.ndim == 0:
            activation = activation.reshape(1)
            self.activation_distance = activation
        if activation.ndim != 1 or activation.numel() != 1:
            raise ValueError("activation_distance must contain exactly one scalar")
        if not bool(torch.isfinite(activation).all().item()):
            raise ValueError("activation_distance must be finite")
        if bool((activation < 0).any().item()):
            raise ValueError("activation_distance must be non-negative")

        checker = self._scene_collision_checker
        if checker is not None:
            # Route through the setter to derive the checker count instead of
            # trusting an obsolete serialized private field.
            self._scene_collision_checker = None
            self.scene_collision_checker = checker

    @staticmethod
    def _query_protocol(checker: Any, swept: bool) -> bool:
        prefix = "get_swept_sphere" if swept else "get_sphere"
        return any(
            callable(getattr(checker, name, None))
            for name in (f"{prefix}_distance", f"{prefix}_collision")
        )

    def _validate_checker(self, checker: Any) -> None:
        if not isinstance(checker, SceneCollision) and not callable(checker):
            if not self._query_protocol(checker, swept=self.use_sweep):
                raise TypeError(
                    "scene_collision_checker must be a SceneCollision or provide the "
                    "configured discrete/swept sphere query"
                )
        if self.use_sweep and not self._query_protocol(checker, swept=True):
            raise TypeError(
                "use_sweep=True requires get_swept_sphere_distance or "
                "get_swept_sphere_collision on scene_collision_checker"
            )
        checker_device = getattr(getattr(checker, "device_cfg", None), "device", None)
        if checker_device is not None and not self.device_cfg.is_same_torch_device(checker_device):
            raise ValueError("scene_collision_checker device does not match device_cfg")

    @property
    def scene_collision_checker(self) -> Optional[Any]:
        """Configured native :class:`SceneCollision` or a query-compatible checker."""
        return self._scene_collision_checker

    @scene_collision_checker.setter
    def scene_collision_checker(self, scene_collision_checker: Any) -> None:
        if scene_collision_checker is None:
            self._scene_collision_checker = None
            self.update_num_scene_collision_checkers(0)
            return
        self._validate_checker(scene_collision_checker)
        self._scene_collision_checker = scene_collision_checker
        count = getattr(scene_collision_checker, "get_num_scene_collision_checkers", None)
        self.update_num_scene_collision_checkers(count() if callable(count) else 1)

    def update_num_spheres(self, num_spheres: int) -> None:
        if isinstance(num_spheres, bool) or not isinstance(num_spheres, int):
            raise TypeError("num_spheres must be an integer")
        if num_spheres < 0:
            raise ValueError("num_spheres must be non-negative")
        self.num_spheres = num_spheres

    def update_num_scene_collision_checkers(self, num_scene_collision_checkers: int) -> None:
        if isinstance(num_scene_collision_checkers, bool) or not isinstance(
            num_scene_collision_checkers, int
        ):
            raise TypeError("num_scene_collision_checkers must be an integer")
        if num_scene_collision_checkers < 0:
            raise ValueError("num_scene_collision_checkers must be non-negative")
        self._num_scene_collision_checkers = num_scene_collision_checkers


__all__ = ["BaseCostCfg", "SceneCollision", "SceneCollisionCost", "SceneCollisionCostCfg"]
