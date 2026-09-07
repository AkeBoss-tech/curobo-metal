"""Behavioral coverage for the portable V2 scene-collision cost config."""

from __future__ import annotations

import pytest
import torch

from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg


class _DiscreteChecker:
    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device_cfg = DeviceCfg(device)

    def get_sphere_distance(self, spheres, env_query_idx=None):
        del env_query_idx
        return torch.ones(spheres.shape[:-1], device=spheres.device, dtype=spheres.dtype)


class _SweptChecker:
    def get_swept_sphere_distance(self, spheres, trajectory_dt=None, env_query_idx=None):
        del trajectory_dt, env_query_idx
        return torch.ones(spheres.shape[:-1], device=spheres.device, dtype=spheres.dtype)


def test_config_compiles_scalar_and_native_checker_count() -> None:
    device_cfg = DeviceCfg("cpu")
    scene = SceneCfg(cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], [1, 1, 1], device_cfg=DeviceCfg("cpu"))])
    checker = SceneCollision(SceneCollisionCfg(device_cfg=device_cfg, scene_model=scene))
    cfg = SceneCollisionCostCfg(
        weight=1.0,
        activation_distance=0.05,
        _scene_collision_checker=checker,
        device_cfg=device_cfg,
    )

    assert cfg.class_type is SceneCollisionCost
    assert cfg.activation_distance.shape == (1,)
    assert cfg.activation_distance.device == device_cfg.device
    assert cfg.scene_collision_checker is checker
    assert cfg._num_scene_collision_checkers == checker.get_num_scene_collision_checkers()

    tensor_scalar = SceneCollisionCostCfg(weight=1.0, activation_distance=torch.tensor(0.02), device_cfg=DeviceCfg("cpu"))
    assert tensor_scalar.activation_distance.shape == (1,)


def test_config_accepts_query_protocol_and_checks_swept_capability() -> None:
    discrete = _DiscreteChecker()
    cfg = SceneCollisionCostCfg(weight=1.0, _scene_collision_checker=discrete, device_cfg=DeviceCfg("cpu"))
    assert cfg.scene_collision_checker is discrete
    assert cfg._num_scene_collision_checkers == 1

    swept = SceneCollisionCostCfg(
        weight=1.0, use_sweep=True, _scene_collision_checker=_SweptChecker()
    , device_cfg=DeviceCfg("cpu"))
    assert swept._num_scene_collision_checkers == 1
    with pytest.raises(TypeError, match="configured discrete/swept"):
        SceneCollisionCostCfg(weight=1.0, use_sweep=True, _scene_collision_checker=discrete, device_cfg=DeviceCfg("cpu"))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"activation_distance": [-0.01]}, "non-negative"),
        ({"activation_distance": [0.01, 0.02]}, "exactly one"),
        ({"activation_distance": float("nan")}, "finite"),
        ({"num_spheres": -1}, "non-negative"),
        ({"num_spheres": True}, "integer"),
        ({"use_sweep": 1}, "bool"),
    ],
)
def test_config_rejects_ambiguous_or_invalid_inputs(kwargs, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        SceneCollisionCostCfg(weight=1.0, **kwargs)


def test_config_rejects_checker_on_other_device_when_mps_exists() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("requires Apple Metal")
    with pytest.raises(ValueError, match="device does not match"):
        SceneCollisionCostCfg(
            weight=1.0,
            device_cfg=DeviceCfg(torch.device("mps")),
            _scene_collision_checker=_DiscreteChecker(torch.device("cpu")),
        )
