"""Direct behavioral coverage for the portable world-collision cost facade."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg


def _checker(device: torch.device) -> SceneCollision:
    cfg = DeviceCfg(device)
    scene = SceneCfg(cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], [0.4, 0.4, 0.4])])
    return SceneCollision(SceneCollisionCfg(device_cfg=cfg, scene_model=scene, max_distance=1.0))


def test_native_scene_cost_aggregates_once_and_resets_workspace() -> None:
    device = torch.device("cpu")
    config = SceneCollisionCostCfg(
        weight=2.0, activation_distance=0.1, num_spheres=2,
        _scene_collision_checker=_checker(device),
    )
    cost = SceneCollisionCost(config)
    cost.setup_batch_tensors(2, 2)
    spheres = torch.tensor(
        [
            [[[0.0, 0.0, 0.0, 0.05], [0.8, 0.0, 0.0, 0.05]],
             [[0.1, 0.0, 0.0, 0.05], [0.7, 0.0, 0.0, 0.05]]],
            [[[0.5, 0.0, 0.0, 0.05], [0.6, 0.0, 0.0, 0.05]],
             [[0.0, 0.1, 0.0, 0.05], [0.0, 0.7, 0.0, 0.05]]],
        ],
        requires_grad=True,
    )
    state = SimpleNamespace(robot_spheres=spheres)
    output = cost(state)

    expected_buffer = CollisionBuffer.from_shape(spheres.shape, DeviceCfg(device))
    raw = config.scene_collision_checker.get_sphere_distance(
        state, expected_buffer, torch.ones(1), config.activation_distance
    )
    expected = 2.0 * 0.5 * (config.activation_distance[0] - raw).clamp_min(0).square().sum(-1)
    torch.testing.assert_close(output, expected)
    assert output.shape == (2, 2)
    output.sum().backward()
    assert spheres.grad is not None and torch.isfinite(spheres.grad).all()
    assert cost.get_gradient_buffer().shape == spheres.shape

    retained = cost.get_gradient_buffer()[0].clone()
    cost.reset(torch.tensor([1], dtype=torch.int64))
    torch.testing.assert_close(cost.get_gradient_buffer()[0], retained)
    assert not bool(cost.get_gradient_buffer()[1].any())


def test_scene_cost_topology_sweep_and_binary_lifecycle() -> None:
    device = torch.device("cpu")
    checker = _checker(device)
    config = SceneCollisionCostCfg(
        weight=1.0, activation_distance=0.05, num_spheres=1,
        use_sweep=True, use_speed_metric=True, convert_to_binary=True,
        _scene_collision_checker=checker,
    )
    cost = SceneCollisionCost(config)
    cost.setup_batch_tensors(1, 3)
    cost.update_num_spheres(2)
    assert cost.get_gradient_buffer().shape == (1, 3, 2, 4)
    spheres = torch.tensor(
        [[[[0.4, 0.0, 0.0, 0.05], [0.7, 0.0, 0.0, 0.05]],
          [[0.1, 0.0, 0.0, 0.05], [0.7, 0.0, 0.0, 0.05]],
          [[0.0, 0.0, 0.0, 0.05], [0.7, 0.0, 0.0, 0.05]]]],
        requires_grad=True,
    )
    result = cost(SimpleNamespace(robot_spheres=spheres), trajectory_dt=torch.tensor([0.1]))
    assert result.shape == (1, 3) and result.device == spheres.device
    assert result[0, 1] >= 1.0  # binary mode adds one for a collision penalty.
    result.sum().backward()
    assert torch.isfinite(spheres.grad).all()
    with pytest.raises(ValueError, match="horizon"):
        cost(SimpleNamespace(robot_spheres=spheres[:, :2]), trajectory_dt=torch.tensor([0.1]))
    with pytest.raises(ValueError, match="trajectory_dt"):
        cost(SimpleNamespace(robot_spheres=spheres), trajectory_dt=torch.tensor([0.1, 0.1]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_scene_collision_cost_mps_is_differentiable_without_cpu_fallback() -> None:
    device = torch.device("mps")
    cfg = DeviceCfg(device)
    cost = SceneCollisionCost(SceneCollisionCostCfg(
        weight=1.5, activation_distance=0.1, num_spheres=1, device_cfg=cfg,
        _scene_collision_checker=_checker(device),
    ))
    cost.setup_batch_tensors(1, 2)
    spheres = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.05]], [[0.5, 0.0, 0.0, 0.05]]]],
        device=device, requires_grad=True,
    )
    output = cost(SimpleNamespace(robot_spheres=spheres))
    output.sum().backward()
    assert output.device.type == spheres.grad.device.type == "mps"
    assert torch.isfinite(spheres.grad).all()
