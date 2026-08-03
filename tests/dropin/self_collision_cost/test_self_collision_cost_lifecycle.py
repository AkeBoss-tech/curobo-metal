"""Lifecycle coverage for the production portable self-collision facade."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class _KinematicsConfig:
    num_spheres: int = 3
    sphere_padding: torch.Tensor | None = None
    collision_pairs: torch.Tensor | None = None
    num_blocks_per_batch: int = 2


def _config(device: torch.device = torch.device("cpu"), *, pairs=None) -> SelfCollisionCostCfg:
    if pairs is None:
        pairs = torch.tensor([[0, 1], [0, 2]], dtype=torch.long, device=device)
    return SelfCollisionCostCfg(
        weight=2.0,
        device_cfg=DeviceCfg(device),
        store_pair_distance=True,
        self_collision_kin_config=_KinematicsConfig(
            sphere_padding=torch.tensor([0.1, 0.0, 0.0], device=device),
            collision_pairs=pairs,
        ),
    )


def _spheres(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.1], [0.1, 0.0, 0.0, 0.1], [0.7, 0.0, 0.0, 0.1]]]],
        device=device,
    )


def test_config_targets_the_concrete_cost_and_validates_options() -> None:
    config = _config()
    assert config.class_type is SelfCollisionCost
    assert config.clone().class_type is SelfCollisionCost
    with pytest.raises(TypeError, match="store_pair_distance"):
        SelfCollisionCostCfg(weight=1.0, store_pair_distance=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_spheres"):
        SelfCollisionCostCfg(weight=1.0, self_collision_kin_config=object())


def test_setup_reuses_buffers_and_reset_clears_all_persistent_outputs() -> None:
    cost = SelfCollisionCost(_config())
    cost.setup_batch_tensors(2, 1)
    distance_id = id(cost._out_distance)
    gradient_id = id(cost._out_grad)
    cost.setup_batch_tensors(2, 1)
    assert id(cost._out_distance) == distance_id
    assert id(cost._out_grad) == gradient_id

    cost(_spheres().expand(2, -1, -1, -1).clone())
    assert cost._out_distance is not None
    assert cost._block_batch_max_value is not None
    cost._block_batch_max_value.fill_(3)
    cost._block_batch_max_index.fill_(2)
    cost.reset(torch.tensor([1], dtype=torch.long))
    assert cost._out_distance[0].sum() > 0
    assert cost._out_distance[1].sum() == 0
    assert cost._block_batch_max_value[1].sum() == 0
    assert cost._block_batch_max_index[1].sum() == 0


def test_empty_pair_topology_is_zero_and_differentiable() -> None:
    empty = torch.empty((0, 2), dtype=torch.long)
    spheres = _spheres().requires_grad_()
    cost = SelfCollisionCost(_config(pairs=empty))
    value = cost(spheres)
    torch.testing.assert_close(value, torch.zeros_like(value))
    value.sum().backward()
    torch.testing.assert_close(spheres.grad, torch.zeros_like(spheres))
    assert cost._pair_distance is not None and cost._pair_distance.shape == (1, 1, 0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_mps_reuses_persistent_buffers_without_cpu_fallback() -> None:
    device = torch.device("mps")
    cost = SelfCollisionCost(_config(device))
    cost.setup_batch_tensors(1, 1)
    initial = id(cost._out_distance)
    output = cost(_spheres(device).requires_grad_())
    output.sum().backward()
    cost.setup_batch_tensors(1, 1)
    assert output.device.type == "mps"
    assert id(cost._out_distance) == initial
