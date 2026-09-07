"""Behavioral coverage for the portable V2 self-collision cost facade."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class _Pairs:
    num_spheres: int = 3
    sphere_padding: torch.Tensor | None = None
    collision_pairs: torch.Tensor | None = None
    num_blocks_per_batch: int = 1


def _cfg(device: torch.device = torch.device("cpu"), *, store: bool = True):
    return SelfCollisionCostCfg(
        weight=2.0,
        device_cfg=DeviceCfg(device),
        store_pair_distance=store,
        self_collision_kin_config=_Pairs(
            sphere_padding=torch.tensor([0.1, 0.0, 0.2], device=device),
            collision_pairs=torch.tensor([[0, 1], [0, 2]], device=device),
        ),
    )


def _spheres(device: torch.device = torch.device("cpu"), *, requires_grad: bool = False):
    return torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.1], [0.1, 0.0, 0.0, 0.1], [0.5, 0.0, 0.0, 0.1]]]],
        device=device,
        requires_grad=requires_grad,
    )


def test_squared_overlap_pair_diagnostics_and_autograd() -> None:
    spheres = _spheres(requires_grad=True)
    cost = SelfCollisionCost(_cfg())
    value = cost(spheres)
    # pair (0, 1): (0.1 + 0.1 + 0.1)^2 - 0.1^2 = 0.08;
    # V2 output is half the configured weight times the max penetration.
    torch.testing.assert_close(value, torch.tensor([[[0.08]]]))
    torch.testing.assert_close(cost._pair_distance, torch.tensor([[[0.08, 0.0]]]))
    value.sum().backward()
    torch.testing.assert_close(spheres.grad[0, 0, 0], torch.tensor([0.2, 0.0, 0.0, 0.6]))
    torch.testing.assert_close(spheres.grad[0, 0, 1], torch.tensor([-0.2, 0.0, 0.0, 0.6]))
    torch.testing.assert_close(spheres.grad[0, 0, 2], torch.zeros(4))
    torch.testing.assert_close(cost.get_gradient_buffer(), spheres.grad)
    assert cost._sparse_sphere_idx.tolist() == [[[1, 1, 0]]]


def test_lifecycle_reset_disable_and_batch_validation() -> None:
    cost = SelfCollisionCost(_cfg(store=False))
    cost.setup_batch_tensors(2, 1)
    values = _spheres().expand(2, -1, -1, -1).clone()
    result = cost(values)
    assert result.shape == (2, 1, 1)
    assert cost._pair_distance.shape == (1,)
    cost.reset(torch.tensor([1], dtype=torch.long))
    assert cost._out_distance[0].sum() > 0
    assert cost._out_distance[1].sum() == 0
    cost.disable_cost()
    assert torch.count_nonzero(cost(values)) == 0
    cost.enable_cost()
    assert torch.count_nonzero(cost(values)) == 2
    with pytest.raises(ValueError, match="batch and horizon"):
        cost(_spheres())
    with pytest.raises(ValueError, match="out-of-range"):
        cost.reset(torch.tensor([3], dtype=torch.long))


def test_rejects_missing_config_and_invalid_pairs() -> None:
    with pytest.raises(ValueError, match="self_collision_kin_config"):
        SelfCollisionCost(SelfCollisionCostCfg(weight=1.0))
    config = _cfg()
    config.self_collision_kin_config.collision_pairs = torch.tensor([[0, 3]])
    with pytest.raises(ValueError, match="out-of-range"):
        SelfCollisionCost(config)(_spheres())


def test_disabled_sphere_radius_sentinel_is_ignored_outside_enabled_pairs() -> None:
    config = _cfg()
    config.self_collision_kin_config.collision_pairs = torch.tensor([[0, 1]])
    spheres = _spheres()
    spheres[..., 2, 3] = -100.0

    value = SelfCollisionCost(config)(spheres)

    torch.testing.assert_close(value, torch.tensor([[[0.08]]]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_mps_self_collision_is_resident_and_differentiable() -> None:
    spheres = _spheres(torch.device("mps"), requires_grad=True)
    cost = SelfCollisionCost(_cfg(torch.device("mps")))
    value = cost(spheres)
    value.sum().backward()
    assert value.device.type == spheres.grad.device.type == "mps"
    assert torch.isfinite(spheres.grad).all().item()
