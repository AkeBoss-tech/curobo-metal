"""Portable behavioral coverage for the pinned PRM node sampler."""

from __future__ import annotations

import pytest
import torch

from curobo._src.graph_planner.graph.node_sampling_strategy import NodeSamplingStrategy
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg


def _strategy(device: str = "cpu", *, method: str = "householder") -> NodeSamplingStrategy:
    device_cfg = DeviceCfg(device=device)
    low = torch.tensor([-2.0, -1.0], device=device)
    high = torch.tensor([2.0, 1.0], device=device)
    cfg = PRMGraphPlannerCfg(
        device_cfg=device_cfg,
        sampler_seed=17,
        sampler_buffer_size=32,
        sample_rejection_ratio=2,
        ellipsoid_projection_method=method,
    )
    return NodeSamplingStrategy(
        cfg,
        low,
        high,
        torch.ones(2, device=device),
        2,
        lambda samples: samples[:, 0] >= -0.25,
        device_cfg,
    )


def test_halton_sampler_reset_bounds_and_low_dim_ball() -> None:
    sampler = _strategy()
    first = sampler.generate_action_samples(9)
    sampler.reset_seed()
    torch.testing.assert_close(first, sampler.generate_action_samples(9))
    assert (first >= sampler.action_bound_lows).all()
    assert (first <= sampler.action_bound_highs).all()

    ball = sampler.generate_action_samples(16, unit_ball=True)
    assert (torch.linalg.vector_norm(ball, dim=-1) <= 1.0 + 1.0e-6).all()
    assert (torch.linalg.vector_norm(ball, dim=-1) < 0.95).any()


@pytest.mark.parametrize("method", ["householder", "svd", "approximate"])
def test_ellipsoid_methods_filter_and_clamp_feasible_nodes(method: str) -> None:
    sampler = _strategy(method=method)
    result = sampler.generate_feasible_samples_in_ellipsoid(
        torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0]), 12, 3.0
    )
    assert result.ndim == 2 and result.shape[1] == 2
    assert result.shape[0] <= 12
    assert (result[:, 0] >= -0.25).all()
    assert (result >= sampler.action_bound_lows).all()
    assert (result <= sampler.action_bound_highs).all()


def test_feasibility_and_line_distance_validate_public_shapes() -> None:
    sampler = _strategy()
    with pytest.raises(ValueError, match="2D tensor"):
        sampler.check_samples_feasibility(torch.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        sampler.compute_distance_from_line(
            torch.zeros(2, 3), torch.zeros(2), torch.ones(2)
        )

    distance = sampler.compute_distance_from_line(
        torch.tensor([[0.5, 1.0], [-1.0, 0.0], [2.0, 0.0]]),
        torch.zeros(2),
        torch.ones(2),
    )
    torch.testing.assert_close(
        distance, torch.tensor([2.0**-1.5, 1.0, 2.0**0.5])
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_node_sampling_runs_on_mps_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    sampler = _strategy("mps", method="svd")
    first = sampler.generate_action_samples(8)
    sampler.reset_seed()
    assert first.device.type == "mps"
    torch.testing.assert_close(first, sampler.generate_action_samples(8))
    ball = sampler.generate_action_samples(8, unit_ball=True)
    ellipsoid = sampler.generate_feasible_samples_in_ellipsoid(
        torch.tensor([-1.0, 0.0], device="mps"),
        torch.tensor([1.0, 0.0], device="mps"),
        8,
        3.0,
    )
    assert ball.device.type == ellipsoid.device.type == "mps"
