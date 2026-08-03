"""Lifecycle regression coverage for the portable V2 ES facade."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.components.particle_opt_core import SampleMode
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
)
from curobo._src.optim.particle.sample_strategies import ParticleSamplerCfg
from curobo._src.types.device_cfg import DeviceCfg


class _QuadraticRollout:
    action_horizon = 3
    action_dim = 2
    horizon = 3

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - 0.4).square().sum(dim=(-1, -2))


def _optimizer(*, fixed_samples: bool, device_cfg: DeviceCfg = DeviceCfg()) -> EvolutionStrategies:
    sampler = ParticleSamplerCfg(device_cfg=device_cfg, fixed_samples=fixed_samples, seed=31)
    config = EvolutionStrategiesCfg(
        device_cfg=device_cfg,
        num_iters=3,
        num_particles=10,
        num_problems=2,
        null_act_frac=0.4,
        init_cov=0.25,
        seed=7,
        sample_params=sampler,
    )
    return EvolutionStrategies(config, [_QuadraticRollout()])


def test_es_population_preserves_shared_particle_order_and_fixed_sample_schedule():
    optimizer = _optimizer(fixed_samples=True)
    seed = torch.full((2, 3, 2), 0.6)
    optimizer.update_seed(seed)
    first = optimizer._population(2, 3, 2, device=seed.device, dtype=seed.dtype, iteration=0)
    second = optimizer._population(2, 3, 2, device=seed.device, dtype=seed.dtype, iteration=1)

    # 10 particles with a 0.4 fraction leaves six stochastic entries, then
    # appends two negated and two null actions in pinned V2 order.
    assert first.shape == (2, 10, 3, 2)
    torch.testing.assert_close(first[:, 5], seed)
    torch.testing.assert_close(first[:, 6:8], (-seed).unsqueeze(1).expand(-1, 2, -1, -1))
    torch.testing.assert_close(first[:, 8:], torch.zeros_like(first[:, 8:]))
    torch.testing.assert_close(first, second)
    assert optimizer._sample_iter is not None
    assert optimizer._sample_iter.shape == (1, 2, 6, 3, 2)


def test_es_nonfixed_schedule_advances_deterministically_per_iteration():
    first = _optimizer(fixed_samples=False)
    second = _optimizer(fixed_samples=False)
    seed = torch.zeros(2, 3, 2)
    first.update_seed(seed)
    second.update_seed(seed)
    a0 = first._population(2, 3, 2, device=seed.device, dtype=seed.dtype, iteration=0)
    a1 = first._population(2, 3, 2, device=seed.device, dtype=seed.dtype, iteration=1)
    b0 = second._population(2, 3, 2, device=seed.device, dtype=seed.dtype, iteration=0)
    b1 = second._population(2, 3, 2, device=seed.device, dtype=seed.dtype, iteration=1)

    torch.testing.assert_close(a0, b0)
    torch.testing.assert_close(a1, b1)
    assert not torch.equal(a0[:, :6], a1[:, :6])
    assert first._sample_iter is not None
    assert first._sample_iter.shape == (3, 2, 6, 3, 2)


def test_es_sample_query_is_one_per_problem_and_does_not_reset_warm_start():
    optimizer = _optimizer(fixed_samples=False)
    seed = torch.zeros(2, 3, 2)
    optimizer.optimize(seed)
    mean_before = optimizer.mean_action.clone()
    cov_before = optimizer.cov_action.clone()
    best_before = optimizer.best_traj.clone()
    cursor_before = optimizer._sample_cursor
    sample = optimizer._get_action_seq(SampleMode.SAMPLE)

    assert sample.shape == seed.shape
    torch.testing.assert_close(optimizer.mean_action, mean_before)
    torch.testing.assert_close(optimizer.cov_action, cov_before)
    torch.testing.assert_close(optimizer.best_traj, best_before)
    assert optimizer._sample_cursor == cursor_before
    with pytest.raises(ValueError, match="unidentified ES sample mode"):
        optimizer._get_action_seq("not-a-mode")


@pytest.mark.parametrize("learning_rate", [True, float("inf"), float("nan"), "0.1"])
def test_es_rejects_nonfinite_or_nonreal_learning_rate(learning_rate):
    with pytest.raises(ValueError, match="finite positive real"):
        EvolutionStrategiesCfg(learning_rate=learning_rate)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_es_fixed_population_and_sample_query_run_on_mps_without_fallback():
    device_cfg = DeviceCfg(device="mps", dtype=torch.float32)
    optimizer = _optimizer(fixed_samples=True, device_cfg=device_cfg)
    output = optimizer.optimize(torch.zeros(2, 3, 2, device="mps"))
    sample = optimizer._get_action_seq(SampleMode.SAMPLE)
    assert output.device.type == sample.device.type == "mps"
    assert optimizer._sample_iter is not None and optimizer._sample_iter.device.type == "mps"
