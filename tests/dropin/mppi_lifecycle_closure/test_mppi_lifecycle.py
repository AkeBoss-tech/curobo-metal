"""Additional CPU/MPS value-lifecycle coverage for portable MPPI."""

import pytest
import torch

from curobo._src.optim.particle.mppi import BaseActionType, MPPI, MPPICfg
from curobo._src.optim.particle.sample_strategies import ParticleSamplerCfg
from curobo._src.types.device_cfg import DeviceCfg


class QuadraticRollout:
    action_horizon = 3
    action_dim = 2
    action_bound_lows = torch.full((2,), -10.0)
    action_bound_highs = torch.full((2,), 10.0)

    def __call__(self, action):
        return action.square().sum(dim=-1)


def _cfg(**kwargs):
    values = dict(num_iters=2, num_particles=8, init_cov=0.4, seed=17)
    values.update(kwargs)
    return MPPICfg(**values)


def test_config_factory_keeps_mppi_specific_values_and_filters_unrelated_entries():
    config_values = MPPICfg.create_data_dict(
        {"num_particles": 12, "beta": 0.25, "null_act_frac": 0.25, "unknown": "ignored"},
        DeviceCfg(),
    )
    config = MPPICfg(**config_values)
    assert config.num_particles == 12
    assert config.beta == 0.25
    assert config.null_act_frac == 0.25
    assert "unknown" not in config_values


def test_sampler_seed_and_best_trajectory_shift_are_persistent_and_deterministic():
    seed = torch.zeros(1, 3, 2)
    first = MPPI(_cfg(sample_params=ParticleSamplerCfg(DeviceCfg(), seed=3)), [QuadraticRollout()])
    second = MPPI(_cfg(sample_params=ParticleSamplerCfg(DeviceCfg(), seed=4)), [QuadraticRollout()])
    assert not torch.equal(first.sample_actions(seed), second.sample_actions(seed))

    optimizer = MPPI(_cfg(base_action=BaseActionType.REPEAT), [QuadraticRollout()])
    optimizer.sample_actions(seed)
    optimizer.mean_action = torch.tensor([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]])
    optimizer.best_traj = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    optimizer.shift(1)
    torch.testing.assert_close(optimizer.mean_action, torch.tensor([[[20.0, 20.0], [30.0, 30.0], [30.0, 30.0]]]))
    # Best trajectory retains its independently shifted value rather than
    # being replaced with the (separate) distribution mean.
    torch.testing.assert_close(optimizer.best_traj, torch.tensor([[[2.0, 2.0], [3.0, 3.0], [3.0, 3.0]]]))


def test_parameter_update_revalidates_and_rebuilds_distribution_state():
    optimizer = MPPI(_cfg(), [QuadraticRollout()])
    optimizer.optimize(torch.zeros(2, 3, 2))
    old_samples = optimizer._sample_iter
    assert optimizer.update_solver_params({"mppi": {"num_particles": 6, "init_cov": 0.2}})
    assert optimizer.config.num_particles == 6
    assert optimizer._sample_iter is None
    assert optimizer.cov_action.shape == (2, 1, 2)
    torch.testing.assert_close(optimizer.cov_action, torch.full((2, 1, 2), 0.2))
    assert old_samples is not optimizer._sample_iter
    with pytest.raises(ValueError, match="null_act_frac"):
        optimizer.update_solver_params({"mppi": {"null_act_frac": 1.0, "num_particles": 1}})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_config_and_sampler_state_remain_mps_resident_without_fallback():
    cfg = _cfg(
        device_cfg=DeviceCfg("mps", torch.float32),
        init_mean=torch.zeros(3, 2),
        sample_params=ParticleSamplerCfg(DeviceCfg(), seed=9),
    )
    optimizer = MPPI(cfg, [QuadraticRollout()])
    output = optimizer.optimize(torch.zeros(1, 3, 2, device="mps"))
    assert cfg.init_mean.device.type == "mps"
    assert cfg.sample_params.device_cfg.device.type == "mps"
    assert output.device.type == optimizer.mean_action.device.type == "mps"
