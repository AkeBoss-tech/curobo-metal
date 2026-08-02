"""Portable behavioural coverage for the pinned MPPI lifecycle."""

import torch

from curobo._src.optim.particle.mppi import BaseActionType, MPPI, MPPICfg
from curobo._src.optim.components.gaussian_distribution import CovType
from curobo._src.optim.particle.sample_strategies import ParticleSamplerCfg
from curobo._src.types.device_cfg import DeviceCfg


class QuadraticRollout:
    action_horizon = 3
    action_dim = 2
    action_bound_lows = torch.full((2,), -1.0)
    action_bound_highs = torch.full((2,), 1.0)

    def __call__(self, action):
        # A per-timestep cost validates gamma and rollout-sequence handling.
        return (action - 0.5).square().sum(dim=-1)


def _config(**kwargs):
    values = dict(
        num_iters=5,
        num_particles=32,
        init_cov=0.35,
        step_size_mean=1.0,
        step_size_cov=0.5,
        beta=0.2,
        seed=17,
        store_debug=True,
        store_rollouts=True,
    )
    values.update(kwargs)
    return MPPICfg(**values)


def test_mppi_distribution_is_deterministic_batched_and_stateful():
    seed = torch.zeros(2, 3, 2)
    first = MPPI(_config(), [QuadraticRollout()])
    second = MPPI(_config(), [QuadraticRollout()])
    first_result = first.optimize(seed)
    second_result = second.optimize(seed)
    torch.testing.assert_close(first_result, second_result)
    assert first_result.shape == seed.shape
    assert first.mean_action.shape == (2, 3, 2)
    assert first.cov_action.shape == first.scale_tril.shape == (2, 1, 2)
    assert first.best_traj.shape == seed.shape
    assert first.top_trajs.shape == (2, 20, 3, 2)
    assert len(first.get_debug()["objective"]) == 5
    assert bool((first_result <= 1.0).all() and (first_result >= -1.0).all())


def test_mppi_sampling_reset_and_shift_fill_contract():
    optimizer = MPPI(_config(base_action=BaseActionType.NULL), [QuadraticRollout()])
    samples = optimizer.sample_actions(torch.ones(1, 3, 2))
    assert samples.shape == (32, 3, 2)
    assert samples.device.type == "cpu"
    optimizer.optimize(torch.full((1, 3, 2), 0.25))
    before = optimizer.mean_action.clone()
    optimizer.shift(1)
    torch.testing.assert_close(optimizer.mean_action[:, :-1], before[:, 1:])
    torch.testing.assert_close(optimizer.mean_action[:, -1], torch.zeros(1, 2))
    optimizer.reinitialize(torch.full((1, 3, 2), -0.25))
    torch.testing.assert_close(optimizer.mean_action, torch.full((1, 3, 2), -0.25))
    optimizer.reset_distribution()
    torch.testing.assert_close(optimizer.cov_action, torch.full((1, 1, 2), 0.35))


def test_mppi_best_and_sample_modes_and_cuda_boundary():
    seed = torch.zeros(1, 3, 2)
    best = MPPI(_config(sample_mode="best"), [QuadraticRollout()])
    result = best.optimize(seed)
    torch.testing.assert_close(result, best.best_traj)
    sampled = MPPI(_config(sample_mode="sample"), [QuadraticRollout()])
    assert sampled.optimize(seed).shape == seed.shape
    try:
        MPPI(_config(), [QuadraticRollout()], use_cuda_graph=True)
    except NotImplementedError as error:
        assert "CUDA Graph" in str(error)
    else:
        raise AssertionError("raw CUDA graph capture must not be emulated")


def test_mppi_particle_mix_and_shared_sample_schedule():
    cfg = _config(num_particles=10, null_act_frac=0.4, sample_per_problem=False)
    optimizer = MPPI(cfg, [QuadraticRollout()])
    actions = optimizer.sample_actions(torch.full((2, 3, 2), 0.25))
    population = actions.reshape(2, 10, 3, 2)
    assert optimizer.null_per_problem == 2
    assert optimizer.neg_per_problem == 2
    assert optimizer.sampled_particles_per_problem == 6
    # First six entries are shared stochastic samples; the remaining entries
    # are two negated means followed by two null actions, in V2 order.
    torch.testing.assert_close(population[0, :6], population[1, :6])
    torch.testing.assert_close(population[:, 6:8], torch.full((2, 2, 3, 2), -0.25))
    torch.testing.assert_close(population[:, 8:], torch.zeros(2, 2, 3, 2))


def test_mppi_sigma_i_and_nonfixed_sample_cursor_lifecycle():
    cfg = _config(
        cov_type=CovType.SIGMA_I,
        init_cov=0.2,
        num_particles=12,
        sample_params=ParticleSamplerCfg(DeviceCfg(), fixed_samples=False, seed=5),
    )
    optimizer = MPPI(cfg, [QuadraticRollout()])
    seed = torch.zeros(2, 3, 2)
    first = optimizer.sample_actions(seed)
    second = optimizer.sample_actions(None)
    assert first.shape == second.shape == (24, 3, 2)
    assert not torch.equal(first, second)
    result = optimizer.optimize(seed)
    assert result.shape == seed.shape
    assert optimizer.cov_action.shape == (2, 1, 1)
    assert optimizer.full_scale_tril.shape == optimizer.full_inv_cov.shape == (2, 2, 2)
    torch.testing.assert_close(optimizer._get_action_seq("mean"), optimizer.mean_action)
    assert optimizer.get_rollouts().shape == (2, 12, 3, 2)
    assert optimizer.update_solver_params({"mppi": {"num_iters": 2}}) is True
    assert optimizer.config.num_iters == 2


def test_mppi_mps_has_no_cpu_fallback_when_available():
    if not torch.backends.mps.is_available():
        return
    config = _config(device_cfg=DeviceCfg(device="mps", dtype=torch.float32))
    output = MPPI(config, [QuadraticRollout()]).optimize(torch.zeros(1, 3, 2, device="mps"))
    assert output.device.type == "mps"
