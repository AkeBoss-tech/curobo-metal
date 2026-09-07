"""Behavioural coverage for the portable V2 ``ParticleOptCore`` substrate."""

import torch
import pytest

from curobo._src.optim.components.particle_opt_core import ParticleOptCore, SampleMode
from curobo._src.optim.particle.mppi import BaseActionType, MPPICfg
from curobo._src.types.device_cfg import DeviceCfg


class _Rollout:
    action_horizon = 3
    action_dim = 2
    horizon = 3
    action_bound_lows = torch.full((2,), -2.0)
    action_bound_highs = torch.full((2,), 2.0)

    def __call__(self, action):
        return (action - 0.6).square().sum(dim=(-1, -2))


def _core(*, device_cfg=DeviceCfg("cpu"), null_fraction=0.0):
    config = MPPICfg(
        num_iters=4,
        inner_iters=2,
        num_particles=12,
        init_cov=0.5,
        null_act_frac=null_fraction,
        step_size_mean=1.0,
        seed=11,
        store_debug=True,
        sample_mode=SampleMode.BEST,
        base_action=BaseActionType.REPEAT,
        device_cfg=device_cfg,
    )
    core = None

    def update(trajectory):
        # This is deliberately an owner callback, matching V2's split between
        # shared particle lifecycle and optimizer-specific distribution update.
        costs = trajectory.costs_and_constraints.get_sum_cost_and_constraint(False)
        costs = costs.reshape(config.num_problems, config.num_particles, -1).sum(-1)
        weights = torch.softmax(-(costs - costs.amin(-1, keepdim=True)), dim=-1)
        assert core is not None
        population = core._last_population
        assert population is not None
        mean = (weights[..., None, None] * population).sum(dim=1)
        variance = (weights[..., None, None] * (population - mean[:, None]).square()).sum(dim=1).mean(dim=-2, keepdim=True)
        return {"mean": mean, "cov": variance.clamp_min(1e-4)}

    core = ParticleOptCore(config, [_Rollout()], update)
    return core


def test_particle_core_runs_batched_callback_lifecycle_and_returns_best_action():
    core = _core(device_cfg=DeviceCfg("cpu"))
    seed = torch.zeros(2, 3, 2)
    result = core.optimize(seed)
    assert result.shape == seed.shape
    assert (result - 0.6).square().sum() < (seed - 0.6).square().sum()
    assert core.top_trajs.shape == (2, 12, 3, 2)
    assert core.get_rollouts().shape == core.top_trajs.shape
    assert core._last_costs.shape == (2, 12, 1)
    assert core.problem_col.tolist() == [0, 1]
    assert core.opt_dt >= 0.0 and len(core.get_recorded_trace()["debug"]) == core.config.num_iters
    assert torch.isfinite(core._dist.cov).all() and bool((core._dist.cov > 0).all())


def test_particle_core_population_order_bounds_reset_and_shift_are_deterministic():
    core = _core(null_fraction=0.5, device_cfg=DeviceCfg("cpu"))
    seed = torch.full((1, 3, 2), 0.8)
    core.update_seed(seed)
    population = core.sample_actions()
    assert population.shape == (12, 3, 2)
    # V2 appends the negated mean and null-action sequences after sampled noise.
    torch.testing.assert_close(population[6:9], (-seed[0]).expand(3, -1, -1))
    torch.testing.assert_close(population[9:], torch.zeros_like(population[9:]))
    assert bool((population <= 2.0).all()) and bool((population >= -2.0).all())
    first = core._dist._sample_set.clone()
    assert core.reset_seed()
    torch.testing.assert_close(first, core._dist._sample_set)
    core.update_num_problems(2)
    core.reinitialize(torch.arange(12.0).reshape(2, 3, 2))
    before = core._dist.mean.clone()
    assert core.shift(1)
    torch.testing.assert_close(core._dist.mean[:, :-1], before[:, 1:])
    torch.testing.assert_close(core._dist.mean[:, -1:], before[:, -1:])
    core.reset_distribution(torch.tensor([1]))
    torch.testing.assert_close(core._dist.mean[1], torch.zeros_like(core._dist.mean[1]))


def test_particle_core_callback_validation_and_cuda_boundary_are_explicit():
    config = MPPICfg(num_particles=2, device_cfg=DeviceCfg("cpu"))
    with pytest.raises(TypeError, match="callable"):
        ParticleOptCore(config, [_Rollout()], None)
    with pytest.raises(NotImplementedError, match="CUDA Graph"):
        ParticleOptCore(config, [_Rollout()], lambda _: None, use_cuda_graph=True)
    core = _core(device_cfg=DeviceCfg("cpu"))
    with pytest.raises(ValueError, match="seed action"):
        core.update_seed(torch.zeros(5))
    with pytest.raises(ValueError, match="optimizer mppi"):
        core.update_solver_params({})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_particle_core_executes_on_mps_without_cpu_fallback():
    core = _core(device_cfg=DeviceCfg(device="mps", dtype=torch.float32))
    output = core.optimize(torch.zeros(1, 3, 2, device="mps"))
    assert output.device.type == "mps"
    assert core._dist.mean.device.type == "mps"
