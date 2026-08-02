"""Behavioural coverage for portable V2 gradient and ES facades."""

import torch
import pytest

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.gradient_descent import GradientDescentOpt, GradientDescentOptCfg
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
    calc_exp,
    compute_es_mean,
)
from curobo._src.optim.components.particle_opt_core import SampleMode
from curobo._src.types.device_cfg import DeviceCfg


class _Rollout:
    action_horizon = 2
    action_dim = 2
    horizon = 2

    def __call__(self, action):
        return action.square().sum(dim=(-1, -2))


class _ShiftedQuadraticRollout:
    action_horizon = 3
    action_dim = 2
    horizon = 3

    def __call__(self, action):
        return (action - 0.75).square().sum(dim=-1)


def test_gradient_descent_uses_v2_step_and_best_return_policy():
    optimizer = GradientDescentOpt(
        GradientDescentOptCfg(num_iters=40, gradient_descent_step_scale=0.1), [_Rollout()]
    )
    seed = torch.full((1, 2, 2), 2.0)
    result = optimizer.optimize(seed)
    assert result.square().sum() < seed.square().sum()
    assert optimizer.get_recorded_trace() == {"debug": [], "debug_cost": []}
    assert optimizer.update_solver_params({"gradient_descent": {"num_iters": 2}})


def test_es_distribution_state_noise_and_zscore_are_portable():
    optimizer = EvolutionStrategies(EvolutionStrategiesCfg(num_particles=4), [_Rollout()])
    assert optimizer.mean_action.shape == (1, 2, 2)
    torch.testing.assert_close(optimizer.generate_noise([4], 9), optimizer.generate_noise([4], 9))
    sampled = optimizer.sample_actions(torch.zeros(1, 2, 2))
    assert sampled.shape == (4, 2, 2)
    utilities = calc_exp(torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(utilities.mean(-1), torch.zeros(1))
    actions = torch.arange(24.0).reshape(1, 3, 2, 4)
    mean = torch.zeros(1, 2, 4)
    output = compute_es_mean(utilities, actions, mean, torch.eye(4).unsqueeze(0), 3, 0.1)
    assert output.shape == mean.shape and torch.isfinite(output).all()


def test_es_runs_natural_gradient_distribution_lifecycle_deterministically():
    config = EvolutionStrategiesCfg(
        num_iters=12,
        num_particles=48,
        init_cov=0.4,
        learning_rate=0.15,
        step_size_mean=1.0,
        step_size_cov=0.3,
        seed=23,
        store_debug=True,
        store_rollouts=True,
        sample_mode=SampleMode.BEST,
    )
    seed = torch.zeros(2, 3, 2)
    first = EvolutionStrategies(config, [_ShiftedQuadraticRollout()])
    second = EvolutionStrategies(config, [_ShiftedQuadraticRollout()])
    first_result = first.optimize(seed)
    second_result = second.optimize(seed)
    torch.testing.assert_close(first_result, second_result)
    assert first_result.square().sum() > 0.0
    assert first_result.shape == seed.shape
    assert first.top_trajs.shape == (2, 20, 3, 2)
    assert first._last_utilities.shape == (2, 48)
    assert torch.isfinite(first.cov_action).all() and bool((first.cov_action > 0).all())
    assert len(first.get_debug()["objective"]) == config.num_iters
    torch.testing.assert_close(first._get_action_seq("best"), first.best_traj)
    samples = first.sample_actions(first.mean_action)
    assert samples.shape == (96, 3, 2)
    before = first.mean_action.clone()
    first.shift(1)
    torch.testing.assert_close(first.mean_action[:, :-1], before[:, 1:])


def test_es_utility_degeneracy_and_argument_validation_are_explicit():
    utilities = calc_exp(torch.tensor([[1.0], [float("inf")]]))
    torch.testing.assert_close(utilities, torch.zeros_like(utilities))
    with pytest.raises(ValueError, match="learning_rate"):
        EvolutionStrategiesCfg(learning_rate=0.0)
    with pytest.raises(ValueError, match="full_inv_cov"):
        compute_es_mean(
            torch.ones(1, 2), torch.zeros(1, 2, 1, 2), torch.zeros(1, 1, 2),
            torch.eye(3).unsqueeze(0), 2, 0.1,
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS hardware")
def test_es_runs_on_mps_without_cpu_fallback():
    config = EvolutionStrategiesCfg(
        num_iters=3, num_particles=12, init_cov=0.2, learning_rate=0.1,
        device_cfg=DeviceCfg(device="mps", dtype=torch.float32), seed=3,
    )
    optimizer = EvolutionStrategies(config, [_ShiftedQuadraticRollout()])
    output = optimizer.optimize(torch.zeros(1, 3, 2, device="mps"))
    assert output.device.type == "mps"
    assert optimizer.cov_action.device.type == "mps"


def test_gradient_core_callbacks_and_cuda_boundary():
    called = []
    core = GradientOptCore(
        GradientDescentOptCfg(num_iters=1), [_Rollout()], lambda state: state,
        on_resize=called.append,
    )
    core.update_num_problems(2)
    assert called == [2]
    assert core.shift(1)
    assert core.debug_dump() is None
    try:
        GradientOptCore(GradientDescentOptCfg(), [_Rollout()], lambda state: state, use_cuda_graph=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("portable gradient core must reject raw CUDA graph capture")
