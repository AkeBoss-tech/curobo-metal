"""Behavioural coverage for portable V2 gradient and ES facades."""

import torch

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.gradient_descent import GradientDescentOpt, GradientDescentOptCfg
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
    calc_exp,
    compute_es_mean,
)


class _Rollout:
    action_horizon = 2
    action_dim = 2
    horizon = 2

    def __call__(self, action):
        return action.square().sum(dim=(-1, -2))


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
