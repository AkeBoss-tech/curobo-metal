"""Portable behavioural coverage for the pinned MPPI lifecycle."""

import torch

from curobo._src.optim.particle.mppi import BaseActionType, MPPI, MPPICfg
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
    return MPPICfg(
        num_iters=5,
        num_particles=32,
        init_cov=0.35,
        step_size_mean=1.0,
        step_size_cov=0.5,
        beta=0.2,
        seed=17,
        store_debug=True,
        store_rollouts=True,
        **kwargs,
    )


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


def test_mppi_mps_has_no_cpu_fallback_when_available():
    if not torch.backends.mps.is_available():
        return
    config = _config(device_cfg=DeviceCfg(device="mps", dtype=torch.float32))
    output = MPPI(config, [QuadraticRollout()]).optimize(torch.zeros(1, 3, 2, device="mps"))
    assert output.device.type == "mps"
