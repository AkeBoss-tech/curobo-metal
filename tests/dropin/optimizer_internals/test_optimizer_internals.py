import torch

from curobo._src.optim.gradient.conjugate_gradient import jit_cg_compute_step_direction
from curobo._src.optim.gradient.gradient_descent import GradientDescentOpt, GradientDescentOptCfg
from curobo._src.optim.gradient.lbfgs_jit_helpers import jit_lbfgs_compute_step_direction
from curobo._src.optim.optim_factory import create_optimizer
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.optim.particle.particle_opt_utils import cost_to_go, gaussian_entropy
from curobo._src.optim.particle.sample_strategies import ParticleSamplerCfg, MixedParticleSampler
from curobo._src.optim.util.levenberg_marquardt_step import (
    LevenbergMarquardtState,
    LevenbergMarquardtStep,
)
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
from curobo._src.optim.optim_factory import create_optimization_config, create_optimizer
from curobo._src.optim.particle.evolution_strategies import calc_exp, compute_es_mean
from curobo._src.optim.particle.mppi import (
    MPPICfg,
    jit_calculate_exp_util_from_costs,
    jit_mean_cov_diag_a,
)
from curobo._src.types.device_cfg import DeviceCfg


def test_iteration_state_clone_and_copy_are_device_resident():
    state = OptimizationIterationState(torch.tensor([[1.0, 2.0]]), cost=torch.tensor([5.0]))
    clone = state.clone()
    clone.action.add_(1)
    assert torch.equal(state.action, torch.tensor([[1.0, 2.0]]))
    state.copy_(clone)
    assert torch.equal(state.action, clone.action)


def test_gradient_helpers_and_optimizer_factory():
    grad = torch.tensor([[2.0, -1.0]])
    direction = jit_cg_compute_step_direction(grad, grad * 0.5, -grad, 1.0, "polak_ribiere")
    assert direction.shape == grad.shape and torch.isfinite(direction).all()
    step = jit_lbfgs_compute_step_direction(
        torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2), grad, 2, 1e-8,
    )
    assert torch.equal(step, -grad)

    cfg = GradientDescentOptCfg(num_iters=25, step_scale=0.1)
    optimizer = create_optimizer(cfg, [lambda x: x.square().sum(-1)])
    assert isinstance(optimizer, GradientDescentOpt)
    result = optimizer.optimize(torch.tensor([[2.0, -1.0]]))
    assert result.square().sum() < 1e-3


def test_deterministic_mixed_sampling_and_particle_math():
    cfg = ParticleSamplerCfg(
        DeviceCfg(), sample_ratio={"random": 0.5, "stomp": 0.5}, filter_coeffs=None
    )
    sampler = MixedParticleSampler(cfg, 6, 2)
    first = sampler.get_samples([8])
    sampler.reset_seed()
    second = sampler.get_samples([8])
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(cost_to_go(torch.tensor([[1.0, 2.0, 3.0]]), torch.ones(3)),
                               torch.tensor([[6.0, 5.0, 3.0]]))
    assert torch.isfinite(gaussian_entropy(cov=torch.eye(2)))


def test_portable_lm_step_and_gradients():
    parameter = torch.tensor([[2.0, -1.0]], requires_grad=True)
    jacobian = torch.eye(2).unsqueeze(0)
    output = torch.empty_like(parameter)
    reduction = torch.empty(1)
    state = LevenbergMarquardtState(
        jacobian, parameter, torch.tensor([0.1]), parameter, output, reduction
    )
    solved = LevenbergMarquardtStep(2, 2)(state)
    assert solved.shape == parameter.shape
    solved.sum().backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_mps_sampling_and_gradient_descent_without_fallback():
    if not torch.backends.mps.is_available():
        return
    device_cfg = DeviceCfg(device="mps", dtype=torch.float32)
    samples = MixedParticleSampler(ParticleSamplerCfg(device_cfg), 4, 2).get_samples([4])
    assert samples.device.type == "mps"
    cfg = GradientDescentOptCfg(num_iters=5, step_scale=0.1, device_cfg=device_cfg)
    result = GradientDescentOpt(cfg, [lambda x: x.square().sum(-1)]).optimize(
        torch.ones(2, 4, 2, device="mps")
    )
    assert result.device.type == "mps"


def test_particle_update_helpers_are_batched_and_normalized():
    costs = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]])
    actions = torch.arange(24.0).reshape(2, 3, 2, 2)
    weights = jit_calculate_exp_util_from_costs(costs, torch.ones(2), beta=0.5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))
    mean, covariance = jit_mean_cov_diag_a(
        costs, actions, torch.ones(2), torch.zeros(2, 2, 2), torch.ones(2, 2, 2),
        1.0, 1.0, 1e-4, 0.5,
    )
    assert mean.shape == covariance.shape == (2, 2, 2)
    assert bool((covariance >= 1e-4).all())
    # ES uses centered z-score utilities rather than MPPI's probability
    # weights.  It therefore has zero mean along the particle dimension.
    torch.testing.assert_close(calc_exp(costs.sum(-1)).mean(-1), torch.zeros(2))
    assert compute_es_mean(weights, actions, torch.zeros_like(mean), None, 3, 1.0).shape == mean.shape


def test_external_torch_and_factory_routes_execute_real_optimizers():
    objective = lambda value: value.square().sum(-1)
    config = TorchOptCfg(num_iters=20, step_scale=0.1, torch_optim_name="SGD")
    output = TorchOpt(config, [objective]).optimize(torch.tensor([[2.0, -1.0]]))
    assert output.square().sum() < 0.1
    for name, expected in (("es", "es"), ("scipy", "scipy"), ("torch", "torch")):
        cfg = create_optimization_config({"solver_type": name, "num_iters": 2}, DeviceCfg())
        assert cfg.solver_type == expected
        assert create_optimizer(cfg, [objective]).config is cfg


def test_portable_optimizer_exposes_bounds_and_goal_dt_lifecycle():
    class Rollout:
        action_horizon = 2
        action_dim = 3
        action_bound_lows = torch.full((3,), -1.0)
        action_bound_highs = torch.ones(3)
        def __call__(self, value): return value.square().sum(-1)
        def update_dt(self, value): self.dt = value

    rollout = Rollout()
    optimizer = GradientDescentOpt(GradientDescentOptCfg(num_iters=1), [rollout])
    torch.testing.assert_close(optimizer.action_bound_lows, torch.full((3,), -1.0))
    optimizer.update_goal_dt(0.02)
    assert rollout.dt == 0.02 and not optimizer.use_cuda_graph
