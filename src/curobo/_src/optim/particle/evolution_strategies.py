"""Portable Evolution Strategies compatibility layer.

ES uses the shared deterministic particle implementation for the actual
search, and exposes the V2 distribution/lifecycle helpers for callers that
inspect or control the sampling state.
"""

from dataclasses import dataclass

import torch

from .mppi import MPPI, MPPICfg
from .sample_strategies import ParticleSamplerCfg
from ..components.gaussian_distribution import GaussianDistribution


@dataclass
class EvolutionStrategiesCfg(MPPICfg):
    solver_type: str = "es"
    solver_name: str = "es"
    learning_rate: float = 0.1


class EvolutionStrategies(MPPI):
    # The portable particle primitive implements CEM-style covariance updates;
    # ES differs in its exported z-score utilities and natural-gradient mean
    # update below.  It deliberately does not pretend to expose Warp kernels.
    strategy = "cem"

    def __init__(self, config, rollout_list, use_cuda_graph: bool = False):
        super().__init__(config, rollout_list, use_cuda_graph=use_cuda_graph)
        sampler_cfg = config.sample_params or ParticleSamplerCfg(config.device_cfg, seed=config.seed)
        init_mean = config.init_mean
        if init_mean is None:
            init_mean = torch.zeros(1, self.action_horizon, self.action_dim,
                                    device=config.device_cfg.device, dtype=config.device_cfg.dtype)
        self._dist = GaussianDistribution(
            config.device_cfg, self.action_horizon, self.action_dim,
            config.cov_type, init_mean, config.init_cov, sampler_cfg,
            config.random_mean, config.seed,
        )
        self._dist.reset(config.num_problems)
        self._distribution_shape = (self.action_horizon, self.action_dim)
        self._top_trajs = None
        self._best_action = None

    def _ensure_distribution_layout(self, action: torch.Tensor) -> tuple[int, int]:
        """Infer a callable rollout's event layout from a concrete seed."""
        if action.ndim >= 3:
            horizon, action_dim = action.shape[-2:]
        elif action.ndim >= 1:
            horizon, action_dim = 1, action.shape[-1]
        else:
            raise ValueError("ES seed_action must have at least one dimension")
        if (horizon, action_dim) == self._distribution_shape:
            return horizon, action_dim
        sampler_cfg = self.config.sample_params or ParticleSamplerCfg(
            self.config.device_cfg, seed=self.config.seed
        )
        init_mean = torch.zeros(
            1, horizon, action_dim, device=action.device, dtype=action.dtype
        )
        self._dist = GaussianDistribution(
            self.config.device_cfg, horizon, action_dim, self.config.cov_type,
            init_mean, self.config.init_cov, sampler_cfg, self.config.random_mean,
            self.config.seed,
        )
        self._dist.reset(self.config.num_problems)
        self._distribution_shape = (horizon, action_dim)
        return horizon, action_dim

    @property
    def mean_action(self):
        return self._dist.mean

    @mean_action.setter
    def mean_action(self, value):
        self._dist.mean = value

    @property
    def best_traj(self):
        return self._best_action

    @best_traj.setter
    def best_traj(self, value):
        self._best_action = value

    @property
    def cov_action(self):
        return self._dist.cov

    @cov_action.setter
    def cov_action(self, value):
        self._dist.update_cov_scale(value)

    @property
    def scale_tril(self):
        return self._dist.scale_tril

    @scale_tril.setter
    def scale_tril(self, value):
        self._dist.scale_tril = value

    @property
    def inv_cov_action(self):
        return self._dist.inv_cov

    @property
    def full_scale_tril(self):
        return self._dist.full_scale_tril

    @property
    def full_inv_cov(self):
        return self._dist.full_inv_cov

    @property
    def sample_lib(self):
        return self._dist.sample_lib

    @property
    def particles_per_problem(self):
        return self.config.num_particles

    @property
    def sampled_particles_per_problem(self):
        return self.config.num_particles

    @property
    def total_num_particles(self):
        return self.config.num_problems * self.config.num_particles

    @property
    def problem_col(self):
        return torch.arange(self.config.num_problems, device=self.device_cfg.device)

    @property
    def gamma_seq(self):
        return torch.ones(self.action_horizon, device=self.device_cfg.device, dtype=self.device_cfg.dtype)

    @property
    def top_trajs(self):
        return self._top_trajs

    def optimize(self, seed_action):
        horizon, action_dim = self._ensure_distribution_layout(seed_action)
        result = super().optimize(seed_action)
        self._best_action = result.detach().clone()
        self._dist.mean = result.detach().reshape(self.config.num_problems, horizon, action_dim).clone()
        return result

    def update_num_problems(self, num_problems):
        super().update_num_problems(num_problems)
        if hasattr(self, "_dist"):
            self._dist.reset(num_problems)

    def update_init_mean(self, init_mean):
        self._dist.update_mean(init_mean, self.config.num_problems)

    def reset_mean(self, reset_problem_ids=None):
        return self._dist.reset_mean(self.config.num_problems, reset_problem_ids)

    def reset_covariance(self, reset_problem_ids=None):
        del reset_problem_ids
        return self._dist.reset_covariance(self.config.num_problems)

    def reset_distribution(self, reset_problem_ids=None):
        self.reset_mean(reset_problem_ids)
        self.reset_covariance(reset_problem_ids)

    def initialize_samples(self):
        return self._dist.initialize_samples(
            self.config.num_problems, self.config.num_particles, self.config.num_iters,
            True, self.config.sample_per_problem,
        )

    def update_samples(self):
        return self._dist.update_samples(
            self.config.num_problems, self.config.num_particles, self.config.num_iters,
            True, self.config.sample_per_problem,
        )

    def generate_noise(self, shape, base_seed=None):
        return self._dist.generate_noise(shape, base_seed)

    def sample_actions(self, init_act):
        noise = self.generate_noise([self.total_num_particles])
        mean = init_act.reshape(self.config.num_problems, self.action_horizon, self.action_dim)
        return mean.repeat_interleave(self.config.num_particles, dim=0) + noise * self.scale_tril.repeat_interleave(self.config.num_particles, dim=0).unsqueeze(-2)

    def update_seed(self, init_act):
        self.update_init_mean(init_act)
        return self.mean_action

    def get_rollouts(self):
        return self.get_all_rollout_instances()


def calc_exp(total_costs):
    """V2 ES z-score utilities (not MPPI softmax weights)."""
    negated = -total_costs
    std = torch.std(negated, dim=-1, keepdim=True, unbiased=True)
    eps = torch.finfo(total_costs.dtype).eps
    # V2 returns NaN for a singleton population.  A portable optimizer needs
    # deterministic finite values for that supported shape, so use zero utility.
    return torch.where(std > eps, (negated - negated.mean(dim=-1, keepdim=True)) / std, torch.zeros_like(negated))


def compute_es_mean(w, actions, mean_action, full_inv_cov, num_particles, learning_rate):
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    while w.ndim < actions.ndim:
        w = w.unsqueeze(-1)
    std_w = torch.std(w, dim=(-3, -2, -1), keepdim=True, unbiased=False).clamp_min(torch.finfo(actions.dtype).eps)
    centred = (actions - mean_action.unsqueeze(-3)) / std_w
    weighted = (w * centred).sum(dim=-3)
    if full_inv_cov is None:
        inv_diag = torch.ones_like(mean_action)
    else:
        inv_diag = torch.diagonal(full_inv_cov, dim1=-2, dim2=-1).unsqueeze(-2)
    return mean_action + float(learning_rate) * weighted * inv_diag / float(num_particles)


__all__ = ["EvolutionStrategiesCfg", "EvolutionStrategies", "calc_exp", "compute_es_mean"]
