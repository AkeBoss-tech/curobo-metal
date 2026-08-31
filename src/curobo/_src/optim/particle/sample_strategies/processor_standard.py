from typing import Optional, Tuple

import torch

from curobo._src.types.device_cfg import DeviceCfg

from .stomp_covariance import get_stomp_cov


class _StandardParticleProcessorPortableMixin:
    def __init__(self,horizon,action_dim,device_cfg=None,filter_coeffs=None):
        self.input_horizon=self.horizon=horizon; self.action_dim=action_dim
        self.input_ndims=horizon*action_dim; self.filter_coeffs=filter_coeffs
    def _filter_samples(self,eps):
        if self.filter_coeffs is None:
            return eps
        beta_0, beta_1, beta_2 = self.filter_coeffs
        for index in range(2, eps.shape[1]):
            eps[:, index, :] = (
                beta_0 * eps[:, index, :]
                + beta_1 * eps[:, index - 1, :]
                + beta_2 * eps[:, index - 2, :]
            )
        return eps
    def _filter_smooth(self,samples):
        if samples.shape[0] == 0:
            return samples
        if self.stomp_matrix is None:
            matrix, scale, _ = get_stomp_cov(
                self.horizon, zero_out_boundary=True, stencil_type="3point"
            )
            self.stomp_matrix = self.device_cfg.to_device(matrix)
            self.stomp_scale_tril = self.device_cfg.to_device(scale)
        filtered = torch.matmul(self.stomp_matrix, samples)
        return filtered / torch.max(torch.abs(filtered))
    def process_samples(self,samples,filter_smooth=False):
        return self._filter_smooth(samples) if filter_smooth else self._filter_samples(samples)


class StandardParticleProcessor(_StandardParticleProcessorPortableMixin):
    """Pinned particle-processor declaration using portable filtering."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        filter_coeffs: Optional[Tuple[float, float, float]] = None,
        device_cfg: DeviceCfg = None,
    ):
        _StandardParticleProcessorPortableMixin.__init__(
            self,
            horizon=horizon,
            action_dim=action_dim,
            device_cfg=device_cfg,
            filter_coeffs=filter_coeffs,
        )
        self.device_cfg = device_cfg
        self.ndims = horizon * action_dim
        self.stomp_matrix = None
        self.stomp_scale_tril = None

    def process_samples(self, samples: torch.Tensor, filter_smooth: bool = False) -> torch.Tensor:
        return _StandardParticleProcessorPortableMixin.process_samples(
            self, samples, filter_smooth
        )
