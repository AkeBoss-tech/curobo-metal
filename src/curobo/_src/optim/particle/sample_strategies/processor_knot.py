import numpy as np
import scipy.interpolate as si
import torch

from curobo._src.types.device_cfg import DeviceCfg


class _KnotParticleProcessorPortableMixin:
    def __init__(self,horizon,action_dim,n_knots=3,device_cfg=None,degree=3,**kwargs):
        del device_cfg,kwargs; self.input_horizon=n_knots; self.horizon=horizon; self.action_dim=action_dim; self.input_ndims=n_knots*action_dim; self.degree=degree
    @staticmethod
    def bspline(c_arr,t_arr=None,n=100,degree=3):
        if c_arr.ndim != 1:
            raise ValueError("B-spline control points must be a one-dimensional tensor")
        if n < 0:
            raise ValueError("n must be nonnegative")
        if c_arr.numel() == 0:
            raise ValueError("B-spline requires at least one control point")
        if n == 0:
            return c_arr.new_empty((0,))
        if c_arr.numel() == 1:
            return c_arr.expand(n).clone()
        sample_device, sample_dtype = c_arr.device, c_arr.dtype
        values = c_arr.detach().cpu().numpy()
        parameter = (
            np.linspace(0, values.shape[0], values.shape[0])
            if t_arr is None
            else t_arr.detach().cpu().numpy()
        )
        actual_degree = min(int(degree), max(1, values.shape[0] - 1))
        spline = si.splrep(parameter, values, k=actual_degree, s=0.5)
        samples = si.splev(np.linspace(0, values.shape[0], n), spline, ext=3)
        return torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    def process_samples(self,samples,filter_smooth=False):
        del filter_smooth
        if samples.ndim != 2:
            raise ValueError("knot samples must have [batch, n_knots * action_dim] dimensions")
        batch = samples.shape[0]
        knots = samples.reshape(batch, self.action_dim, self.input_horizon)
        output = samples.new_empty((batch, self.horizon, self.action_dim))
        for batch_index in range(batch):
            for action_index in range(self.action_dim):
                output[batch_index, :, action_index] = self.bspline(
                    knots[batch_index, action_index], n=self.horizon, degree=self.degree
                )
        return output


class KnotParticleProcessor(_KnotParticleProcessorPortableMixin):
    """Pinned knot-processor declaration using portable interpolation."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        n_knots: int,
        degree: int = 3,
        device_cfg: DeviceCfg = None,
    ):
        _KnotParticleProcessorPortableMixin.__init__(
            self,
            horizon=horizon,
            action_dim=action_dim,
            n_knots=n_knots,
            device_cfg=device_cfg,
            degree=degree,
        )
        self.n_knots = n_knots
        self.ndims = horizon * action_dim
        self.device_cfg = device_cfg or DeviceCfg()

    def process_samples(self, samples: torch.Tensor, filter_smooth: bool = False) -> torch.Tensor:
        return _KnotParticleProcessorPortableMixin.process_samples(
            self, samples, filter_smooth
        )

    @staticmethod
    def bspline(c_arr: torch.Tensor, t_arr=None, n=100, degree=3):
        return _KnotParticleProcessorPortableMixin.bspline(c_arr, t_arr, n, degree)
