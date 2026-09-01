"""Device-resident deterministic sample buffer."""

from __future__ import annotations

import numpy as np
import torch
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator
import torch.autograd.profiler as profiler
from typing import List, Optional

from .sequencer_base import BaseSequencer


class SampleBuffer:
    def __init__(
        self, sequencer: BaseSequencer, ndims: int, device_cfg: DeviceCfg = DeviceCfg(),
        up_bounds: List[float] = [1], low_bounds: List[float] = [0],
        store_buffer: Optional[int] = 2000,
    ):
        if sequencer.ndims != ndims:
            raise ValueError("sequencer ndims does not match ndims")
        self.sequencer, self.ndims, self.device_cfg = sequencer, ndims, device_cfg
        self.up_bounds = device_cfg.to_device(up_bounds)
        self.low_bounds = device_cfg.to_device(low_bounds)
        self.range_b = self.up_bounds - self.low_bounds
        self.proj_mat = torch.sqrt(torch.tensor(2.0, **device_cfg.as_torch_dict()))
        self.i_mat = torch.eye(ndims, **device_cfg.as_torch_dict())
        self._store_buffer = store_buffer
        self.fixed_samples = store_buffer is not None
        self._sample_buffer = None if store_buffer is None else device_cfg.to_device(
            sequencer.random(store_buffer)
        )
        self._index_buffer: Optional[torch.Tensor] = None
        self._int_gen = torch.Generator(device=device_cfg.device).manual_seed(sequencer.seed)
        self._initial_state = self._int_gen.get_state().clone()

    def reset(self):
        self.sequencer.reset()
        self._int_gen.set_state(self._initial_state)

    def fast_forward(self, steps: int):
        if self.fixed_samples:
            raise ValueError("fast forward will not work with fixed samples")
        self.sequencer.fast_forward(steps)

    def _get_samples(self, num_samples: int):
        if self._sample_buffer is None:
            return self.device_cfg.to_device(self.sequencer.random(num_samples))
        if self._index_buffer is None or self._index_buffer.shape != (num_samples,):
            self._index_buffer = torch.empty(
                (num_samples,), device=self.device_cfg.device, dtype=torch.int64
            )
        idx = torch.randint(
            self._sample_buffer.shape[0], (num_samples,),
            generator=self._int_gen, device=self.device_cfg.device,
            out=self._index_buffer,
        )
        return self._sample_buffer[idx]

    def get_samples(self, num_samples: int, bounded: bool = False) -> torch.Tensor:
        samples = self._get_samples(num_samples)
        return self.bound_samples(samples, self.range_b, self.low_bounds) if bounded else samples

    def get_gaussian_samples(self, num_samples: int, variance: float = 1.0) -> torch.Tensor:
        return self.gaussian_transform(
            self.get_samples(num_samples), self.proj_mat, self.i_mat, float(np.sqrt(variance))
        )

    @staticmethod
    def bound_samples(samples: torch.Tensor, range_b: torch.Tensor, low_bounds: torch.Tensor):
        return samples * range_b + low_bounds

    @staticmethod
    def gaussian_transform(
        uniform_samples: torch.Tensor,
        proj_mat: torch.Tensor,
        i_mat: torch.Tensor,
        std_dev: float,
    ):
        return (proj_mat * torch.erfinv(1.98 * uniform_samples - 0.99)) @ (i_mat * std_dev)

    @staticmethod
    def sample_by_random_index(
        sample_buffer: torch.Tensor,
        num_samples: int,
        int_generator: torch.Generator,
        device: torch.device,
        out_buffer: Optional[torch.Tensor],
    ):
        idx = torch.randint(
            sample_buffer.shape[0], (num_samples,), generator=int_generator,
            device=device, out=out_buffer,
        )
        return sample_buffer[idx], idx

    @classmethod
    def create_halton_sample_buffer(
        cls, ndims: int, up_bounds: List[float], low_bounds: List[float],
        store_buffer: Optional[int] = 2000, seed: int = 123,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        from .sequencer_halton import HaltonSequencer
        return cls(HaltonSequencer(ndims, seed), ndims, device_cfg, up_bounds, low_bounds, store_buffer)

    @classmethod
    def create_random_sample_buffer(
        cls, ndims: int, up_bounds: List[float], low_bounds: List[float],
        store_buffer: Optional[int] = 2000, seed: int = 123,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        from .sequencer_random import RandomSequencer
        return cls(RandomSequencer(ndims, seed), ndims, device_cfg, up_bounds, low_bounds, store_buffer)

    @classmethod
    def create_roberts_sample_buffer(
        cls, ndims: int, up_bounds: List[float], low_bounds: List[float],
        store_buffer: Optional[int] = 2000, seed: int = 123,
        device_cfg: DeviceCfg = DeviceCfg(),
    ):
        from .sequencer_roberts import RobertsSequencer
        return cls(RobertsSequencer(ndims, seed), ndims, device_cfg, up_bounds, low_bounds, store_buffer)
