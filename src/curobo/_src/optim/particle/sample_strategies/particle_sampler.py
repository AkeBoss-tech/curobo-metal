"""Unified portable particle samplers matching the pinned cuRobo V2 lifecycle."""

from __future__ import annotations

from typing import Any

import torch
import torch.autograd.profiler as profiler

from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.sampling.sequencer_halton import HaltonSequencer
from curobo._src.util.sampling.sequencer_random import RandomSequencer
from curobo._src.util.logging import log_and_raise

from .particle_sampler_cfg import ParticleSamplerCfg
from .processor_knot import KnotParticleProcessor
from .processor_standard import StandardParticleProcessor
from .processor_stomp import StompParticleProcessor


class ParticleSampler:
    """Generate device-resident Gaussian samples and apply a trajectory processor."""

    def __init__(
        self,
        sample_config: ParticleSamplerCfg,
        horizon: int,
        action_dim: int,
        generator: SampleBuffer,
        post_processor: Any,
    ):
        self.sample_config = sample_config
        self.sample_shape: int | list[int] = 0
        self.samples: torch.Tensor | None = None
        self.horizon = horizon
        self.action_dim = action_dim
        self.ndims = horizon * action_dim
        self.generator = generator
        self.post_processor = post_processor
        if post_processor.input_ndims != generator.ndims:
            raise ValueError(
                "input_ndims of post_processor "
                f"{post_processor.input_ndims} and generator {generator.ndims} do not match"
            )

    def reset_seed(self):
        self.generator.reset()

    def get_samples(
        self,
        sample_shape,
        base_seed=None,
        filter_smooth=False,
        **kwargs,
    ):
        del base_seed, kwargs
        if len(sample_shape) != 1:
            raise ValueError("sample shape should be a single value")
        if self.sample_shape != sample_shape or not self.sample_config.fixed_samples:
            self.sample_shape = sample_shape
            raw = self.generator.get_gaussian_samples(sample_shape[0]).view(
                sample_shape[0], self.post_processor.input_horizon, self.action_dim
            )
            self.samples = self.post_processor.process_samples(raw, filter_smooth)
        assert self.samples is not None
        if self.samples.shape[0] != sample_shape[0]:
            raise ValueError("sampling failed")
        return self.samples

    @classmethod
    def create_halton_particle_sampler(
        cls, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int
    ) -> "ParticleSampler":
        ndims = horizon * action_dim
        generator = SampleBuffer(
            HaltonSequencer(ndims=ndims, seed=sample_config.seed), ndims=ndims,
            device_cfg=sample_config.device_cfg, store_buffer=2000,
        )
        processor = StandardParticleProcessor(
            horizon=horizon, action_dim=action_dim, device_cfg=sample_config.device_cfg,
            filter_coeffs=sample_config.filter_coeffs,
        )
        return cls(sample_config, horizon, action_dim, generator, processor)

    @classmethod
    def create_random_particle_sampler(
        cls, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int
    ) -> "ParticleSampler":
        ndims = horizon * action_dim
        generator = SampleBuffer(
            RandomSequencer(ndims=ndims, seed=sample_config.seed), ndims=ndims,
            device_cfg=sample_config.device_cfg, store_buffer=None,
        )
        processor = StandardParticleProcessor(
            horizon=horizon, action_dim=action_dim, device_cfg=sample_config.device_cfg,
            filter_coeffs=sample_config.filter_coeffs,
        )
        return cls(sample_config, horizon, action_dim, generator, processor)

    @classmethod
    def create_knot_particle_sampler(
        cls, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int,
        sequencer_type: str = "halton",
    ) -> "ParticleSampler":
        ndims = sample_config.n_knots * action_dim
        if sequencer_type == "halton":
            sequencer = HaltonSequencer(ndims=ndims, seed=sample_config.seed)
            store_buffer = 2000
        elif sequencer_type == "random":
            sequencer = RandomSequencer(ndims=ndims, seed=sample_config.seed)
            store_buffer = None
        else:
            raise ValueError(f"Unknown knot sequencer type: {sequencer_type}")
        generator = SampleBuffer(
            sequencer, ndims=ndims, device_cfg=sample_config.device_cfg,
            store_buffer=store_buffer,
        )
        processor = KnotParticleProcessor(
            horizon=horizon, action_dim=action_dim, n_knots=sample_config.n_knots,
            degree=sample_config.degree, device_cfg=sample_config.device_cfg,
        )
        return cls(sample_config, horizon, action_dim, generator, processor)

    @classmethod
    def create_stomp_particle_sampler(
        cls, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int
    ) -> "ParticleSampler":
        ndims = horizon * action_dim
        generator = SampleBuffer(
            HaltonSequencer(ndims=ndims, seed=sample_config.seed), ndims=ndims,
            device_cfg=sample_config.device_cfg, store_buffer=2000,
        )
        processor = StompParticleProcessor(
            horizon=horizon, action_dim=action_dim, device_cfg=sample_config.device_cfg,
            stencil_type=sample_config.stencil_type,
        )
        return cls(sample_config, horizon, action_dim, generator, processor)


def create_particle_sampler(
    sample_type: str, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int,
    **kwargs,
) -> ParticleSampler:
    del kwargs
    if sample_type == "halton":
        return ParticleSampler.create_halton_particle_sampler(sample_config, horizon, action_dim)
    if sample_type == "random":
        return ParticleSampler.create_random_particle_sampler(sample_config, horizon, action_dim)
    if sample_type == "halton-knot":
        return ParticleSampler.create_knot_particle_sampler(sample_config, horizon, action_dim, "halton")
    if sample_type == "random-knot":
        return ParticleSampler.create_knot_particle_sampler(sample_config, horizon, action_dim, "random")
    if sample_type == "stomp":
        return ParticleSampler.create_stomp_particle_sampler(sample_config, horizon, action_dim)
    raise ValueError(f"Unknown sample type: {sample_type}")


class MixedParticleSampler:
    """Combine active particle strategies according to configured ratios."""

    def __init__(self, sample_config: ParticleSamplerCfg, horizon: int, action_dim: int):
        self.sample_config = sample_config
        self.horizon = horizon
        self.action_dim = action_dim
        self.particle_samplers: dict[str, ParticleSampler] = {}
        self.sample_fns: dict[str, Any] = {}
        self.samples: torch.Tensor | None = None
        self._last_sample_shape = 0
        for sample_type, ratio in sample_config.sample_ratio.items():
            if ratio > 0.0:
                sampler = create_particle_sampler(sample_type, sample_config, horizon, action_dim)
                self.particle_samplers[sample_type] = sampler
                self.sample_fns[sample_type] = sampler.get_samples

    def reset_seed(self):
        for sampler in self.particle_samplers.values():
            sampler.reset_seed()

    def get_samples(
        self, sample_shape, base_seed=None, **kwargs,
    ):
        if len(sample_shape) != 1:
            raise ValueError("sample shape should be a single value")
        count = sample_shape[0]
        if (
            not self.sample_config.fixed_samples
            or self.samples is None
            or count != self._last_sample_shape
        ):
            batches = []
            for sample_type, ratio in self.sample_config.sample_ratio.items():
                if ratio == 0.0 or sample_type not in self.sample_fns:
                    continue
                part_count = round(count * ratio)
                if part_count:
                    batches.append(self.sample_fns[sample_type](
                        [part_count], base_seed=base_seed, **kwargs
                    ))
            if batches:
                self.samples = torch.cat(batches, dim=0)
            else:
                self.samples = torch.zeros(
                    (count, self.horizon, self.action_dim),
                    dtype=self.sample_config.device_cfg.dtype,
                    device=self.sample_config.device_cfg.device,
                )
            self._last_sample_shape = count
        return self.samples


__all__ = ["ParticleSampler", "MixedParticleSampler", "create_particle_sampler"]
