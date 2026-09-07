"""Regression coverage for numeric platform boundaries.

These checks keep the portable contract explicit: CPU supports float64 and
Metal uses float32 for tensor-backed operations.  The pinned upstream cases
that ask MPS for float64 are classified as mechanism-only exclusions in the
gauntlet policy; this file verifies that the runtime boundary remains honest.
"""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.particle.sample_strategies.particle_sampler_cfg import (
    ParticleSamplerCfg,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg


def test_cpu_float64_device_cfg_remains_supported() -> None:
    cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float64)
    assert cfg.dtype is torch.float64
    tensor = cfg.to_device([1.0, 2.0])
    assert tensor.device.type == "cpu"
    assert tensor.dtype is torch.float64


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_float64_device_cfg_is_rejected_at_boundary() -> None:
    with pytest.raises(TypeError, match="MPS.*float32"):
        DeviceCfg(device=torch.device("mps"), dtype=torch.float64)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_particle_sampler_uses_supported_float32_configuration() -> None:
    cfg = ParticleSamplerCfg(
        device_cfg=DeviceCfg(device=torch.device("mps"), dtype=torch.float32)
    )
    assert cfg.device_cfg.device.type == "mps"
    assert cfg.device_cfg.dtype is torch.float32


def test_cpu_camera_filter_preserves_float64_dtype() -> None:
    depth = torch.tensor([[0.005, 0.02], [0.03, 0.015]], dtype=torch.float64)
    camera = CameraObservation(depth_image=depth)
    camera.filter_depth(distance=0.01)
    assert camera.depth_image is not None
    assert camera.depth_image.dtype is torch.float64
    assert camera.depth_image.device.type == "cpu"
