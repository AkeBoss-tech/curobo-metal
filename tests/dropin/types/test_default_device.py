"""Automatic allocation policy without redirecting explicit requests."""

import pytest
import torch

from curobo.types import DeviceCfg
from curobo_metal.types.device import DeviceCfg as BackendDeviceCfg
from curobo_metal.backend import resolve_device
from curobo._src.util.config_io import resolve_device_cfg


@pytest.mark.parametrize("available", [False, True])
def test_defaults_follow_availability_at_construction(monkeypatch, available):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)
    expected = torch.device("mps:0" if available else "cpu")
    assert resolve_device() == expected
    for cls in (DeviceCfg, BackendDeviceCfg):
        cfg = cls()
        assert cfg.device == expected
        assert cfg.clone() == cfg
        assert cfg.cpu().device.type == "cpu"
        assert cls("cpu", torch.float64).dtype == torch.float64
        assert cls("mps").device.type == "mps"  # Serializable on CPU hosts too.
    assert resolve_device_cfg().device == expected
    assert resolve_device_cfg({}).device == expected


def test_unavailable_explicit_device_is_never_redirected(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("mps")
    with pytest.raises(ValueError, match="supports only"):
        resolve_device("cuda")
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.parametrize("cls", [DeviceCfg, BackendDeviceCfg])
def test_automatic_mps_rejects_float64_without_cpu_fallback(monkeypatch, cls):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(TypeError, match="float32"):
        cls(dtype=torch.float64)


def test_explicit_device_roundtrip():
    cfg = DeviceCfg("cpu", torch.float64)
    restored = resolve_device_cfg({"device": str(cfg.device), "dtype": str(cfg.dtype)})
    assert restored == cfg
    assert restored.to_device([1.0]).device.type == "cpu"


@pytest.mark.parametrize("cls", [DeviceCfg, BackendDeviceCfg])
def test_default_device_matches_allocated_tensor(cls):
    cfg = cls()
    assert cfg.to_device([1.0]).device == cfg.device
