"""Behavioral coverage for the portable public configuration I/O boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from curobo.config_io import (
    ConfigIOError,
    load_yaml,
    resolve_config,
    resolve_dataclass,
    resolve_device_cfg,
    write_yaml,
)
from curobo.types import DeviceCfg


@dataclass
class _PlannerCfg:
    name: str
    steps: int = 12
    device_cfg: DeviceCfg = DeviceCfg()


@dataclass
class _NoDeviceCfg:
    name: str = "portable"


def test_yaml_path_and_typed_records_preserve_portable_values(tmp_path: Path) -> None:
    path = tmp_path / "portable.yml"
    value = _PlannerCfg("demo", steps=4, device_cfg=DeviceCfg("cpu", torch.float64))
    write_yaml({"planner": value, "epsilon": 1e-3}, path)
    loaded = load_yaml(path)
    assert loaded["planner"]["name"] == "demo"
    assert loaded["planner"]["device_cfg"]["device"] == "cpu"
    assert loaded["planner"]["device_cfg"]["dtype"] == "torch.float64"
    assert loaded["epsilon"] == pytest.approx(1e-3)
    assert resolve_config(value) is value


def test_dataclass_and_device_config_coercion_is_explicit() -> None:
    config = resolve_dataclass(
        _PlannerCfg,
        {"name": "portable", "steps": 6, "device_cfg": {"device": "cpu", "dtype": "float64"}},
    )
    assert config.name == "portable"
    assert config.device_cfg.device.type == "cpu"
    assert config.device_cfg.dtype is torch.float64

    mps = resolve_device_cfg({"device": "mps", "dtype": "torch.float32"})
    assert mps.device == torch.device("mps")
    assert mps.dtype is torch.float32
    overridden = resolve_dataclass(_PlannerCfg, config, device_cfg="cpu")
    assert overridden.device_cfg.device.type == "cpu"


def test_config_io_rejects_unrepresentable_or_typoed_input(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="CUDA"):
        resolve_device_cfg("cuda:0")
    with pytest.raises(NotImplementedError, match="USD/Isaac"):
        resolve_config(tmp_path / "world.usda")
    with pytest.raises(NotImplementedError, match="USD/Isaac"):
        write_yaml({"world": "x"}, tmp_path / "world.usdc")
    with pytest.raises(ConfigIOError, match="unknown DeviceCfg"):
        resolve_device_cfg({"device": "cpu", "typo": True})
    with pytest.raises(ConfigIOError, match="unknown _PlannerCfg"):
        resolve_dataclass(_PlannerCfg, {"name": "x", "bogus": 1})
    with pytest.raises(ConfigIOError, match="does not define a device_cfg"):
        resolve_dataclass(_NoDeviceCfg, {}, device_cfg="cpu")
