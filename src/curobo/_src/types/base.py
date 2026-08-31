"""Deprecated pinned path; tensor configuration moved to ``device_cfg``."""

from .device_cfg import DeviceCfg
from curobo._src.util.logging import log_warn

TensorDeviceType = DeviceCfg

__all__ = ["DeviceCfg", "TensorDeviceType"]
