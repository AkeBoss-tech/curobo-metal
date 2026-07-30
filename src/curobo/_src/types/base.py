"""Deprecated pinned path; tensor configuration moved to ``device_cfg``."""

from .device_cfg import DeviceCfg

TensorDeviceType = DeviceCfg

__all__ = ["DeviceCfg", "TensorDeviceType"]
