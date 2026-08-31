"""Portable configuration I/O compatible with pinned cuRoboV2 helpers.

The original module is intentionally small: it reads YAML files and leaves
already-constructed configuration objects alone.  The portable backend keeps
those semantics and adds two explicit opt-in helpers for the configuration
boundary that applications commonly need on non-CUDA machines:

``resolve_device_cfg`` converts a serializable CPU/MPS device description to
``DeviceCfg``; ``resolve_dataclass`` creates a declared dataclass from a YAML
mapping.  Neither helper imports CUDA, Warp, Isaac Sim, or OpenUSD.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Union

import torch
import yaml
from yaml import CLoader as Loader

from curobo._src.types.device_cfg import DeviceCfg


# Preserve the pinned float resolver: PyYAML otherwise treats values such as
# ``1e-3`` as strings with some loader versions.
Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


ConfigT = TypeVar("ConfigT")
_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz", ".isaac"})
_DEVICE_CFG_FIELDS = frozenset(field.name for field in dataclasses.fields(DeviceCfg))


class ConfigIOError(ValueError):
    """Raised when a portable configuration cannot be represented faithfully."""


def _portable_config_path(value: str | os.PathLike[str]) -> str:
    path = os.fspath(value)
    if Path(path).suffix.lower() in _USD_SUFFIXES:
        raise NotImplementedError(
            "USD/Isaac configuration loading is unavailable on the portable "
            "backend; provide an explicit YAML, XRDF, or URDF configuration instead"
        )
    return path


def join_path(path1: Union[str, Path], path2: Union[str, Path]) -> str:
    """Join paths with the standard absolute-suffix behavior used by cuRobo."""
    if isinstance(path1, Path):
        path1 = str(path1)
    if isinstance(path2, Path):
        path2 = str(path2)
    if not isinstance(path2, str):
        return path2
    return os.path.join(path1, path2)


def resolve_config(config: Union[str, ConfigT]) -> Union[Dict, ConfigT]:
    """Load a YAML path, or return a parsed/typed configuration unchanged.

    ``Path`` values are accepted in addition to the pinned ``str`` API.  A
    typed dataclass is intentionally returned by identity; use
    :func:`resolve_dataclass` when converting a mapping into that type is
    desired.
    """
    if isinstance(config, (str, os.PathLike)):
        with open(_portable_config_path(config), encoding="utf-8") as file_p:
            return yaml.load(file_p, Loader=Loader)
    return config


def load_yaml(file_path: Union[str, Dict]) -> Dict:
    """Load a YAML path or return an already parsed mapping unchanged."""
    return resolve_config(file_path)


def _dtype_from_config(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = value.removeprefix("torch.")
        candidate = getattr(torch, name, None)
        if isinstance(candidate, torch.dtype):
            return candidate
    raise ConfigIOError(f"unsupported torch dtype {value!r} in device configuration")


def resolve_device_cfg(
    config: DeviceCfg | str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    default: DeviceCfg | None = None,
) -> DeviceCfg:
    """Resolve a serializable CPU/MPS device description into :class:`DeviceCfg`.

    Accepted mappings use the ``DeviceCfg`` field names and may spell dtypes as
    ``"float32"`` or ``"torch.float32"``.  CUDA is rejected here rather than
    yielding a device object which will later fail or accidentally take a CPU
    fallback path.  Constructing an MPS configuration is allowed even on a
    CPU-only host so a configuration can be serialized or inspected there.
    """
    if config is None:
        return DeviceCfg() if default is None else default
    if isinstance(config, DeviceCfg):
        result = config
    elif isinstance(config, (str, os.PathLike, torch.device)):
        result = DeviceCfg(device=torch.device(config))
    elif isinstance(config, Mapping):
        unknown = set(config).difference(_DEVICE_CFG_FIELDS)
        if unknown:
            raise ConfigIOError(
                "unknown DeviceCfg field(s): " + ", ".join(sorted(map(str, unknown)))
            )
        values = dict(config)
        for name in _DEVICE_CFG_FIELDS.difference({"device"}):
            if name in values:
                values[name] = _dtype_from_config(values[name])
        if "device" in values:
            values["device"] = torch.device(values["device"])
        result = DeviceCfg(**values)
    else:
        raise TypeError(
            "device configuration must be DeviceCfg, a device string, a mapping, or None"
        )
    if result.device.type == "cuda":
        raise NotImplementedError(
            "CUDA device configuration is unavailable on the portable Metal backend; "
            "use 'mps' or 'cpu'"
        )
    return result


def resolve_dataclass(
    config_type: type[ConfigT],
    config: ConfigT | Mapping[str, Any],
    *,
    device_cfg: DeviceCfg | str | Mapping[str, Any] | None = None,
) -> ConfigT:
    """Instantiate a declared dataclass from a mapping without hidden defaults.

    Unknown mapping keys are rejected so YAML typos cannot quietly change a
    planner configuration.  ``device_cfg`` entries are converted through
    :func:`resolve_device_cfg`; an explicit keyword override is applied only
    to dataclasses that actually expose that field.
    """
    if not dataclasses.is_dataclass(config_type):
        raise TypeError("config_type must be a dataclass type")
    fields = {field.name: field for field in dataclasses.fields(config_type)}
    if isinstance(config, config_type):
        result = config
    elif isinstance(config, Mapping):
        unknown = set(config).difference(fields)
        if unknown:
            raise ConfigIOError(
                f"unknown {config_type.__name__} field(s): "
                + ", ".join(sorted(map(str, unknown)))
            )
        values = dict(config)
        if "device_cfg" in values:
            values["device_cfg"] = resolve_device_cfg(values["device_cfg"])
        result = config_type(**values)
    else:
        raise TypeError(
            f"{config_type.__name__} configuration must be an instance or mapping"
        )
    if device_cfg is not None:
        if "device_cfg" not in fields:
            raise ConfigIOError(f"{config_type.__name__} does not define a device_cfg field")
        result = dataclasses.replace(result, device_cfg=resolve_device_cfg(device_cfg))
    return result


def _yaml_value(value: Any) -> Any:
    """Turn portable value records into PyYAML's standard data vocabulary."""
    if isinstance(value, DeviceCfg):
        return {
            name: str(getattr(value, name))
            for name in ("device", "dtype", "collision_geometry_dtype", "collision_gradient_dtype", "collision_distance_dtype")
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _yaml_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _yaml_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_yaml_value(item) for item in value]
    if isinstance(value, list):
        return [_yaml_value(item) for item in value]
    return value


def write_yaml(data: Dict, file_path: str):
    """Write mappings and portable dataclass records as UTF-8 YAML."""
    with open(_portable_config_path(file_path), "w", encoding="utf-8") as file:
        yaml.dump(_yaml_value(data), file)


def copy_file_to_path(source_file: str, destination_path: str) -> str:
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    _, file_name = os.path.split(source_file)
    new_path = join_path(destination_path, file_name)
    if not os.path.exists(new_path):
        shutil.copyfile(source_file, new_path)
    return new_path


def get_filename(file_path: str, remove_extension: bool = False) -> str:
    _, file_name = os.path.split(file_path)
    if remove_extension:
        file_name = os.path.splitext(file_name)[0]
    return file_name


def get_path_of_dir(file_path: str) -> str:
    dir_path, _ = os.path.split(file_path)
    return dir_path


def get_files_from_dir(dir_path, extension: List[str], contains: str) -> List[str]:
    file_names = [
        fn
        for fn in os.listdir(dir_path)
        if any(fn.endswith(ext) for ext in extension) and contains in fn
    ]
    file_names.sort()
    return file_names


def file_exists(path: str) -> bool:
    return path is not None and os.path.exists(path)


def merge_dict_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in a.items():
        if isinstance(value, dict):
            merge_dict_a_into_b(value, b[key])
        else:
            b[key] = value
    return b


def is_platform_windows() -> bool:
    return sys.platform == "win32"


def is_platform_linux() -> bool:
    return sys.platform == "linux"


def is_file_xrdf(file_path: str) -> bool:
    return file_path.endswith(".xrdf") or file_path.endswith(".XRDF")


def create_dir_if_not_exists(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


__all__ = [
    "ConfigIOError", "ConfigT", "copy_file_to_path", "create_dir_if_not_exists",
    "file_exists", "get_filename", "get_files_from_dir", "get_path_of_dir",
    "is_file_xrdf", "is_platform_linux", "is_platform_windows", "join_path",
    "load_yaml", "merge_dict_a_into_b", "resolve_config", "resolve_dataclass",
    "resolve_device_cfg", "write_yaml",
]
