"""Compatibility loader for lists of world configuration mappings."""

from pathlib import Path

from curobo.util_file import load_yaml


def get_world_config_dataloader(source):
    """Yield world dictionaries from a YAML file, directory, or iterable."""
    if isinstance(source, (list, tuple)):
        yield from source
        return
    path = Path(source)
    paths = sorted(path.glob("*.yml")) if path.is_dir() else [path]
    for item in paths:
        yield load_yaml(str(item))


__all__ = ["get_world_config_dataloader"]
