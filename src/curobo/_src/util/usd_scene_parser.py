"""Lazy OpenUSD scene parser compatibility surface."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from curobo._src.geom.types import Capsule, Cuboid, Cylinder, Mesh, SceneCfg, Sphere
from curobo._src.types.pose import Pose
from curobo._src.util.usd_util import _usd, get_prim_world_pose


def _unsupported(*args, **kwargs):
    del args, kwargs
    _usd()
    raise NotImplementedError(
        "portable USD scene conversion is unavailable; use an explicit SceneCfg"
    )


def get_cylinder_attrs(prim, cache=None, transform=None) -> Cylinder:
    return _unsupported(prim, cache, transform)


def get_capsule_attrs(prim, cache=None, transform=None) -> Capsule:
    return _unsupported(prim, cache, transform)


def get_cube_attrs(prim, cache=None, transform=None) -> Cuboid:
    return _unsupported(prim, cache, transform)


def get_sphere_attrs(prim, cache=None, transform=None) -> Sphere:
    return _unsupported(prim, cache, transform)


def get_mesh_attrs(prim, cache=None, transform=None) -> Optional[Mesh]:
    return _unsupported(prim, cache, transform)


class UsdSceneParser:
    def __init__(self) -> None:
        self.stage = None

    def load_stage_from_file(self, file_path: str):
        _, Usd, _ = _usd()
        self.stage = Usd.Stage.Open(file_path)
        return self.stage

    def load_stage(self, stage: Usd.Stage):
        _usd()
        self.stage = stage
        return self

    def get_pose(
        self, prim_path: str, timecode: float = 0.0, inverse: bool = False
    ) -> np.matrix:
        return _unsupported(prim_path, timecode, inverse)

    def get_obstacles_from_stage(
        self,
        only_paths: Optional[List[str]] = None,
        ignore_paths: Optional[List[str]] = None,
        only_substring: Optional[List[str]] = None,
        ignore_substring: Optional[List[str]] = None,
        reference_prim_path: Optional[str] = None,
        timecode: float = 0,
    ) -> SceneCfg:
        return _unsupported(
            only_paths, ignore_paths, only_substring, ignore_substring,
            reference_prim_path, timecode,
        )


__all__ = [
    "UsdSceneParser", "get_cylinder_attrs", "get_capsule_attrs", "get_cube_attrs",
    "get_sphere_attrs", "get_mesh_attrs",
]
