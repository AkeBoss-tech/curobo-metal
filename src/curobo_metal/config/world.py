"""Small primitive world configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class Cuboid:
    name: str
    pose: list[float]
    dims: list[float]


@dataclass
class Sphere:
    name: str
    pose: list[float]
    radius: float


@dataclass
class WorldConfig:
    cuboid: list[Cuboid] = field(default_factory=list)
    sphere: list[Sphere] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorldConfig":
        raw = value.get("world_cfg", value)
        unsupported = set(raw) & {"mesh", "voxel", "voxel_grid", "esdf"}
        if unsupported:
            from .robot import UnsupportedConfigError
            raise UnsupportedConfigError(
                f"portable world loading supports cuboid/sphere only, not {sorted(unsupported)}"
            )
        return cls(
            [Cuboid(str(row.get("name", f"cuboid_{i}")), list(row["pose"]), list(row["dims"]))
             for i, row in enumerate(raw.get("cuboid", []))],
            [Sphere(str(row.get("name", f"sphere_{i}")), list(row["pose"]), float(row["radius"]))
             for i, row in enumerate(raw.get("sphere", []))],
        )
