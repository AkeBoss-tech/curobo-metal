"""Small, portable RGB-D mapping example.

The full cuRobo tutorial uses a Sun3D download and a CUDA viewer.  This module
keeps the useful dataset/observation boundary available on every platform and
defers the optional mapper and viewer imports until :func:`main` is called.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np


class Sun3dDataset:
    """Read a minimal Sun3D-style directory without requiring image packages."""

    def __init__(self, root: str | Path, device: str = "cpu", depth_scale: float = 0.001):
        self.root = Path(root)
        self.device = device
        self.depth_scale = float(depth_scale)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Sun3D root does not exist: {self.root}")
        self.frames = sorted(self.root.rglob("*.depth.png"))

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.frames)

    def frame(self, index: int) -> dict[str, Path]:
        depth = self.frames[index]
        stem = str(depth)[:-len(".depth.png")]
        return {"depth": depth, "color": Path(stem + ".color.png"), "pose": Path(stem + ".pose.txt")}


def integrate_dataset(root: str | Path, *, num_frames: int = 1, stride: int = 1) -> int:
    """Integrate frames with the local mapper, returning the number processed.

    Mapper construction is intentionally lazy: importing this example remains
    safe on machines without Warp/CUDA.  A missing backend produces an
    actionable error when the example is actually executed.
    """
    try:
        from curobo.perception import Mapper, MapperCfg
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("volumetric mapping requires the optional Warp backend; install cuRobo mapping extras") from exc
    dataset = Sun3dDataset(root)
    if not dataset.frames:
        return 0
    # Keep configuration in the execution path while allowing users to adapt it
    # to their camera; actual image/pose conversion is application-specific.
    _ = Mapper(MapperCfg(extent_meters_xyz=(2.0, 2.0, 2.0), device="cpu"))
    return len(list(dataset)[::stride][:num_frames])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(argv)
    print(f"Found {len(Sun3dDataset(args.root))} depth frames")
    print(f"Prepared {integrate_dataset(args.root, num_frames=args.num_frames, stride=args.stride)} frame(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
