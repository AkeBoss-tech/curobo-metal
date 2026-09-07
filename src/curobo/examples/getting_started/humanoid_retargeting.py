"""Portable entry point for humanoid motion retargeting.

BVH/SOMA and visualization integrations are optional.  The public helper below
validates the common pose-sequence contract before handing work to cuRobo.
"""
from __future__ import annotations

import argparse
from typing import Mapping, Sequence

import numpy as np


def validate_pose_sequence(poses: Mapping[str, np.ndarray], *, frames: int | None = None) -> int:
    """Validate named ``(frames, 7)`` position/quaternion pose arrays."""
    if not poses:
        raise ValueError("at least one tracked link is required")
    lengths = set()
    for link, values in poses.items():
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != 7:
            raise ValueError(f"pose sequence for {link!r} must have shape (frames, 7)")
        lengths.add(array.shape[0])
    if len(lengths) != 1 or (frames is not None and lengths != {frames}):
        raise ValueError("all tracked links must contain the same number of frames")
    return lengths.pop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="robot retargeting YAML (optional)")
    parser.parse_args(argv)
    print("Retargeting adapter loaded; install SOMA/BVH dependencies to solve a motion clip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
