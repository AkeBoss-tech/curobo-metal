"""Pose calibration reference entry point (requires an interactive viewer)."""
from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np


def require_viewer_backend() -> None:
    raise RuntimeError("pose calibration requires the optional viser viewer and robot mesh assets")


def compute_pose_error(estimated: Sequence[float], ground_truth: Sequence[float]) -> tuple[float, float]:
    """Return translation (mm) and quaternion angular (degrees) error."""
    a, b = np.asarray(estimated, dtype=float), np.asarray(ground_truth, dtype=float)
    if a.shape != (7,) or b.shape != (7,):
        raise ValueError("poses must be [x, y, z, w, qx, qy, qz]")
    translation = float(np.linalg.norm(a[:3] - b[:3]) * 1000)
    dot = float(np.clip(abs(np.dot(a[3:], b[3:])), -1.0, 1.0))
    return translation, float(2 * np.degrees(np.arccos(dot)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--port", type=int, default=8080); parser.parse_args(argv)
    require_viewer_backend(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
