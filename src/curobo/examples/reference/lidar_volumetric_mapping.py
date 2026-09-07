"""LiDAR mapping reference adapter.

The sensor driver is intentionally not imported at module load time.
"""
from __future__ import annotations

import argparse
from typing import Sequence


def require_sensor_backend() -> None:
    raise RuntimeError("LiDAR reference requires a project-specific sensor driver; pass point clouds to curobo.perception.Mapper")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args(argv)
    require_sensor_backend(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
