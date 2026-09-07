"""Live RGB-D mapping plus MPC reference adapter."""
from __future__ import annotations

import argparse
from typing import Sequence


def require_live_backend() -> None:
    raise RuntimeError("live mapping requires pyrealsense2 and a running camera; use volumetric_mapping for recorded data")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args(argv)
    require_live_backend(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
