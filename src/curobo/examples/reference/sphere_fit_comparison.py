"""Compare collision-sphere fitting strategies when mesh dependencies are installed."""
from __future__ import annotations

import argparse
from typing import Sequence


def require_mesh_backend() -> None:
    raise RuntimeError("sphere fitting requires optional trimesh/viser dependencies and robot mesh assets")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--port", type=int, default=8080); parser.parse_args(argv)
    require_mesh_backend(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
