#!/usr/bin/env python3
"""Validate a CUDA host and print machine-readable replay provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cuda_runtime import collect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.upstream), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
