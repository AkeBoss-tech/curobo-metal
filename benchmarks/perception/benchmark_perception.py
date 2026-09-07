"""Microbenchmark for dense depth fusion and ESDF query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from curobo_metal.ops.perception import CameraObservation, PerceptionConfig, PerceptionMapper


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def run(device_name: str, iterations: int) -> dict:
    device = torch.device(device_name)
    cfg = PerceptionConfig((8, 8, 8), 0.05, grid_center=(0, 0, .4),
                           truncation_distance=.1, environments=1)
    k = torch.tensor([[8., 0., 7.5], [0., 8., 7.5], [0., 0., 1.]],
                     device=device)
    obs = CameraObservation(torch.full((16, 16), .5, device=device), k,
                            torch.eye(4, device=device))
    points = torch.zeros((256, 3), device=device)
    points[:, 2] = .4
    for _ in range(2):
        mapper = PerceptionMapper(cfg, device=device)
        mapper.update(obs)
        mapper.query(points)
    synchronize(device)
    update, query = [], []
    for _ in range(iterations):
        mapper = PerceptionMapper(cfg, device=device)
        start = time.perf_counter()
        mapper.update(obs)
        synchronize(device)
        update.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        mapper.query(points)
        synchronize(device)
        query.append((time.perf_counter() - start) * 1000)
    return {
        "device": device_name,
        "dtype": "float32",
        "iterations": iterations,
        "shape": list(cfg.shape),
        "cameras": 1,
        "query_points": len(points),
        "update_ms_median": sorted(update)[len(update) // 2],
        "query_ms_median": sorted(query)[len(query) // 2],
        "fallback_disabled": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    encoded = json.dumps(run(args.device, args.iterations), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
