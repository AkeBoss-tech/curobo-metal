"""Reproducible eager CPU/MPS benchmarks for production world-collision ops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from curobo_metal.ops.world_collision import Mesh, VoxelGrid, mesh_distance, query_esdf


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def measure(fn, device: torch.device, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1e3 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable")
    dtype = torch.float32 if args.device == "mps" else torch.float64
    torch.manual_seed(7)
    points = (torch.rand((4, 256, 3), device=device, dtype=dtype) - .5).requires_grad_(True)
    axes = torch.linspace(-1, 1, 32, device=device, dtype=dtype)
    x, y, z = torch.meshgrid(axes, axes, axes, indexing="ij")
    values = torch.sqrt(x.square() + y.square() + z.square()) - .4
    grid = VoxelGrid(values, 2 / 31, torch.zeros(3, device=device, dtype=dtype),
                     torch.eye(3, device=device, dtype=dtype), 10)
    vertices = torch.tensor(
        [[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]], device=device, dtype=dtype)
    faces = torch.tensor([[0,1,2], [0,2,3]], device=device)
    mesh = Mesh(vertices, faces, False)
    translation = torch.zeros((1, 1, 3), device=device, dtype=dtype)
    rotation = torch.eye(3, device=device, dtype=dtype)[None, None]

    cases = {
        "esdf_4x256_32cube_forward_ms": lambda: query_esdf(points, [[grid]]),
        "mesh_4x256_2tri_forward_ms": lambda: mesh_distance(
            points, [mesh], translation, rotation, signed=False),
    }
    result = {
        "device": args.device,
        "dtype": str(dtype).removeprefix("torch."),
        "iterations": args.iterations,
        "fallback_disabled": True,
        "measurements": {
            name: measure(fn, device, 5, args.iterations) for name, fn in cases.items()
        },
        "implementation": "portable-pytorch",
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
