"""Compile/warmup/steady-state portable optimizer benchmark."""

from __future__ import annotations

import argparse
import json
import time

import torch

from curobo_metal.optim import (
    ExecutionCache, LBFGSConfig, ParticleConfig, lbfgs_optimize, particle_optimize,
)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--compile", action="store_true", help="torch.compile the objective")
    args = parser.parse_args()
    device = torch.device(args.device)
    x = torch.linspace(-2, 2, args.batch * args.width, device=device).reshape(
        args.batch, args.width
    )
    eager_objective = lambda value: value.square().sum(-1)
    objective = torch.compile(eager_objective) if args.compile else eager_objective
    cache = ExecutionCache()
    cases = {
        "lbfgs": lambda: lbfgs_optimize(
            objective, x, config=LBFGSConfig(iterations=20), cache=cache,
        ),
        "particle": lambda: particle_optimize(
            objective, x,
            config=ParticleConfig(iterations=20, particles=32, elite_count=8, seed=0),
            cache=cache,
        ),
    }
    output = {
        "device": device.type, "batch": args.batch, "width": args.width,
        "torch_compile": args.compile, "cases": {},
    }
    for name, run in cases.items():
        started = time.perf_counter(); run(); synchronize(device)
        compile_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter(); run(); synchronize(device)
        warmup_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        for _ in range(args.repeat):
            run()
        synchronize(device)
        steady_ms = (time.perf_counter() - started) * 1000 / args.repeat
        output["cases"][name] = {
            "compile_ms": compile_ms, "warmup_ms": warmup_ms, "steady_ms": steady_ms,
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
