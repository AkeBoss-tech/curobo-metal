"""Build a tiny custom objective around a cuRobo-style tensor state."""
from __future__ import annotations

import argparse
from typing import Sequence

import torch


def quadratic_cost(position: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a differentiable squared-distance cost for batched positions."""
    if position.shape != target.shape:
        raise ValueError("position and target must have the same shape")
    return (position - target).square().sum(dim=-1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    position = torch.zeros(1, 3, requires_grad=True)
    target = torch.ones_like(position)
    optimizer = torch.optim.SGD([position], lr=0.2)
    for _ in range(args.steps):
        optimizer.zero_grad(); quadratic_cost(position, target).sum().backward(); optimizer.step()
    print(f"final position: {position.detach().flatten().tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
