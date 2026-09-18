"""Plan a deterministic obstacle-avoiding path on CPU or Apple MPS."""

from __future__ import annotations

import argparse

import torch

from curobo_metal.ops.graph_planning import (
    GraphPlanningProblem,
    PersistentRoadmap,
    paths_to_trajectory_seeds,
)


def plan_example(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a collision-free path and a fixed-knot trajectory seed."""
    dtype = torch.float32 if device.type == "mps" else torch.float64

    def outside_obstacle(points: torch.Tensor) -> torch.Tensor:
        # A vertical joint-space obstacle blocks the straight-line path.
        inside = (
            (points[:, 0] >= -0.3)
            & (points[:, 0] <= 0.3)
            & (points[:, 1] >= -1.1)
            & (points[:, 1] <= 1.1)
        )
        return ~inside

    tensor = lambda values: torch.tensor(values, dtype=dtype, device=device)
    problem = GraphPlanningProblem(
        starts=tensor([[-1.5, 0.0]]),
        goals=tensor([[1.5, 0.0]]),
        lower=tensor([-2.0, -2.0]),
        upper=tensor([2.0, 2.0]),
        validity=outside_obstacle,
        sample_count=256,
        seed=11,
        k_neighbors=20,
        edge_step=0.04,
        interpolation_step=0.05,
    )

    result = PersistentRoadmap().plan(problem)
    if not bool(result.success.all().item()):
        raise RuntimeError(f"graph planning failed: {result.status}")
    return result.paths[0], paths_to_trajectory_seeds(result, steps=32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="execution device (default: MPS when available)",
    )
    args = parser.parse_args()
    device_name = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device_name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available in this PyTorch build")

    path, seeds = plan_example(torch.device(device_name))
    print(f"device={device_name}")
    print(f"path_points={path.shape[0]}")
    print(f"trajectory_seed_shape={tuple(seeds.shape)}")


if __name__ == "__main__":
    main()
