"""Standalone CPU/MPS start-to-goal motion-generation example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from curobo_metal.ops.kinematics import KinematicChain
from curobo_metal.ops.trajectory import TrajectoryProblem, generate_motion, trajectory_metrics
from curobo_metal.reference import SerialRobot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    case = json.loads((root / "tests/fixtures/trajectory/two_link_obstacle_free.json").read_text())
    inputs = case["inputs"]
    device = torch.device(args.device)
    tensor = lambda value: torch.tensor(value, dtype=torch.float32, device=device)
    problem = TrajectoryProblem(
        KinematicChain(SerialRobot.from_dict(case["robot"]), device=device),
        tensor(inputs["start"]), tensor(inputs["goal"]), tensor(inputs["lower"]),
        tensor(inputs["upper"]), inputs["steps"], inputs["dt"], max_iterations=100,
    )
    result = generate_motion(problem)
    if result.trajectory is None:
        raise RuntimeError(result.status)
    metrics = trajectory_metrics(problem, result.trajectory.trajectories)
    print(f"status={result.status} selected_seed={result.trajectory.selected_seed}")
    print(f"endpoint_error={result.trajectory.endpoint_error.tolist()}")
    print(f"duration={metrics.duration:.3f}s max_velocity={metrics.maximum_velocity.tolist()}")
    print(result.trajectory.trajectories.detach().cpu())


if __name__ == "__main__":
    main()
