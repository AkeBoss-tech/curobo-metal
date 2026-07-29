"""Standalone portable MotionGen example (run from the repository root)."""

from pathlib import Path

import torch

from curobo_metal.motion_gen import JointState, MotionGen, MotionGenConfig


def main() -> None:
    fixture = Path("tests/fixtures/trajectory/two_link_obstacle_detour.json")
    config = MotionGenConfig.load_from_robot_config(
        fixture, device="cpu", interpolation_dt=0.025
    )
    motion_gen = MotionGen(config)
    motion_gen.warmup()
    start = torch.tensor([-0.8, 0.0], dtype=config.dtype, device=config.device)
    goal = torch.tensor([0.8, 0.0], dtype=config.dtype, device=config.device)
    result = motion_gen.plan_single_js(JointState(start), JointState(goal))
    print(f"success={bool(result.success)} status={result.status.value}")
    if result.interpolated_plan is not None:
        print(
            f"waypoints={result.interpolated_plan.position.shape[-2]} "
            f"dt={result.interpolation_dt}"
        )


if __name__ == "__main__":
    main()
