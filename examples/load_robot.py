"""Load a cuRobo-style YAML/URDF and compile portable robot models."""

from __future__ import annotations

import argparse

from curobo_metal.config import RobotCfg
from curobo_metal.types import DeviceCfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot", help="cuRobo YAML or URDF path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    args = parser.parse_args()

    robot = RobotCfg.create(args.robot, device_cfg=DeviceCfg(args.device))
    whole_body = robot.to_whole_body_model()
    spheres, sphere_links = robot.to_collision_inputs()
    print(f"robot={robot.name} dof={whole_body.dof} links={whole_body.link_count}")
    print(f"joints={robot.joint_names}")
    print(f"tools={robot.tool_frames}")
    print(f"collision_spheres={len(spheres)} link_indices={sphere_links.tolist()}")


if __name__ == "__main__":
    main()
