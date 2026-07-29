"""Regenerate checked-in FK replay cases using the independent CPU oracle."""

from __future__ import annotations

import json
from pathlib import Path

from curobo_metal.reference import SerialRobot, forward_kinematics, save_case

HERE = Path(__file__).parent


def generate(path: Path) -> None:
    source = json.loads(path.read_text(encoding="utf-8"))
    robot = SerialRobot.from_dict(source["robot"])
    result = forward_kinematics(robot, source["inputs"]["q"])
    source["expected"] = {
        "dtype": "float64",
        "link_names": list(result.link_names),
        "transforms": result.transforms.tolist(),
        "transform_jacobian": result.transform_jacobian.tolist(),
        "geometric_jacobian": result.geometric_jacobian.tolist(),
    }
    save_case(path, source)


if __name__ == "__main__":
    for fixture in (HERE / "two_link_planar.json", HERE / "panda_serial.json"):
        generate(fixture)

