"""Versioned JSON replay cases for cross-device differential testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .forward_kinematics import SerialRobot

FORMAT = "curobo-metal-fk-case"
VERSION = 1


def load_case(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        case = json.load(stream)
    if case.get("format") != FORMAT or case.get("version") != VERSION:
        raise ValueError("unsupported forward-kinematics replay format")
    robot = SerialRobot.from_dict(case["robot"])
    q = np.asarray(case["inputs"]["q"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != robot.dof:
        raise ValueError("serialized q does not match robot DOF")
    expected = case.get("expected", {})
    shapes = {
        "transforms": (q.shape[0], len(robot.joints), 4, 4),
        "transform_jacobian": (q.shape[0], len(robot.joints), 4, 4, robot.dof),
        "geometric_jacobian": (q.shape[0], len(robot.joints), 6, robot.dof),
    }
    for key, shape in shapes.items():
        if key not in expected:
            continue
        array = np.asarray(expected[key], dtype=np.float64)
        if array.shape != shape:
            raise ValueError(f"serialized expected.{key} has shape {array.shape}, expected {shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"serialized expected.{key} contains a non-finite value")
    if "link_names" in expected and expected["link_names"] != [joint.name for joint in robot.joints]:
        raise ValueError("serialized expected.link_names does not match robot joints")
    return case


def save_case(path: str | Path, case: Mapping[str, Any]) -> None:
    """Write canonical JSON (stable keys and separators) after validating it."""
    destination = Path(path)
    text = json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    destination.write_text(text, encoding="utf-8")
    load_case(destination)
