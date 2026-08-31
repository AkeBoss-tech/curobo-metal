"""End-to-end smoke coverage for the CUDA-free getting-started equivalents."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples" / "getting_started_portable.py"


@pytest.mark.parametrize(
    "device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
)
def test_portable_getting_started_flows(device: str) -> None:
    """The five ordinary tutorial workflows execute without CUDA fallback."""
    environment = {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--device", device, "--json"],
        cwd=ROOT,
        env={**__import__("os").environ, **environment},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["device"] == device
    assert evidence["forward_kinematics"]["dof"] == 7
    assert evidence["forward_kinematics"]["sphere_count"] == 65
    assert evidence["forward_kinematics"]["gradient_norm"] > 0
    assert evidence["inverse_kinematics"]["success"]
    assert evidence["inverse_kinematics"]["solution_device"] == device
    assert evidence["motion_planning"]["success"]
    assert evidence["motion_planning"]["solution_device"] == device
    assert evidence["motion_planning"]["waypoints"] > 1
    assert evidence["robot_builder"]["sphere_count"] == 1
    assert evidence["robot_builder"]["reloaded_sphere_count"] == 1
    assert evidence["volumetric_mapping"]["stamped_voxels"] > 0
    assert evidence["volumetric_mapping"]["esdf_device"] == device
