"""Portable robot debugger inspection and artifact lifecycle coverage."""

import json

import pytest
import torch

from curobo._src.robot.builder.debugger_robot import RobotDebugger
from curobo._src.types.device_cfg import DeviceCfg


def test_debugger_reports_real_configured_self_collision_data_and_snapshots():
    debugger = RobotDebugger("franka.yml")
    stats = debugger.collision_matrix_stats()
    assert stats["total_spheres"] == 65
    assert 0 < stats["checked_pairs"] < stats["total_possible_pairs"]

    result = debugger.check_default_joint_configuration_collision()
    assert result["has_collision"] is result["collision"]
    assert result["num_spheres"] == stats["total_spheres"]
    assert result["num_checked_pairs"] == stats["checked_pairs"]
    assert debugger.last_result == result
    result["distances"]["local"] = 1.0
    assert "local" not in debugger.last_result["distances"]


def test_debugger_sampling_is_seeded_and_exports_json_safe_inspection_artifacts(tmp_path):
    debugger = RobotDebugger("franka.yml")
    first = debugger.sample_collision_checks(num_samples=6, batch_size=2, seed=17)
    second = debugger.sample_collision_checks(num_samples=6, batch_size=3, seed=17)
    assert first["collision_count"] == second["collision_count"]
    assert first["frequent_collisions"] == second["frequent_collisions"]
    assert all(count <= first["total_samples"] for _, count in first["frequent_collisions"])

    debugger.check_default_joint_configuration_collision()
    path = debugger.export_inspection_report(tmp_path / "debug.json")
    artifact = json.loads(path.read_text())
    assert artifact["matrix"]["checked_pairs"] == debugger.collision_matrix_stats()["checked_pairs"]
    assert artifact["check_count"] == 1
    assert artifact["last_result"]["num_spheres"] == 65


def test_debugger_rejects_invalid_joint_data_and_external_backends_explicitly():
    debugger = RobotDebugger("franka.yml")
    with pytest.raises(ValueError, match="elements"):
        debugger.check_collision_at_config([0.0])
    with pytest.raises(ValueError, match="finite"):
        debugger.check_collision_at_config(torch.full((7,), float("nan")))
    with pytest.raises(ValueError, match="positive integer"):
        debugger.sample_collision_checks(num_samples=0)
    with pytest.raises(FileNotFoundError):
        RobotDebugger.from_xrdf("robot.xrdf")
    with pytest.raises((ImportError, NotImplementedError), match="Viser|Viser robot adapter"):
        debugger.visualize_collision_at_config(torch.zeros(7))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_debugger_runs_kinematics_and_collision_checks_on_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    debugger = RobotDebugger("franka.yml", DeviceCfg(device="mps"))
    result = debugger.check_default_joint_configuration_collision()
    sampled = debugger.sample_collision_checks(num_samples=2, batch_size=1, seed=9)
    assert result["num_spheres"] == 65
    assert sampled["total_samples"] == 2
    assert debugger._collision_cost._pair_distance.device.type == "mps"
