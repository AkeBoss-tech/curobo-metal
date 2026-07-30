import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tools.parity.compare_paired import compare_ready, sha256
from tools.parity.build_cuda_handoff import build as build_cuda_handoff
from tools.parity.replay_registry import BY_ID, PIN
from tools.parity.cuda_adapters import ADAPTERS
from tools.parity import cuda_runtime


ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "artifacts/parity/replay"


def run(module, *args, env=None):
    values = os.environ.copy()
    values["PYTHONPATH"] = str(ROOT / "src")
    if env:
        values.update(env)
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=ROOT, env=values, capture_output=True, text=True, check=False,
    )


def test_committed_corpus_covers_inventory_and_validates():
    result = run("tools.parity.validate_replay", ARTIFACT)
    assert result.returncode == 0, result.stderr
    index = json.loads((ARTIFACT / "index.json").read_text())
    assert {row["capability"] for row in index["cases"]} == set(BY_ID)
    assert len(index["cases"]) == 19


def test_inputs_are_identical_and_safe_npz():
    hashes = set()
    for capability in BY_ID:
        path = ARTIFACT / capability / "inputs.npz"
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        with np.load(path, allow_pickle=False) as data:
            assert {
                "q",
                "empty",
                "invalid_shape",
                "inertial_case_json",
                "robot_urdf_utf8",
            } <= set(data.files)
    assert len(hashes) == 1


def test_asset_independent_cuda_adapters_are_explicitly_registered():
    assert set(ADAPTERS) == {
        "collision.robot_scene",
        "configuration.robot_config_and_loaders",
        "cost.pose_and_composable_costs",
        "dynamics.inverse_dynamics",
        "kinematics.forward_kinematics",
        "kinematics.geometric_jacobian",
        "types.device_cfg",
        "types.pose",
        "types.joint_state",
        "types.solver_results",
    }
    assert set(ADAPTERS) <= set(BY_ID)


def _fake_cuda_evidence(root: Path) -> None:
    for capability in ADAPTERS:
        source = ARTIFACT / capability
        folder = root / capability
        folder.mkdir(parents=True)
        output = folder / "cuda-outputs.npz"
        shutil.copyfile(source / "metal-outputs.npz", output)
        with np.load(output, allow_pickle=False) as data:
            tensors = {
                key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                for key in sorted(data.files)
            }
        metal = json.loads((source / "metal-manifest.json").read_text())
        case = BY_ID[capability]
        with np.load(source / "inputs.npz", allow_pickle=False) as inputs:
            input_tensor_count = len(inputs.files)
        manifest = {
            "format": "curobo-metal-paired-replay",
            "version": 1,
            "capability": capability,
            "operation": case.operation,
            "backend": "cuda",
            "device": "cuda",
            "fallback_enabled": False,
            "upstream_revision": PIN,
            "input_sha256": metal["input"]["sha256"],
            "input_tensor_count": input_tensor_count,
            "status": "complete",
            "runtime": {
                "python": "3.13.9",
                "platform": "test-linux",
                "torch": "2.13.0",
                "torch_cuda": "13.0",
                "cuda_device_count": 1,
                "cuda_device_index": 0,
                "cuda_device_name": "Test GPU",
                "cuda_capability": [9, 0],
                "cuda_total_memory": 1,
                "nvidia_driver": "test",
                "upstream_revision": PIN,
            },
            "output": {
                "file": output.name,
                "sha256": sha256(output),
                "tensors": tensors,
            },
            "equivalence_claimed": False,
            "tolerance": {"rtol": case.rtol, "atol": case.atol},
        }
        (folder / "cuda-manifest.json").write_text(json.dumps(manifest))


def test_paired_verifier_accepts_complete_bound_evidence(tmp_path):
    _fake_cuda_evidence(tmp_path)
    report = compare_ready(ARTIFACT, tmp_path)
    assert report["passed"]
    assert not report["errors"]
    assert {row["capability"] for row in report["results"]} == set(ADAPTERS)


def test_paired_verifier_reports_numerical_mismatch(tmp_path):
    _fake_cuda_evidence(tmp_path)
    capability = "types.pose"
    folder = tmp_path / capability
    output = folder / "cuda-outputs.npz"
    with np.load(output, allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    values["points"][0, 0] += 1.0
    np.savez(output, **values)
    manifest_path = folder / "cuda-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output"]["sha256"] = sha256(output)
    manifest_path.write_text(json.dumps(manifest))
    report = compare_ready(ARTIFACT, tmp_path)
    result = next(row for row in report["results"] if row["capability"] == capability)
    assert not report["passed"]
    assert not result["passed"]
    assert not result["tensors"]["points"]["passed"]


def test_paired_verifier_rejects_input_provenance_tamper(tmp_path):
    _fake_cuda_evidence(tmp_path)
    manifest_path = tmp_path / "types.joint_state/cuda-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    report = compare_ready(ARTIFACT, tmp_path)
    assert not report["passed"]
    assert any(
        row["capability"] == "types.joint_state"
        and "different input" in row["error"]
        for row in report["errors"]
    )


def test_paired_verifier_rejects_incomplete_runtime_provenance(tmp_path):
    _fake_cuda_evidence(tmp_path)
    manifest_path = tmp_path / "types.pose/cuda-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime"]["torch_cuda"] = None
    manifest_path.write_text(json.dumps(manifest))
    report = compare_ready(ARTIFACT, tmp_path)
    assert not report["passed"]
    assert any(
        row["capability"] == "types.pose"
        and "runtime provenance" in row["error"]
        for row in report["errors"]
    )


def test_cuda_handoff_is_deterministic_and_self_verifying(tmp_path):
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    build_cuda_handoff(first)
    build_cuda_handoff(second)
    assert first.read_bytes() == second.read_bytes()
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(first) as archive:
        archive.extractall(extracted)
    verified = subprocess.run(
        [sys.executable, "verify_handoff.py"],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert "verified" in verified.stdout
    target = extracted / "tools/parity/cuda_adapters.py"
    target.write_text(target.read_text() + "\n# tampered\n")
    rejected = subprocess.run(
        [sys.executable, "verify_handoff.py"],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "hash mismatch" in rejected.stderr


def test_manifests_record_device_fallback_gradient_status_and_invalid_evidence():
    gradients = statuses = 0
    for capability, case in BY_ID.items():
        manifest = json.loads((ARTIFACT / capability / "metal-manifest.json").read_text())
        assert manifest["device"] == "mps"
        assert manifest["backend"] == "metal"
        assert manifest["fallback_enabled"] is False
        assert manifest["equivalence_claimed"] is False
        assert manifest["upstream_revision"] == PIN
        assert manifest["evidence"]["invalid_case"] == case.invalid_case
        if capability in ADAPTERS:
            assert manifest["evidence"]["invalid_executed"] is True
            with np.load(
                ARTIFACT / capability / manifest["output"]["file"],
                allow_pickle=False,
            ) as output:
                assert output["invalid_rejected"].shape == (1,)
        gradients += bool(manifest["evidence"]["gradient"])
        statuses += bool(manifest["evidence"]["status"])
    assert gradients >= 4
    assert statuses >= 5


def test_cuda_runner_strictly_refuses_wrong_sha(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q", upstream], check=True)
    subprocess.run(["git", "-C", upstream, "config", "user.email", "parity@example.invalid"], check=True)
    subprocess.run(["git", "-C", upstream, "config", "user.name", "Parity Test"], check=True)
    (upstream / "README").write_text("wrong revision\n")
    subprocess.run(["git", "-C", upstream, "add", "README"], check=True)
    subprocess.run(["git", "-C", upstream, "commit", "-qm", "wrong"], check=True)
    inputs = ARTIFACT / "types.pose/inputs.npz"
    result = run(
        "tools.parity.run_pinned_cuda", "--upstream", upstream, "--input", inputs,
        "--capability", "types.pose", "--output", tmp_path / "cuda",
    )
    assert result.returncode != 0
    assert f"required {PIN}" in result.stderr
    assert not (tmp_path / "cuda/cuda-manifest.json").exists()


def test_cuda_preflight_rejects_missing_device_before_upstream_import(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cuda_runtime, "revision", lambda _path: PIN)
    original_path = list(sys.path)
    try:
        with pytest.raises(RuntimeError, match="CUDA device 0 is unavailable"):
            cuda_runtime.collect(tmp_path)
    finally:
        assert sys.path == original_path


def test_generation_refuses_fallback_enabled(tmp_path):
    result = run(
        "tools.parity.generate_replay", "--device", "cpu", "--output", tmp_path / "out",
        env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
    )
    assert result.returncode != 0
    assert "must be unset or 0" in result.stderr
