import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.parity.replay_registry import BY_ID, PIN
from tools.parity.cuda_adapters import ADAPTERS


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
            assert {"q", "empty", "invalid_shape", "inertial_case_json"} <= set(data.files)
    assert len(hashes) == 1


def test_asset_independent_cuda_adapters_are_explicitly_registered():
    assert set(ADAPTERS) == {
        "types.device_cfg",
        "types.pose",
        "types.joint_state",
    }
    assert set(ADAPTERS) <= set(BY_ID)


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


def test_generation_refuses_fallback_enabled(tmp_path):
    result = run(
        "tools.parity.generate_replay", "--device", "cpu", "--output", tmp_path / "out",
        env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
    )
    assert result.returncode != 0
    assert "must be unset or 0" in result.stderr
