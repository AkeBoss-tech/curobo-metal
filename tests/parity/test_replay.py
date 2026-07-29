import json
import subprocess
import sys

import numpy as np


def run(*args):
    return subprocess.run([sys.executable, "-m", "tools.parity.replay", *map(str, args)], check=False, capture_output=True, text=True)


def test_pack_and_compare_empty_batch_and_nan(tmp_path):
    inputs = tmp_path / "inputs.npz"
    outputs = tmp_path / "outputs.npz"
    np.savez(inputs, q=np.empty((0, 7), dtype=np.float32))
    np.savez(outputs, value=np.array([np.nan, 1.0], dtype=np.float32))
    bundles = [tmp_path / "cuda", tmp_path / "metal"]
    for backend, bundle in zip(("cuda", "metal"), bundles):
        result = run("pack", "--bundle", bundle, "--inputs", inputs, "--outputs", outputs, "--case-id", "empty", "--operation", "fk", "--backend", backend)
        assert result.returncode == 0, result.stderr
    result = run("compare", *bundles)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"]


def test_compare_detects_numerical_mismatch(tmp_path):
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, q=np.array([[0.0]], dtype=np.float32))
    bundles = []
    for backend, value in (("cuda", 0.0), ("metal", 1.0)):
        output = tmp_path / f"{backend}.npz"
        np.savez(output, value=np.array([value], dtype=np.float32))
        bundle = tmp_path / f"{backend}-bundle"
        assert run("pack", "--bundle", bundle, "--inputs", inputs, "--outputs", output, "--case-id", "mismatch", "--operation", "cost", "--backend", backend).returncode == 0
        bundles.append(bundle)
    assert run("compare", *bundles, "--atol", 0).returncode == 1
