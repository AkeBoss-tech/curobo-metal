from __future__ import annotations

import json
from pathlib import Path

from tools.gauntlet.portable_test_adapter import adapt_source


def test_adapts_only_device_literals_and_availability_gate() -> None:
    source = '''
device = torch.device("cuda:0")
name = "cuda_device_cfg"
if torch.cuda.is_available():
    run_cuda_graph()
'''
    result = adapt_source(source)
    assert "torch.device('mps:0')" in result.source
    assert '"cuda_device_cfg"' in result.source
    assert "torch.backends.mps.is_available()" in result.source
    assert "run_cuda_graph()" in result.source
    assert result.device_string_replacements == 1
    assert result.availability_replacements == 1


def test_conftest_mode_keeps_cuda_seed_guard_but_adapts_fixture_devices() -> None:
    source = 'params=["cpu", "cuda:0"]\nif torch.cuda.is_available(): pass\n'
    result = adapt_source(source, adapt_availability=False)
    assert "'mps:0'" in result.source
    assert "torch.cuda.is_available()" in result.source
    assert result.availability_replacements == 0


def test_policy_forbids_assertion_tolerance_and_cuda_api_rewrites() -> None:
    root = Path(__file__).resolve().parents[2]
    policy = json.loads((root / "gauntlet/portable-dropin-parity.json").read_text())
    forbidden = " ".join(policy["test_adaptation"]["forbidden"])
    assert "assertions" in forbidden
    assert "tolerances" in forbidden
    assert "CUDA graph" in forbidden
    assert "Warp APIs" in forbidden
    exclusions = policy["mechanism_exclusions"]
    assert len(exclusions) == 5
    assert all("::test_" in item["case_pattern"] for item in exclusions)
    assert all("float64" in item["reason"] for item in exclusions)
