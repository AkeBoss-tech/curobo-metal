from __future__ import annotations

import json
from pathlib import Path

from tools.gauntlet.portable_test_adapter import adapt_source


def test_adapts_only_device_literals_and_availability_gate() -> None:
    source = """
device = torch.device("cuda:0")
name = "cuda_device_cfg"
if torch.cuda.is_available():
    run_cuda_graph()
"""
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
    assert result.availability_preserved == 0
    assert result.is_cuda_assertion_replacements == 0


def test_redirects_pinned_unshipped_helper_import() -> None:
    source = (
        "from curobo.examples.reference.lidar_volumetric_mapping import "
        "tsdf_surface_voxels_with_blocks\n"
    )
    result = adapt_source(source)
    assert result.source == (
        "from _curobo_upstream_helpers.lidar_volumetric_mapping import "
        "tsdf_surface_voxels_with_blocks\n"
    )
    assert result.helper_import_replacements == 1


def test_preserves_availability_gate_for_raw_cuda_graph_case() -> None:
    source = """
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_capture():
    executor = GraphExecutor(device="cuda", use_cuda_graph=True)
    executor.capture()
"""
    result = adapt_source(
        source,
        preserve_availability_scopes=frozenset({"test_graph_capture"}),
    )
    assert "torch.cuda.is_available()" in result.source
    assert "device='mps'" in result.source
    assert result.availability_replacements == 0
    assert result.availability_preserved == 1


def test_adapts_cuda_worded_gate_when_case_uses_only_portable_tensors() -> None:
    source = """
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_portable_math():
    value = torch.ones(2, device="cuda")
    assert value.sum() == 2
"""
    result = adapt_source(source)
    assert "torch.backends.mps.is_available()" in result.source
    assert "device='mps'" in result.source
    assert result.availability_replacements == 1
    assert result.availability_preserved == 0


def test_adapts_device_timing_barrier_but_preserves_reviewed_event_case() -> None:
    source = """
def test_portable_timing():
    torch.cuda.synchronize()

def test_cuda_event_timing():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    torch.cuda.synchronize()
"""
    result = adapt_source(
        source,
        preserve_availability_scopes=frozenset({"test_cuda_event_timing"}),
    )
    assert result.source.count("torch.mps.synchronize()") == 1
    assert result.source.count("torch.cuda.synchronize()") == 1
    assert result.synchronize_replacements == 1


def test_keeps_warp_case_raw_without_suppressing_adjacent_portable_case() -> None:
    source = """
def test_warp_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    init_warp()

def test_portable_kernel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.ones(1, device="cuda")
"""
    result = adapt_source(
        source,
        preserve_availability_scopes=frozenset({"test_warp_kernel"}),
    )
    assert result.source.count("torch.cuda.is_available()") == 1
    assert result.source.count("torch.backends.mps.is_available()") == 1
    assert result.availability_replacements == 1
    assert result.availability_preserved == 1


def test_multiple_availability_replacements_preserve_source_boundaries() -> None:
    source = """
def test_first():
    device = "cuda:0"
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

def test_second():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
"""
    result = adapt_source(source)
    assert result.source.count("torch.backends.mps.is_available()") == 2
    assert "pytest.skip" in result.source
    assert result.availability_replacements == 2
    compile(result.source, "<adapted>", "exec")


def test_preserves_policy_reviewed_class_level_gate() -> None:
    source = """
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCudaGraph:
    def test_capture(self):
        capture_graph()
"""
    result = adapt_source(
        source,
        preserve_availability_scopes=frozenset({"TestCudaGraph"}),
    )
    assert "torch.cuda.is_available()" in result.source
    assert result.availability_replacements == 0
    assert result.availability_preserved == 1


def test_adapts_only_assertion_scoped_cuda_residency_checks() -> None:
    source = """
value = tensor.is_cuda
assert result.success.is_cuda
assert not cloned.solution.is_cuda
assert result.success.is_cuda, tensor.is_cuda
"""
    result = adapt_source(source)
    assert "value = tensor.is_cuda" in result.source
    assert "assert (result.success.device.type == 'mps')" in result.source
    assert "assert not (cloned.solution.device.type == 'mps')" in result.source
    assert (
        "assert (result.success.device.type == 'mps'), tensor.is_cuda" in result.source
    )
    assert result.is_cuda_assertion_replacements == 3
    compile(result.source, "<adapted>", "exec")


def test_policy_forbids_assertion_tolerance_and_cuda_api_rewrites() -> None:
    root = Path(__file__).resolve().parents[2]
    policy = json.loads((root / "gauntlet/portable-dropin-parity.json").read_text())
    forbidden = " ".join(policy["test_adaptation"]["forbidden"])
    assert "assertions" in forbidden
    assert "tolerances" in forbidden
    assert "CUDA graph" in forbidden
    assert "Warp APIs" in forbidden
    exclusions = policy["mechanism_exclusions"]
    assert exclusions
    assert all("::test_" in item["case_pattern"] for item in exclusions)
    assert all(item["reason"].strip() for item in exclusions)


def test_adapts_only_pinned_device_cfg_portability_oracles() -> None:
    source = """
class TestDeviceCfg:
    def test_default_initialization(self):
        cfg = DeviceCfg()
        assert cfg.device == torch.device('mps', 0)

    def test_from_basic_cpu(self):
        cfg = DeviceCfg.from_basic("cpu", 0)
        assert cfg.device == torch.device("cpu", 0)

    def test_from_basic_cuda(self):
        cfg = DeviceCfg.from_basic('mps', 0)
        assert cfg.device == torch.device('mps', 0)
"""
    result = adapt_source(source, test_module="curobo.tests._src.types.test_device_cfg")
    assert "assert cfg.device == torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')" in result.source
    assert 'cfg.is_same_torch_device(torch.device("cpu", 0))' in result.source
    assert "cfg.is_same_torch_device(torch.device('mps', 0))" in result.source
    assert result.oracle_replacements == 3
    compile(result.source, "<adapted>", "exec")


def test_does_not_adapt_device_cfg_oracles_in_other_modules() -> None:
    source = "assert cfg.device == torch.device('mps', 0)\n"
    result = adapt_source(source)
    assert result.source == source
    assert result.oracle_replacements == 0


def test_guards_only_pinned_voxel_raw_warp_scaffold_for_collection() -> None:
    source = """import torch
_test_compute_local_sdf = None
for _module_path in OBSTACLE_SDF_MODULES:
    _test_compute_local_sdf = wp.func(load(_module_path))

@wp.kernel(enable_backward=False)
def _test_sdf_kernel():
    _test_compute_local_sdf()

def _make_empty_esdf(
    device="cuda:0",
):
    return VoxelGrid(device=device)

class TestVoxelDataCreation:
    def test_create_from_scene_cfg(self):
        assert _make_empty_esdf() is not None
"""
    result = adapt_source(source, test_module="curobo.tests._src.geom.sdf.test_voxel_collision")
    assert result.mechanism_scaffold_guards == 1
    assert "if torch.cuda.is_available():\n    for _module_path" in result.source
    assert "\ndef _make_empty_esdf(" in result.source
    assert "device='mps:0'" in result.source
    compile(result.source, "<adapted>", "exec")


def test_does_not_guard_similar_scaffold_in_another_module() -> None:
    source = "for _module_path in OBSTACLE_SDF_MODULES:\n    register(_module_path)\n"
    result = adapt_source(source, test_module="curobo.tests._src.geom.sdf.test_other")
    assert result.source == source
    assert result.mechanism_scaffold_guards == 0
