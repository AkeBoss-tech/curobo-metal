import importlib
from pathlib import Path

import pytest
import torch


OWNED_MODULES = (
    "curobo._src.curobolib",
    "curobo._src.curobolib.backends",
    "curobo._src.curobolib.backends.cuda_core_backend.dynamics",
    "curobo._src.curobolib.backends.cuda_core_backend.geometry",
    "curobo._src.curobolib.backends.cuda_core_backend.kinematics",
    "curobo._src.curobolib.backends.cuda_core_backend.optimization",
    "curobo._src.curobolib.backends.cuda_core_backend.pba",
    "curobo._src.curobolib.backends.cuda_core_backend.trajectory",
    "curobo._src.curobolib.cuda_ops.dynamics",
    "curobo._src.curobolib.cuda_ops.geometry",
    "curobo._src.curobolib.cuda_ops.kinematics",
    "curobo._src.curobolib.cuda_ops.optimization",
    "curobo._src.curobolib.cuda_ops.tensor_checks",
    "curobo._src.curobolib.cuda_ops.trajectory",
    "curobo._src.util.cuda_event_timer",
    "curobo._src.util.cuda_graph_util",
    "curobo._src.util.cuda_stream_util",
    "curobo._src.util.torch_util",
    "curobo._src.util.warp",
    "curobo._src.util.warp_interpolation",
)


def test_owned_modules_import_without_cuda():
    for module in OWNED_MODULES:
        assert importlib.import_module(module)


def test_portable_backend_and_explicit_native_boundaries(tmp_path: Path):
    from curobo._src.curobolib.backends import get_backend, get_backend_name
    from curobo._src.curobolib.backends.cuda_core_backend.kernel_cache import (
        CudaCoreKernelCache,
    )
    from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch
    from curobo._src.curobolib.backends.cuda_core_backend.pba import launch_pba3d
    from curobo._src.util.warp import init_warp

    assert get_backend_name() == "portable"
    assert "trajectory" in get_backend()
    with pytest.raises(NotImplementedError, match="CUDA"):
        launch()
    with pytest.raises(NotImplementedError, match="CUDA"):
        CudaCoreKernelCache().get_or_compile_kernel([], "kernel", [], [])
    with pytest.raises(NotImplementedError, match="ESDF"):
        launch_pba3d(None, None, 1, 1, 1)
    with pytest.raises(NotImplementedError, match="Warp"):
        init_warp()


def test_graph_executor_is_direct_and_shape_stable(tmp_path: Path):
    from curobo._src.util.cuda_graph_util import create_graph_executor

    executor = create_graph_executor(lambda x: x.square(), "cpu", use_cuda_graph=True)
    assert torch.equal(executor(torch.tensor([2.0])), torch.tensor([4.0]))
    assert torch.equal(executor(torch.tensor([2.0, 3.0])), torch.tensor([4.0, 9.0]))
    assert executor.is_initialized
    executor.debug_dump(tmp_path / "graph.txt")
    assert "no CUDA graph" in (tmp_path / "graph.txt").read_text()
    executor.reset()
    assert not executor.is_initialized


def test_lbfgs_low_level_step_and_tensor_checks():
    from curobo._src.curobolib.cuda_ops.optimization import LBFGScu
    from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors

    q = torch.tensor([[2.0, -1.0]])
    grad = torch.tensor([[4.0, -2.0]])
    step = torch.zeros_like(q)
    y = torch.zeros((1, 1, 2, 1))
    s = torch.zeros_like(y)
    rho = torch.zeros((1, 1, 1))
    result = LBFGScu.apply(
        step, rho, y, s, q, grad, torch.zeros_like(q), torch.zeros_like(q)
    )
    assert torch.equal(result, -grad)
    check_float32_tensors(q.device, q=q)
    with pytest.raises((TypeError, ValueError)):
        check_float32_tensors(q.device, bad=q.double())


def test_acceleration_rollout_uses_portable_torch():
    from curobo._src.curobolib.backends.cuda_core_backend.trajectory import (
        launch_integration_acceleration_kernel,
    )

    acceleration = torch.ones((1, 3, 1))
    outputs = [torch.empty_like(acceleration) for _ in range(4)]
    launch_integration_acceleration_kernel(
        *outputs, acceleration, torch.zeros((1, 1)), torch.zeros((1, 1)),
        torch.zeros((1, 1)), torch.zeros(1, dtype=torch.int32),
        torch.tensor([0.1]), 1, 3, 1,
    )
    position, velocity, acceleration_out, jerk = outputs
    assert torch.allclose(acceleration_out, acceleration)
    assert position.shape == velocity.shape == jerk.shape == acceleration.shape
    assert torch.isfinite(position).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_portable_low_level_ops_run_on_mps_without_fallback():
    from curobo._src.curobolib.cuda_ops.optimization import LBFGScu

    device = torch.device("mps")
    q = torch.tensor([[1.0, 2.0]], device=device)
    grad = torch.tensor([[2.0, 4.0]], device=device)
    result = LBFGScu.apply(
        torch.zeros_like(q),
        torch.zeros((1, 1, 1), device=device),
        torch.zeros((1, 1, 2, 1), device=device),
        torch.zeros((1, 1, 2, 1), device=device),
        q,
        grad,
        torch.zeros_like(q),
        torch.zeros_like(q),
    )
    torch.mps.synchronize()
    assert result.device.type == "mps"
    assert torch.equal(result.cpu(), torch.tensor([[-2.0, -4.0]]))
