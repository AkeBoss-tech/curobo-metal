"""Shape and failure-boundary tests for CUDA-only low-level facades."""

from __future__ import annotations

import inspect

import pytest
import torch


def test_raw_launch_helpers_keep_pinned_vararg_shape_and_fail_closed():
    from curobo._src.curobolib.backends.cuda_core_backend._launch import (
        RawCudaKernelUnavailableError,
    )
    from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import (
        launch_kernel,
    )

    assert list(inspect.signature(launch_kernel).parameters) == [
        "kernel_name", "stream", "config", "kernel", "kernel_args"
    ]
    assert inspect.signature(launch_kernel).parameters["kernel_args"].kind is (
        inspect.Parameter.VAR_POSITIONAL
    )
    with pytest.raises(RawCudaKernelUnavailableError, match="raw CUDA"):
        launch_kernel("fake", object(), object(), object(), 1, 2)


def test_raw_kinematics_and_spline_surfaces_reject_after_argument_binding():
    from curobo._src.curobolib.backends.cuda_core_backend import kinematics, trajectory

    assert "output_threads_per_batch" in inspect.signature(
        kinematics.launch_kinematics_forward_spheres
    ).parameters
    assert "compute_jacobian_grad" in inspect.signature(
        kinematics.launch_kinematics_backward
    ).parameters
    assert "bspline_degree" in inspect.signature(
        trajectory.launch_bspline_interpolation_forward_kernel
    ).parameters

    with pytest.raises(NotImplementedError, match="CUDA pointer"):
        kinematics.launch_kinematics_forward(*([None] * 15))
    with pytest.raises(NotImplementedError, match="raw CUDA"):
        trajectory.launch_bspline_interpolation_backward_kernel(*([None] * 14))


def test_tensor_checks_match_keyword_only_curobo_surface():
    from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors

    parameter = inspect.signature(check_float32_tensors).parameters["tensors"]
    assert parameter.kind is inspect.Parameter.VAR_KEYWORD
    check_float32_tensors(torch.device("cpu"), value=torch.ones(2))
    with pytest.raises(ValueError, match="expected a tensor"):
        check_float32_tensors(torch.device("cpu"), value=None)


def test_raw_autograd_backward_members_are_explicit_boundaries():
    from curobo._src.curobolib.cuda_ops.dynamics import RNEAForwardFunction
    from curobo._src.curobolib.cuda_ops.kinematics import KinematicsFusedFunction

    assert hasattr(RNEAForwardFunction, "backward")
    assert hasattr(KinematicsFusedFunction, "backward")
