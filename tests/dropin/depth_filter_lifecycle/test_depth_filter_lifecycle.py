"""Portable lifecycle checks for the pinned V2 depth-filter facade."""

import pytest
import torch

from curobo._src.perception.filter_depth import FilterDepth, FilterDepthConfig


def test_batched_buffers_are_reused_and_owned_outputs_are_honored():
    filt = FilterDepth((3, 4), bilateral_kernel_size=None, flying_pixel_threshold=None, device="cpu", num_batch=2)
    depth = torch.full((2, 3, 4), 1.0, dtype=torch.float32)
    first, first_mask = filt(depth)
    second, second_mask = filt(depth * 2.0)
    assert first.data_ptr() == second.data_ptr() == filt._depth_out.data_ptr()
    assert first_mask.data_ptr() == second_mask.data_ptr() == filt._valid_mask_out.data_ptr()

    output = torch.empty_like(depth)
    mask = torch.empty_like(depth, dtype=torch.bool)
    received, received_mask = filt(depth, output, mask)
    assert received is output and received_mask is mask
    assert received_mask.dtype is torch.bool and received_mask.all()


def test_source_flying_threshold_is_relative_and_zero_update_disables_it():
    filt = FilterDepth((3, 3), bilateral_kernel_size=None, flying_pixel_threshold=0.5, device="cpu")
    depth = torch.ones((1, 3, 3), dtype=torch.float32)
    depth[0, 1, 1] = 3.0
    _, valid = filt(depth)
    assert not valid[0, 1, 1]
    filt.update_config(flying_pixel_threshold=0)
    _, valid = filt(depth)
    assert valid.all()


def test_range_finite_and_bilateral_contract_matches_buffers_for_dynamic_shape():
    filt = FilterDepth((2, 2), flying_pixel_threshold=None, bilateral_kernel_size=3, device="cpu")
    depth = torch.tensor([[[1.0, float("nan"), 1.0], [1.0, 0.01, 1.0]]], dtype=torch.float32)
    output, valid = filt(depth)
    assert output.shape == valid.shape == depth.shape
    assert output[0, 0, 1] == 0 and output[0, 1, 1] == 0
    assert not valid[0, 0, 1] and not valid[0, 1, 1]
    assert torch.isfinite(output).all()


def test_config_roundtrip_validation_and_buffer_errors_are_precise():
    config = FilterDepthConfig(depth_minimum_distance=0.2, depth_maximum_distance=2.0, bilateral_kernel_size=None)
    filt = FilterDepth.from_config(config, (2, 2), device="cpu")
    assert filt.config == config
    depth = torch.ones((1, 2, 2), dtype=torch.float32)
    with pytest.raises(TypeError, match="float32"):
        filt(depth.double())
    with pytest.raises(ValueError, match="depth_out"):
        filt(depth, torch.empty((2, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="positive"):
        FilterDepth((0, 2))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS is unavailable")
def test_mps_filter_is_fallback_free_and_preserves_mps_buffers():
    filt = FilterDepth((3, 3), bilateral_kernel_size=3, device="mps", num_batch=2)
    depth = torch.ones((2, 3, 3), dtype=torch.float32, device="mps")
    depth[0, 1, 1] = 2.0
    output, valid = filt(depth)
    assert output.device.type == valid.device.type == "mps"
    assert output.data_ptr() == filt._depth_out.data_ptr()
    assert valid.dtype is torch.bool and torch.isfinite(output).all()
