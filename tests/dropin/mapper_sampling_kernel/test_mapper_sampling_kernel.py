"""Portable contracts for mapper Warp-facing sampling helpers."""

import math

import pytest
import torch

from curobo._src.perception.mapper.kernel import wp_filter_depth, wp_tsdf_sample


class _TSDF:
    has_dynamic = True
    has_static = True
    block_data = torch.tensor([[[4.0, 2.0]]])
    static_block_data = torch.tensor([[3.0]])


def test_flying_pixel_scalar_contract_and_raw_kernel_boundary():
    assert wp_filter_depth.compute_flying_pixel_threshold(0.0) == pytest.approx(0.08)
    assert wp_filter_depth.compute_flying_pixel_threshold(1.0) == pytest.approx(0.005)
    assert wp_filter_depth.is_flying_pixel(1.0, 1.0, 1.2, 1.0, 1.0, 0.1)
    assert not wp_filter_depth.is_flying_pixel(1.0, 1.0, 1.05, 1.0, 1.0, 0.1)

    with pytest.raises(NotImplementedError, match="Warp"):
        wp_filter_depth.filter_depth_fused_kernel(
            None, None, None, 0.0, math.inf, 0, 0.0, 0, 0, 1.0, 1.0
        )


def test_tsdf_sampling_preserves_channel_and_weight_semantics():
    tsdf = _TSDF()
    assert wp_tsdf_sample.sample_dynamic_sdf(tsdf, 0, 0, 0.5).item() == 2.0
    assert wp_tsdf_sample.sample_static_sdf(tsdf, 0, 0).item() == 3.0
    assert wp_tsdf_sample.sample_combined_sdf(tsdf, 0, 0, 0.5).item() == 2.0
    assert wp_tsdf_sample.has_valid_observation(tsdf, 0, 0, 0.5)

    tsdf.has_dynamic = False
    tsdf.has_static = False
    assert wp_tsdf_sample.sample_dynamic_sdf(tsdf, 0, 0, 0.5) == 1e10
    assert wp_tsdf_sample.sample_static_sdf(tsdf, 0, 0) == 1e10
    assert not wp_tsdf_sample.has_valid_observation(tsdf, 0, 0, 0.5)
