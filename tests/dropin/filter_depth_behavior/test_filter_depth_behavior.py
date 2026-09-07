"""Portable behavioral coverage for the public batched depth-filter contract."""

import pytest
import torch

from curobo._src.perception.filter_depth import FilterDepth


def test_filter_depth_rejects_unbatched_image():
    depth_filter = FilterDepth(
        image_shape=(4, 5),
        flying_pixel_threshold=None,
        bilateral_kernel_size=None,
        device="cpu",
    )

    with pytest.raises(ValueError, match=r"\(B, H, W\)"):
        depth_filter(torch.ones(4, 5))


def test_filter_depth_accepts_batched_image():
    depth_filter = FilterDepth(
        image_shape=(4, 5),
        flying_pixel_threshold=None,
        bilateral_kernel_size=None,
        device="cpu",
    )
    filtered, valid = depth_filter(torch.full((2, 4, 5), 2.0))

    assert filtered.shape == (2, 4, 5)
    assert valid.shape == (2, 4, 5)
    assert valid.all()
