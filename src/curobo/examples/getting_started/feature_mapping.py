"""Optional RGB feature fusion example.

Feature extraction is deliberately an adapter: callers provide a callable that
turns an RGB array into a feature grid, while cuRobo owns geometric integration.
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from .volumetric_mapping import Sun3dDataset


def pca_colorize(features: np.ndarray) -> np.ndarray:
    """Project ``(..., channels)`` features to uint8 RGB using SVD."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] < 1:
        raise ValueError("features must have a channel dimension")
    flat = values.reshape(-1, values.shape[-1]); centered = flat - flat.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[: min(3, vt.shape[0])].T
    components = np.pad(components, ((0, 0), (0, 3 - components.shape[1])))
    lo, hi = components.min(0), components.max(0)
    rgb = (components - lo) / np.maximum(hi - lo, 1e-8) * 255
    return rgb.astype(np.uint8).reshape(values.shape[:-1] + (3,))


def pca_colorize_tensor(features: np.ndarray) -> np.ndarray:
    """Compatibility alias accepting array-like tensor data."""
    return pca_colorize(features)


def extract_features(image: np.ndarray, encoder: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Encode an image and validate the resulting ``(H, W, C)`` feature grid."""
    features = np.asarray(encoder(image))
    if features.ndim != 3:
        raise ValueError(f"feature encoder must return (height, width, channels), got {features.shape}")
    return features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=1)
    args = parser.parse_args(argv)
    dataset = Sun3dDataset(args.root)
    print(f"Feature mapping is ready for {min(args.num_frames, len(dataset))} frame(s); provide a C-RADIO encoder to integrate features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
