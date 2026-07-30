"""Small Python helpers used across the portable API."""

from __future__ import annotations

from collections import defaultdict


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {key: default_to_regular(value) for key, value in d.items()}
    elif isinstance(d, dict):
        d = {key: default_to_regular(value) for key, value in d.items()}
    return d


def list_idx_if_not_none(d_list, idx):
    return [None if value is None else value[idx] for value in d_list]


def robust_floor(x: float, threshold: float = 0.0001):
    rounded = round(x)
    return rounded if abs(x - rounded) < threshold else int(x // 1)
