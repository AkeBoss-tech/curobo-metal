"""Small Python helpers used across the portable API."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Union

import torch

from curobo._src.util.logging import log_and_raise


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {key: default_to_regular(value) for key, value in d.items()}
    elif isinstance(d, dict):
        d = {key: default_to_regular(value) for key, value in d.items()}
    return d


def list_idx_if_not_none(d_list: List, idx: Union[int, torch.Tensor]):
    output = []
    for value in d_list:
        if value is None:
            output.append(None)
            continue
        if isinstance(idx, int) and (idx >= len(value) or idx < -len(value)):
            log_and_raise(f"Index {idx} out of range for {value.shape}")
        output.append(value[idx])
    return output


def robust_floor(x: float, threshold: float = 0.0001) -> int:
    rounded = round(x)
    return rounded if abs(x - rounded) < threshold else int(math.floor(x))
