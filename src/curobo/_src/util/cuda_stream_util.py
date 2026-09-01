from contextlib import contextmanager, nullcontext
from typing import Dict, Optional

import torch

from curobo import runtime as curobo_runtime


def create_cuda_stream_pair(
    device: torch.device, enabled: Optional[bool] = None
) -> tuple:
    if enabled is False or device.type != "cuda":
        return None, None
    return torch.cuda.Stream(device=device), torch.cuda.Event()


@contextmanager
def cuda_stream_context(
    stream_name: str,
    streams_dict: Dict[str, torch.cuda.Stream],
    events_dict: Dict[str, torch.cuda.Event],
    device: torch.device,
    enabled: Optional[bool] = None,
):
    yield


def synchronize_cuda_streams(
    events_dict: Dict[str, torch.cuda.Event],
    device: torch.device,
    enabled: Optional[bool] = None,
):
    return None
