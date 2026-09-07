"""Backend-neutral JSON observations; this module never changes execution devices."""

import json

import torch


def tensor(value: torch.Tensor, *, values: bool = True) -> dict:
    assert isinstance(value, torch.Tensor)
    assert bool(torch.isfinite(value).all().item()), "non-finite application result"
    result = {"shape": list(value.shape), "dtype": str(value.dtype), "device": value.device.type}
    if values:
        result["values"] = value.detach().cpu().tolist()
    return result


def emit(observations: dict) -> None:
    print("CUROBO_APPLICATION_RESULT=" + json.dumps(observations, sort_keys=True, allow_nan=False))
