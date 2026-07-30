"""CUDA runtime preflight and provenance collection for paired replay."""

from __future__ import annotations

import inspect
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .replay_registry import PIN


def revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    values = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(values) or None


def collect(upstream: Path) -> dict[str, Any]:
    """Fail closed unless CUDA and exact pinned upstream imports are usable."""
    actual = revision(upstream)
    if actual != PIN:
        raise RuntimeError(f"refusing upstream revision {actual}; required {PIN}")
    sys.path.insert(0, str(upstream.resolve()))
    import torch
    from curobo.types import DeviceCfg, JointState, Pose

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device 0 is unavailable")
    source_root = upstream.resolve()
    for symbol in (DeviceCfg, Pose, JointState):
        source = Path(inspect.getfile(symbol)).resolve()
        if not source.is_relative_to(source_root):
            raise RuntimeError(f"{symbol.__name__} imported outside pinned upstream: {source}")
    probe = torch.tensor([1.0], device="cuda")
    if probe.device.type != "cuda" or probe.item() != 1.0:
        raise RuntimeError("CUDA tensor execution probe failed")
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_index": 0,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory": int(properties.total_memory),
        "nvidia_driver": _driver_version(),
        "upstream_revision": actual,
    }
