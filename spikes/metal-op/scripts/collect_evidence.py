#!/usr/bin/env python3
"""Run the spike and emit machine-readable proof without enabling CPU fallback."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from curobo_metal_op_spike import metal_square


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    x = torch.tensor([-3.0, -0.5, 0.0, 2.0, 7.0], device="mps", requires_grad=True)
    y = metal_square(x)
    y.sum().backward()
    torch.mps.synchronize()

    expected_y = x.detach().cpu().square()
    expected_grad = 2.0 * x.detach().cpu()
    torch.testing.assert_close(y.detach().cpu(), expected_y)
    torch.testing.assert_close(x.grad.detach().cpu(), expected_grad)

    evidence = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": command("sw_vers", "-productVersion"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_device_count": torch.mps.device_count(),
        "compile_shader_available": hasattr(torch.mps, "compile_shader"),
        "cpu_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "input_device": str(x.device),
        "output_device": str(y.device),
        "gradient_device": str(x.grad.device),
        "forward": y.detach().cpu().tolist(),
        "gradient": x.grad.detach().cpu().tolist(),
        "status": "pass",
    }
    destination = Path(__file__).resolve().parents[3] / "artifacts/toolchain/evidence.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()

