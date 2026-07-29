#!/usr/bin/env python3
"""Decompose composed-PyTorch FK costs on CPU or MPS.

Wall-clock samples are synchronized on both sides.  The ``core_without_finite``
measurement is diagnostic only: production validation remains enabled.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from curobo_metal.backend import resolve_device, synchronize
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics
from curobo_metal.reference import SerialRobot, load_case

ROOT = Path(__file__).parents[2]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "panda_serial.json"
forward_module = importlib.import_module("curobo_metal.ops.kinematics.forward")


def synchronized_ms(device: torch.device, call: Callable[[], Any]) -> float:
    synchronize(device)
    started = time.perf_counter_ns()
    call()
    synchronize(device)
    return (time.perf_counter_ns() - started) / 1_000_000


def distribution(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999) - 1)],
    }


def measure(
    device: torch.device, call: Callable[[], Any], warmup: int, iterations: int
) -> dict[str, Any]:
    for _ in range(warmup):
        synchronized_ms(device, call)
    return distribution([synchronized_ms(device, call) for _ in range(iterations)])


def structural_validate(chain: KinematicChain, q: torch.Tensor):
    """Mirror _validate_q except for the synchronizing finite-value reduction."""
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if q.device.type != chain.device.type or q.dtype != chain.dtype:
        raise ValueError("diagnostic received mismatched q")
    input_was_batched = q.ndim == 2
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if q.ndim != 2 or q.shape[1] != chain.dof:
        raise ValueError("diagnostic received invalid shape")
    return q, input_was_batched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("MPS profiling requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    case = load_case(args.fixture)
    robot = SerialRobot.from_dict(case["robot"])
    source = torch.tensor(case["inputs"]["q"], device=device, dtype=torch.float32)
    repeats = (args.batch_size + source.shape[0] - 1) // source.shape[0]
    q = source.repeat((repeats, 1))[: args.batch_size]
    non_contiguous_backing = torch.empty(
        (args.batch_size, robot.dof, 2), device=device, dtype=q.dtype
    )
    non_contiguous_backing[..., 0] = q
    q_non_contiguous = non_contiguous_backing[..., 0]
    assert not q_non_contiguous.is_contiguous()
    chain = KinematicChain(robot, device=device, dtype=q.dtype)

    with torch.inference_mode():
        measurements = {
            "full_forward": measure(
                device, lambda: forward_kinematics(chain, q), args.warmup, args.iterations
            ),
            "finite_validation": measure(
                device,
                lambda: bool(torch.isfinite(q).all().item()),
                args.warmup,
                args.iterations,
            ),
            "contiguous_copy": measure(
                device, lambda: q.T.contiguous(), args.warmup, args.iterations
            ),
            "non_contiguous_forward": measure(
                device,
                lambda: forward_kinematics(chain, q_non_contiguous),
                args.warmup,
                args.iterations,
            ),
        }
        original_validate = forward_module._validate_q
        try:
            forward_module._validate_q = structural_validate
            measurements["core_without_finite_validation"] = measure(
                device, lambda: forward_kinematics(chain, q), args.warmup, args.iterations
            )
        finally:
            forward_module._validate_q = original_validate

    q_grad = q.detach().clone().requires_grad_(True)

    def forward_backward() -> None:
        q_grad.grad = None
        forward_kinematics(chain, q_grad).transforms.sum().backward()

    measurements["forward_and_backward"] = measure(
        device, forward_backward, args.warmup, args.iterations
    )
    measurements["grad_enabled_forward"] = measure(
        device,
        lambda: forward_kinematics(chain, q_grad),
        args.warmup,
        args.iterations,
    )

    # CPU activity measures Python/operator dispatch, not asynchronous GPU time.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
    ) as profile:
        with torch.inference_mode():
            forward_kinematics(chain, q)
        synchronize(device)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))
    operators = [
        {
            "name": event.key,
            "calls": event.count,
            "self_cpu_time_us": event.self_cpu_time_total,
            "cpu_time_us": event.cpu_time_total,
            "self_cpu_memory_bytes": event.self_cpu_memory_usage,
        }
        for event in sorted(
            profile.key_averages(),
            key=lambda event: event.self_cpu_time_total,
            reverse=True,
        )
    ]
    result = {
        "format": "curobo-metal-fk-profile",
        "version": 1,
        "device": str(device),
        "batch_size": args.batch_size,
        "fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "measurements": measurements,
        "profiler_scope": "one synchronized inference forward; CPU activity only",
        "operators": operators,
        "output_bytes": {
            name: value.numel() * value.element_size()
            for name, value in vars(forward_kinematics(chain, q)).items()
            if isinstance(value, torch.Tensor)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
