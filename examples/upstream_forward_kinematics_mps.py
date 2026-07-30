"""Run the pinned cuRobo forward-kinematics tutorial on CPU or Apple MPS.

This preserves the observable workload from
``curobo.examples.getting_started.forward_kinematics``: load ``franka.yml``,
run single and 1,000-way batched FK, print the end-effector pose, and
differentiate a position loss. It additionally checks every transform and the
loss gradient against the independent NumPy float64 oracle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch

from curobo_metal.config import RobotCfg
from curobo_metal.ops.kinematics import forward_kinematics
from curobo_metal.reference.forward_kinematics import (
    forward_kinematics as reference_forward_kinematics,
)
from curobo_metal.types import DeviceCfg, Pose

PINNED_UPSTREAM = "8e734f3ced1df898990bcd92de40abce475907db"


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def run(
    robot_yaml: str | Path,
    *,
    device: str = "mps",
    batch_size: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(device)
    robot_cfg = RobotCfg.create(
        robot_yaml,
        device_cfg=DeviceCfg(target_device, torch.float32),
        load_collision_spheres=False,
    )
    serial = robot_cfg.to_serial_robot()
    chain = robot_cfg.to_kinematic_chain()
    print(f"Robot has {chain.dof} degrees of freedom")
    print(f"Tool frames: {robot_cfg.tool_frames}")

    q = torch.zeros((1, chain.dof), device=target_device, dtype=torch.float32)
    state = forward_kinematics(chain, q)
    ee_pose = Pose.from_matrix(state.transforms[:, -1])
    print("\nSingle FK:")
    print(f"  EE position: {ee_pose.position}")
    print(f"  EE quaternion (wxyz): {ee_pose.quaternion}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_batch_cpu = torch.rand(
        (batch_size, chain.dof), generator=generator, dtype=torch.float32
    )
    q_batch = q_batch_cpu.to(target_device)
    _synchronize(target_device)
    start = time.perf_counter()
    state_batch = forward_kinematics(chain, q_batch)
    _synchronize(target_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"\nBatched FK ({batch_size} configs): {elapsed_ms:.2f} ms")
    print(
        "  EE positions shape: "
        f"torch.Size([{batch_size}, 1, 3]) "
        f"(portable tensor: {tuple(state_batch.transforms[:, -1:, :3, 3].shape)})"
    )

    reference = reference_forward_kinematics(
        serial, q_batch_cpu.numpy().astype(np.float64)
    )
    transform_error = float(
        np.max(
            np.abs(
                state_batch.transforms.detach().cpu().numpy().astype(np.float64)
                - reference.transforms
            )
        )
    )

    q_grad = torch.zeros(
        (1, chain.dof),
        device=target_device,
        dtype=torch.float32,
        requires_grad=True,
    )
    grad_state = forward_kinematics(chain, q_grad)
    ee_position = grad_state.transforms[:, -1, :3, 3]
    target = torch.tensor(
        [[0.5, 0.0, 0.5]], device=target_device, dtype=torch.float32
    )
    loss = torch.sum((ee_position - target) ** 2)
    loss.backward()

    reference_zero = reference_forward_kinematics(
        serial, np.zeros((1, chain.dof), dtype=np.float64)
    )
    residual = reference_zero.transforms[0, -1, :3, 3] - np.array(
        [0.5, 0.0, 0.5], dtype=np.float64
    )
    reference_gradient = (
        2.0
        * residual
        @ reference_zero.transform_jacobian[0, -1, :3, 3, :]
    )
    gradient = q_grad.grad.detach().cpu().numpy()[0]
    gradient_error = float(
        np.max(np.abs(gradient.astype(np.float64) - reference_gradient))
    )
    print("\nDifferentiable FK:")
    print(f"  Gradient w.r.t. joints: {q_grad.grad}")
    print("\nReference comparison:")
    print(f"  Maximum transform error: {transform_error:.3e}")
    print(f"  Maximum gradient error:  {gradient_error:.3e}")
    if transform_error > 1e-4 or gradient_error > 1e-4:
        raise RuntimeError("Metal/CPU FK did not match the independent reference")
    print("  PASS")
    return {
        "format": "curobo-metal-upstream-fk-example",
        "version": 1,
        "upstream_revision": PINNED_UPSTREAM,
        "robot": robot_cfg.name,
        "dof": chain.dof,
        "tool_frames": robot_cfg.tool_frames,
        "device": str(target_device),
        "dtype": "float32",
        "batch_size": batch_size,
        "seed": seed,
        "single_ee_position": ee_pose.position.detach().cpu().tolist(),
        "single_ee_quaternion_wxyz": ee_pose.quaternion.detach().cpu().tolist(),
        "loss_gradient": gradient.tolist(),
        "transform_max_abs": transform_error,
        "gradient_max_abs": gradient_error,
        "batch_ms": elapsed_ms,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_yaml", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = run(
        args.robot_yaml,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
