#!/usr/bin/env python3
"""Generate deterministic portable inputs and local output bundles."""

from __future__ import annotations

import argparse
import io
import hashlib
import json
import os
import platform
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

from curobo_metal.ops.costs import pose_cost
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics
from curobo_metal.ops.trajectory import minimum_jerk_trajectory
from curobo_metal.ops.trajectory.dynamics_aware import bspline_matrices
from curobo_metal.ops.world_collision import Mesh, VoxelGrid, mesh_distance, query_esdf
from curobo_metal.optim import LBFGSConfig, ParticleConfig, lbfgs_optimize, particle_optimize
from curobo_metal.reference import SerialRobot
from curobo_metal.reference import TreeRobot
from curobo_metal.types import DeviceCfg, JointState, MotionGenStatus, PlanningResult, Pose
from curobo_metal.ops.collision import sphere_sphere_signed_distance
from curobo_metal.ops.whole_body import WholeBodyModel, inverse_dynamics

from .replay_registry import CASES, PIN, Case

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/parity/replay"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npz(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write byte-deterministic, uncompressed, pickle-free NPZ."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(tensors):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(tensors[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def inputs() -> dict[str, np.ndarray]:
    inertial = (ROOT / "tests/fixtures/whole_body/branched_toy.json").read_bytes()
    return {
        "q": np.array([[0.0, 0.25], [-0.5, 0.75]], np.float32),
        "points": np.array([[0.2, 0.2, 0.5], [2.0, 0.0, 0.0]], np.float32),
        "pose": np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], np.float32),
        "trajectory_endpoints": np.array([[0.0, 0.0], [1.0, -0.5]], np.float32),
        "invalid_shape": np.zeros((1, 3), np.float32),
        "singleton": np.array([[0.125]], np.float32),
        "empty": np.empty((0, 2), np.float32),
        "inertial_case_json": np.frombuffer(inertial, np.uint8),
    }


def _tensor(data, device):
    return torch.as_tensor(data, device=device)


def probe(case: Case, raw: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    q = _tensor(raw["q"], device)
    if case.probe == "serialization":
        payload = json.dumps({"joint_names": ["j0", "j1"], "mimic": {"j1": [0.5, 0.1]}}, sort_keys=True)
        return {"serialized_utf8": np.frombuffer(payload.encode(), np.uint8)}
    if case.probe == "device":
        cfg = DeviceCfg(device, torch.float32)
        return {"value": cfg.to_device(raw["singleton"]).cpu().numpy(), "device_code": np.array([0 if device == "cpu" else 1], np.int32)}
    if case.probe == "pose":
        value = Pose.from_list(raw["pose"][0].tolist(), DeviceCfg(device, torch.float32))
        points = value.transform_points(_tensor(raw["points"], device))
        return {"matrix": value.get_matrix().detach().cpu().numpy(), "points": points.detach().cpu().numpy()}
    if case.probe == "joint_state":
        state = JointState.from_position(q, ["j0", "j1"]).finite_difference(0.25)
        return {"position": state.position.cpu().numpy(), "velocity": state.velocity.cpu().numpy()}
    if case.probe == "result":
        result = PlanningResult(True, MotionGenStatus.SUCCESS, JointState.from_position(q, ["j0", "j1"]))
        return {"success": np.array([bool(result.success)]), "status_utf8": np.frombuffer(str(result.status.value).encode(), np.uint8)}
    if case.probe == "fk":
        robot = SerialRobot.from_dict({"name": "two_link", "joints": [
            {"name": "j0", "type": "revolute", "axis": [0, 0, 1], "origin": {"xyz": [1, 0, 0]}},
            {"name": "j1", "type": "revolute", "axis": [0, 1, 0], "origin": {"xyz": [1, 0, 0]}},
        ]})
        x = q.clone().requires_grad_(True)
        out = forward_kinematics(KinematicChain(robot, device=device), x)
        out.transforms.sum().backward()
        return {"transforms": out.transforms.detach().cpu().numpy(), "jacobian": out.geometric_jacobian.detach().cpu().numpy(), "input_gradient": x.grad.cpu().numpy()}
    if case.probe == "sphere":
        spheres = _tensor(np.array([[0, 0, 0, .5], [.75, 0, 0, .5]], np.float32), device).requires_grad_(True)
        out = sphere_sphere_signed_distance(spheres, _tensor(np.array([[0, 1]], np.int64), device))
        out.distances.sum().backward()
        return {"distance": out.distances.detach().cpu().numpy(), "input_gradient": spheres.grad.cpu().numpy()}
    if case.probe == "mesh":
        vertices = _tensor(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), device)
        mesh = Mesh(vertices, _tensor(np.array([[0, 1, 2]], np.int64), device), False)
        out = mesh_distance(_tensor(raw["points"], device), [mesh],
                            torch.zeros((1, 1, 3), device=device),
                            torch.eye(3, device=device).reshape(1, 1, 3, 3),
                            signed=False)
        return {"distance": out.reduced_distance.cpu().numpy(), "gradient": out.reduced_gradient.cpu().numpy()}
    if case.probe == "voxel":
        values = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 2, 2)
        grid = VoxelGrid(values, 1.0, torch.zeros(3, device=device), torch.eye(3, device=device), 99.0)
        out = query_esdf(_tensor(np.array([[.1, .1, .1], [-.25, .25, .25]], np.float32), device), [[grid]])
        return {"distance": out.distance.cpu().numpy(), "valid": out.valid.cpu().numpy(), "winner": out.winning_grid.cpu().numpy()}
    if case.probe == "cost":
        x = q.repeat(1, 3).clone().requires_grad_(True)
        value = pose_cost(x)
        value.sum().backward()
        return {"value": value.detach().cpu().numpy(), "input_gradient": x.grad.cpu().numpy()}
    if case.probe in {"particle", "lbfgs"}:
        objective = lambda x: ((x - .2) ** 2).sum(-1)
        if case.probe == "particle":
            result = particle_optimize(objective, q, config=ParticleConfig(iterations=3, particles=8, elite_count=2, seed=7))
        else:
            result = lbfgs_optimize(objective, q, config=LBFGSConfig(iterations=5))
        return {"solution": result.solution.detach().cpu().numpy(), "objective": result.objective.detach().cpu().numpy(), "converged": result.converged.cpu().numpy()}
    if case.probe == "trajectory":
        out = minimum_jerk_trajectory(q[0], q[1], 5)
        return {"trajectory": out.detach().cpu().numpy(), "status_utf8": np.frombuffer(b"success", np.uint8)}
    if case.probe == "bspline":
        out = bspline_matrices(6, 9, degree=3, dtype=torch.float32, device=torch.device(device))
        return {"position_matrix": out.position.cpu().numpy(), "velocity_matrix": out.velocity.cpu().numpy()}
    if case.probe == "graph":
        from curobo_metal.ops.graph_planning import interpolate_edge
        out = interpolate_edge(q[0], q[1], .25)
        return {"path": out.cpu().numpy(), "status_utf8": np.frombuffer(b"success", np.uint8)}
    if case.probe == "dynamics":
        inertial = json.loads(raw["inertial_case_json"].tobytes().decode())
        model = WholeBodyModel(TreeRobot.from_dict(inertial["robot"]), device=device)
        state = inertial["inputs"]
        position = _tensor(np.asarray(state["q"], np.float32), device).requires_grad_(True)
        velocity = _tensor(np.asarray(state["qd"], np.float32), device).requires_grad_(True)
        acceleration = _tensor(np.asarray(state["qdd"], np.float32), device).requires_grad_(True)
        torque = inverse_dynamics(model, position, velocity, acceleration).torque
        gradients = torch.autograd.grad(torque.sum(), (position, velocity, acceleration))
        return {"torque": torque.detach().cpu().numpy(),
                "position_gradient": gradients[0].cpu().numpy(),
                "velocity_gradient": gradients[1].cpu().numpy(),
                "acceleration_gradient": gradients[2].cpu().numpy(),
                "status_utf8": np.frombuffer(b"success", np.uint8)}
    raise AssertionError(case.probe)


def generate(output: Path, device: str) -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    output.mkdir(parents=True, exist_ok=True)
    index = {"format": "curobo-metal-paired-replay", "version": 1, "upstream_revision": PIN, "cases": []}
    raw = inputs()
    for case in CASES:
        folder = output / case.capability
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        input_path, output_path = folder / "inputs.npz", folder / "metal-outputs.npz"
        save_npz(input_path, raw)
        save_npz(output_path, probe(case, raw, device))
        manifest = {
            "format": "curobo-metal-paired-replay", "version": 1,
            "capability": case.capability, "operation": case.operation,
            "backend": "metal" if device == "mps" else "cpu-reference",
            "device": device, "fallback_enabled": False, "upstream_revision": PIN,
            "input": {"file": input_path.name, "sha256": sha(input_path)},
            "output": {"file": output_path.name, "sha256": sha(output_path)},
            "tolerance": {"rtol": case.rtol, "atol": case.atol},
            "evidence": {"gradient": "input_gradient" in np.load(output_path).files,
                         "status": any(k.startswith("status") for k in np.load(output_path).files),
                         "invalid_case": case.invalid_case,
                         "probe_scope": case.probe},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__},
            "equivalence_claimed": False,
        }
        (folder / "metal-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        index["cases"].append({"capability": case.capability, "manifest": f"{case.capability}/metal-manifest.json", "manifest_sha256": sha(folder / "metal-manifest.json")})
    (output / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    generate(args.output, args.device)


if __name__ == "__main__":
    main()
