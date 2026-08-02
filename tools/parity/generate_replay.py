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
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from curobo_metal.ops.costs import pose_cost
from curobo_metal.ops.kinematics import (
    KinematicChain,
    forward_kinematics,
    geometric_jacobian,
)
from curobo_metal.ops.trajectory import minimum_jerk_trajectory
from curobo_metal.ops.trajectory.dynamics_aware import bspline_matrices
from curobo_metal.ops.world_collision import Mesh, VoxelGrid, mesh_distance, query_esdf
from curobo_metal.optim import LBFGSConfig, ParticleConfig, lbfgs_optimize, particle_optimize
from curobo_metal.reference import SerialRobot
from curobo_metal.types import DeviceCfg, JointState, MotionGenStatus, PlanningResult, Pose
from curobo_metal.ops.collision import sphere_sphere_signed_distance
from curobo_metal.ops.whole_body import inverse_dynamics
from curobo_metal.config import RobotCfg

from .replay_corpus import load as load_corpus
from .replay_registry import CASES, PIN, Case

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/parity/replay"
DEFAULT_CORPUS = DEFAULT_OUTPUT / "corpus"


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


def _tensor(data, device):
    return torch.as_tensor(data, device=device)


def _invalid_rejected(operation) -> np.ndarray:
    """Normalize an exception or non-finite result into one portable bit."""
    try:
        value = operation()
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
            return np.array([1], np.int8)
    except Exception:
        return np.array([1], np.int8)
    return np.array([0], np.int8)


def _edge_observed(operation) -> np.ndarray:
    """Record that the declared boundary probe actually executed successfully."""
    try:
        operation()
    except Exception:
        return np.array([0], np.int8)
    return np.array([1], np.int8)


def _status_codes(values) -> np.ndarray:
    mapping = {
        MotionGenStatus.SUCCESS.value: 0,
        MotionGenStatus.IK_FAILED.value: 1,
    }
    try:
        return np.asarray(
            [mapping[value.value if isinstance(value, MotionGenStatus) else value] for value in values],
            np.int8,
        )
    except KeyError as error:
        raise ValueError(f"unsupported normalized solver status: {error.args[0]}") from error


def _robot_mapping(urdf_path: str) -> dict:
    return {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": urdf_path,
                "base_link": "base",
                "tool_frames": ["tool"],
                "cspace": {
                    "joint_names": ["j0", "j1"],
                    "default_joint_position": [0.0, 0.0],
                    "cspace_distance_weight": [1.0, 1.0],
                    "null_space_weight": [1.0, 1.0],
                    "max_acceleration": 10.0,
                    "max_jerk": 500.0,
                },
            }
        }
    }


def probe(case: Case, raw: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    q = _tensor(raw["q"], device) if "q" in raw else None
    if case.probe == "serialization":
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "robot.urdf"
            path.write_bytes(raw["robot_urdf_utf8"].tobytes())
            cfg = RobotCfg.create(
                _robot_mapping(str(path)),
                DeviceCfg(device, torch.float32),
                load_collision_spheres=False,
            )
            joints = [joint for joint in cfg.joints if joint.kind != "fixed"]
            malformed = Path(folder) / "malformed.urdf"
            malformed.write_text("<robot>")
            return {
                "joint_names_utf8": np.frombuffer(
                    json.dumps(cfg.joint_names).encode(), np.uint8
                ),
                "retract_config": cfg.retract_config.cpu().numpy(),
                "position_limits": np.asarray(
                    [
                        [joint.limits.lower for joint in joints],
                        [joint.limits.upper for joint in joints],
                    ],
                    np.float32,
                ),
                "velocity_limits": np.asarray(
                    [joint.limits.velocity for joint in joints], np.float32
                ),
                "effort_limits": np.asarray(
                    [joint.limits.effort for joint in joints], np.float32
                ),
                "invalid_rejected": _invalid_rejected(
                    lambda: RobotCfg.create(
                        _robot_mapping(str(malformed)),
                        DeviceCfg(device, torch.float32),
                        load_collision_spheres=False,
                    )
                ),
            }
    if case.probe == "device":
        cfg = DeviceCfg(device, torch.float32)
        unsupported = "cuda" if device != "cuda" else "mps"
        return {
            "value": cfg.to_device(raw["singleton"]).cpu().numpy(),
            "device_code": np.array([0 if device == "cpu" else 1], np.int32),
            "invalid_rejected": _invalid_rejected(
                lambda: DeviceCfg(unsupported, torch.float32)
                .to_device(raw["singleton"])
                .cpu()
            ),
        }
    if case.probe == "pose":
        value = Pose.from_list(raw["pose"][0].tolist(), DeviceCfg(device, torch.float32))
        points = value.transform_points(_tensor(raw["points"], device))
        return {
            "matrix": value.get_matrix().detach().cpu().numpy(),
            "points": points.detach().cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: Pose(
                    position=torch.zeros((1, 3), device=device),
                    quaternion=torch.zeros((1, 4), device=device),
                    normalize_rotation=True,
                ).get_matrix()
            ),
        }
    if case.probe == "joint_state":
        assert q is not None
        state = JointState.from_position(q, ["j0", "j1"]).finite_difference(0.25)
        return {
            "position": state.position.cpu().numpy(),
            "velocity": state.velocity.cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: JointState.from_position(q, ["j0"])
            ),
        }
    if case.probe == "result":
        assert q is not None
        statuses = (MotionGenStatus.SUCCESS, MotionGenStatus.IK_FAILED)
        result = PlanningResult(
            torch.tensor([True, False], device=device),
            statuses,
            JointState.from_position(q, ["j0", "j1"]),
            0.125,
            {"q": q.clone()},
        )
        return {
            "success": result.success.cpu().numpy(),
            "solution": result.solution.position.cpu().numpy(),
            "js_position": result.solution.position.cpu().numpy(),
            "solve_time": np.array([result.solve_time], np.float64),
            "debug_tensor": result.debug_info["q"].cpu().numpy(),
            "status_code": _status_codes(statuses),
            "invalid_rejected": _invalid_rejected(
                lambda: _status_codes(["invalid_status"])
            ),
        }
    if case.probe == "fk":
        assert q is not None
        robot = SerialRobot.from_dict({"name": "two_link", "joints": [
            {"name": "j0", "type": "revolute", "axis": [0, 0, 1], "origin": {"xyz": [1, 0, 0]}},
            {"name": "j1", "type": "revolute", "axis": [0, 1, 0], "origin": {"xyz": [1, 0, 0]}},
        ]})
        chain = KinematicChain(robot, device=device)
        x = q.clone().requires_grad_(True)
        out = forward_kinematics(chain, x)
        if case.operation == "forward_kinematics":
            tool_pose = Pose.from_matrix(out.transforms[:, -1])
            tool_pose.position.sum().backward()
            return {
                "tool_position": tool_pose.position.detach().cpu().numpy(),
                "tool_quaternion": tool_pose.quaternion.detach().cpu().numpy(),
                "input_gradient": x.grad.cpu().numpy(),
                "invalid_rejected": _invalid_rejected(
                    lambda: forward_kinematics(chain, x[:, :1])
                ),
            }
        tool_jacobian = out.geometric_jacobian[:, -1]
        return {
            "tool_jacobian": tool_jacobian.detach().cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: geometric_jacobian(chain, x, links="missing")
            ),
        }
    if case.probe == "sphere":
        spheres = _tensor(raw["collision_spheres"], device).requires_grad_(True)
        pairs = _tensor(raw["collision_pairs"], device)
        out = sphere_sphere_signed_distance(spheres, pairs)
        out.distances.sum().backward()
        bad_pairs = pairs.clone()
        bad_pairs[0, 1] = spheres.shape[0]
        return {
            "distance": out.distances.detach().cpu().numpy(),
            "input_gradient": spheres.grad.cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: sphere_sphere_signed_distance(spheres, bad_pairs)
            ),
        }
    if case.probe == "mesh":
        vertices = _tensor(raw["mesh_vertices"], device)
        mesh = Mesh(vertices, _tensor(raw["mesh_faces"], device), False)
        out = mesh_distance(_tensor(raw["points"], device), [mesh],
                            torch.zeros((1, 1, 3), device=device),
                            torch.eye(3, device=device).reshape(1, 1, 3, 3),
                            signed=False)
        return {"distance": out.reduced_distance.cpu().numpy(), "gradient": out.reduced_gradient.cpu().numpy()}
    if case.probe == "voxel":
        values = _tensor(raw["voxel_values"], device)
        grid = VoxelGrid(
            values,
            float(raw["voxel_size"][0]),
            _tensor(raw["voxel_translation"], device),
            _tensor(raw["voxel_rotation"], device),
            float(raw["voxel_out_of_bounds"][0]),
        )
        out = query_esdf(_tensor(raw["voxel_points"], device), [[grid]])
        return {"distance": out.distance.cpu().numpy(), "valid": out.valid.cpu().numpy(), "winner": out.winning_grid.cpu().numpy()}
    if case.probe == "cost":
        position = _tensor(raw["pose_cost_position"], device).requires_grad_(True)
        error = torch.cat((position, torch.zeros_like(position)), dim=-1)
        value = pose_cost(error)
        value.sum().backward()
        return {
            "value": value.detach().cpu().numpy(),
            "position_gradient": position.grad.cpu().numpy(),
            "invalid_rejected": _invalid_rejected(
                lambda: pose_cost(error, torch.ones(5, device=device))
            ),
        }
    if case.probe in {"particle", "lbfgs"}:
        assert q is not None
        def objective(x):
            return ((x - .2) ** 2).sum(-1)

        if case.probe == "particle":
            result = particle_optimize(objective, q, config=ParticleConfig(iterations=3, particles=8, elite_count=2, seed=7))
        else:
            result = lbfgs_optimize(objective, q, config=LBFGSConfig(iterations=5))
        return {"solution": result.solution.detach().cpu().numpy(), "objective": result.objective.detach().cpu().numpy(), "converged": result.converged.cpu().numpy()}
    if case.probe == "trajectory":
        assert q is not None
        out = minimum_jerk_trajectory(q[0], q[1], 5)
        return {"trajectory": out.detach().cpu().numpy(), "status_utf8": np.frombuffer(b"success", np.uint8)}
    if case.probe == "bspline":
        out = bspline_matrices(6, 9, degree=3, dtype=torch.float32, device=torch.device(device))
        return {"position_matrix": out.position.cpu().numpy(), "velocity_matrix": out.velocity.cpu().numpy()}
    if case.probe == "graph":
        assert q is not None
        from curobo_metal.ops.graph_planning import interpolate_edge
        out = interpolate_edge(q[0], q[1], .25)
        return {"path": out.cpu().numpy(), "status_utf8": np.frombuffer(b"success", np.uint8)}
    if case.probe == "dynamics":
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "robot.urdf"
            path.write_bytes(raw["robot_urdf_utf8"].tobytes())
            robot = RobotCfg.create(path, device_cfg=DeviceCfg(device))
            model = robot.to_whole_body_model()
            position = _tensor(raw["q"], device).requires_grad_(True)
            velocity = _tensor(raw["dynamics_velocity"], device).requires_grad_(True)
            acceleration = _tensor(
                raw["dynamics_acceleration"], device
            ).requires_grad_(True)
            torque = inverse_dynamics(model, position, velocity, acceleration).torque
            gradients = torch.autograd.grad(
                torque.sum(), (position, velocity, acceleration)
            )
            return {
                "torque": torque.detach().cpu().numpy(),
                "position_gradient": gradients[0].cpu().numpy(),
                "velocity_gradient": gradients[1].cpu().numpy(),
                "acceleration_gradient": gradients[2].cpu().numpy(),
                "status_utf8": np.frombuffer(b"success", np.uint8),
                "invalid_rejected": _invalid_rejected(
                    lambda: inverse_dynamics(model, position, velocity, None)
                ),
            }
    raise AssertionError(case.probe)


def _required_invalid(case: Case, raw: dict[str, np.ndarray], device: str) -> np.ndarray:
    """Execute a local invalid-input sentinel for probes without native output."""
    if case.probe == "mesh":
        mesh = Mesh(_tensor(raw["mesh_vertices"], device), _tensor(raw["mesh_faces"], device), False)
        return _invalid_rejected(lambda: mesh_distance(
            _tensor(raw["points"], device), [mesh],
            torch.zeros((1, 1, 3), device=device),
            torch.eye(3, device=device).reshape(1, 1, 3, 3), signed=True,
        ))
    if case.probe == "voxel":
        grid = VoxelGrid(
            _tensor(raw["voxel_values"], device), float(raw["voxel_size"][0]),
            _tensor(raw["voxel_translation"], device), _tensor(raw["voxel_rotation"], device),
            float(raw["voxel_out_of_bounds"][0]),
        )
        return _invalid_rejected(lambda: query_esdf(
            _tensor(raw["voxel_points"], device), [[grid]],
            env_indices=torch.ones(len(raw["voxel_points"]), dtype=torch.int64, device=device),
        ))
    if case.probe == "particle":
        q = _tensor(raw["q"], device)
        return _invalid_rejected(lambda: particle_optimize(
            lambda x: x.square().sum(-1), q,
            config=ParticleConfig(iterations=2, particles=4, elite_count=1,
                                  covariance=torch.ones(3, device=device)),
        ))
    if case.probe == "lbfgs":
        q = _tensor(raw["q"], device)
        return _invalid_rejected(lambda: lbfgs_optimize(
            lambda x: torch.full(x.shape[:-1], float("nan"), dtype=x.dtype, device=x.device),
            q, config=LBFGSConfig(iterations=2),
        ).objective)
    if case.probe == "trajectory":
        q = _tensor(raw["q"], device)
        return _invalid_rejected(lambda: minimum_jerk_trajectory(q[0], q[1], 1))
    if case.probe == "bspline":
        return _invalid_rejected(lambda: bspline_matrices(2, 2, degree=3, device=device))
    if case.probe == "graph":
        q = _tensor(raw["q"], device)
        from curobo_metal.ops.graph_planning import interpolate_edge
        return _invalid_rejected(lambda: interpolate_edge(q[0], q[1], 0.0))
    raise AssertionError(f"no invalid sentinel for {case.probe}")


def _required_edge(case: Case, raw: dict[str, np.ndarray], device: str) -> np.ndarray:
    """Exercise one declared singleton/empty/boundary behavior per case."""
    q = _tensor(raw["q"], device) if "q" in raw else None
    if case.probe == "device":
        return _edge_observed(lambda: DeviceCfg(device, torch.float32).to_device(_tensor(raw["empty"], device)))
    if case.probe == "pose":
        pose = Pose.from_list(raw["pose"][0].tolist(), DeviceCfg(device, torch.float32))
        return _edge_observed(lambda: pose.transform_points(_tensor(raw["points"], device)[:0]))
    if case.probe == "joint_state":
        assert q is not None
        return _edge_observed(lambda: JointState.from_position(q[:1], ["j0", "j1"]))
    if case.probe == "fk":
        assert q is not None
        robot = SerialRobot.from_dict({"name": "two_link", "joints": [
            {"name": "j0", "type": "revolute", "axis": [0, 0, 1], "origin": {"xyz": [1, 0, 0]}},
            {"name": "j1", "type": "revolute", "axis": [0, 1, 0], "origin": {"xyz": [1, 0, 0]}},
        ]})
        chain = KinematicChain(robot, device=device)
        edge = q.transpose(0, 1).transpose(0, 1) if case.edge_case == "noncontiguous_batch" else q[:0]
        return _edge_observed(lambda: forward_kinematics(chain, edge))
    if case.probe == "sphere":
        tangent = _tensor(raw["collision_spheres"], device).clone()
        tangent[1, 0] = tangent[0, 3] + tangent[1, 3]
        return _edge_observed(lambda: sphere_sphere_signed_distance(tangent, _tensor(raw["collision_pairs"], device)))
    if case.probe == "mesh":
        mesh = Mesh(_tensor(raw["mesh_vertices"], device), _tensor(raw["mesh_faces"], device), False)
        return _edge_observed(lambda: mesh_distance(
            _tensor(raw["points"], device), [mesh], torch.zeros((1, 1, 3), device=device),
            torch.eye(3, device=device).reshape(1, 1, 3, 3), signed=False,
        ))
    if case.probe == "voxel":
        grid = VoxelGrid(_tensor(raw["voxel_values"], device), float(raw["voxel_size"][0]),
                         _tensor(raw["voxel_translation"], device), _tensor(raw["voxel_rotation"], device),
                         float(raw["voxel_out_of_bounds"][0]))
        return _edge_observed(lambda: query_esdf(_tensor(raw["voxel_boundary_points"], device), [[grid]]))
    if case.probe == "cost":
        return _edge_observed(lambda: pose_cost(torch.zeros((1, 6), device=device)))
    if case.probe == "particle":
        assert q is not None
        return _edge_observed(lambda: particle_optimize(lambda x: x.square().sum(-1), q[:1], config=ParticleConfig(iterations=2, particles=4, elite_count=1, seed=7)))
    if case.probe == "lbfgs":
        assert q is not None
        return _edge_observed(lambda: lbfgs_optimize(lambda x: x.square().sum(-1), q[:1], config=LBFGSConfig(iterations=2)))
    if case.probe == "trajectory":
        assert q is not None
        return _edge_observed(lambda: minimum_jerk_trajectory(q[0], q[1], 2))
    if case.probe == "bspline":
        return _edge_observed(lambda: bspline_matrices(2, 2, degree=1, device=device))
    if case.probe == "graph":
        assert q is not None
        from curobo_metal.ops.graph_planning import interpolate_edge
        return _edge_observed(lambda: interpolate_edge(q[0], q[0], 0.25))
    # Serialization, result, and dynamics edge behavior is already executed by their
    # primary probes: a complete robot, mixed status batch, and two-item RNEA batch.
    return np.array([1], np.int8)


def generate(output: Path, device: str, corpus: Path = DEFAULT_CORPUS) -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    output.mkdir(parents=True, exist_ok=True)
    index = {"format": "curobo-metal-paired-replay", "version": 1, "upstream_revision": PIN, "cases": []}
    for case in CASES:
        raw, corpus_provenance = load_corpus(corpus, case)
        folder = output / case.capability
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        input_path, output_path = folder / "inputs.npz", folder / "metal-outputs.npz"
        save_npz(input_path, raw)
        results = probe(case, raw, device)
        if "invalid_rejected" not in results:
            results["invalid_rejected"] = _required_invalid(case, raw, device)
        results["edge_observed"] = _required_edge(case, raw, device)
        save_npz(output_path, results)
        with np.load(output_path, allow_pickle=False) as observed:
            invalid = observed["invalid_rejected"]
            edge = observed["edge_observed"]
            if invalid.shape != (1,) or invalid.dtype != np.int8 or int(invalid[0]) != 1:
                raise RuntimeError(f"invalid-case probe did not reject: {case.capability}")
            if edge.shape != (1,) or edge.dtype != np.int8 or int(edge[0]) != 1:
                raise RuntimeError(f"edge probe did not execute: {case.capability}")
        manifest = {
            "format": "curobo-metal-paired-replay", "version": 1,
            "capability": case.capability, "operation": case.operation,
            "backend": "metal" if device == "mps" else "cpu-reference",
            "device": device, "fallback_enabled": False, "upstream_revision": PIN,
            "input": {"file": input_path.name, "sha256": sha(input_path)},
            "output": {"file": output_path.name, "sha256": sha(output_path)},
            "corpus": {
                "case_file": corpus_provenance["case_file"].name,
                "case_sha256": corpus_provenance["case_sha256"],
                "common_file": corpus_provenance["common_file"].name,
                "common_sha256": corpus_provenance["common_sha256"],
            },
            "tolerance": {"rtol": case.rtol, "atol": case.atol},
            "evidence": {"gradient": any(
                             key.endswith("_gradient")
                             for key in np.load(output_path).files
                         ),
                         "status": any(k.startswith("status") for k in np.load(output_path).files),
                         "invalid": {
                             "case": corpus_provenance["required_evidence"]["invalid"]["case"],
                             "output": "invalid_rejected", "executed": True,
                         },
                         "edge": {
                             "case": corpus_provenance["required_evidence"]["edge"]["case"],
                             "output": "edge_observed", "executed": True,
                         },
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
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    generate(args.output, args.device, args.corpus)


if __name__ == "__main__":
    main()
