#!/usr/bin/env python3
"""Strictly validate and compare pinned CUDA outputs with committed Metal replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cuda_adapters import ADAPTERS
from .replay_registry import BY_ID, PIN


FORMAT = "curobo-metal-paired-replay"
VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _owned_file(folder: Path, name: object) -> Path:
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError(f"unsafe replay filename: {name!r}")
    path = folder / name
    if not path.is_file():
        raise ValueError(f"missing replay file: {path}")
    return path


def _validate_common(manifest: dict[str, Any], capability: str) -> None:
    case = BY_ID[capability]
    expected = {
        "format": FORMAT,
        "version": VERSION,
        "capability": capability,
        "operation": case.operation,
        "upstream_revision": PIN,
        "fallback_enabled": False,
        "equivalence_claimed": False,
        "tolerance": {"rtol": case.rtol, "atol": case.atol},
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{capability}: invalid {key}: {manifest.get(key)!r}; expected {value!r}"
            )


def _validate_required_evidence(
    manifest: dict[str, Any], outputs: np.lib.npyio.NpzFile, capability: str
) -> None:
    """Reject a paired report that silently drops invalid or edge coverage."""
    case = BY_ID[capability]
    expected = {
        "invalid": {
            "case": case.invalid_case,
            "output": "invalid_rejected",
            "executed": True,
        },
        "edge": {
            "case": case.edge_case,
            "output": "edge_observed",
            "executed": True,
        },
    }
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict) or any(
        evidence.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"{capability}: required invalid/edge evidence metadata is missing")
    for label, record in expected.items():
        key = record["output"]
        if key not in outputs.files:
            raise ValueError(f"{capability}: committed {label}-case evidence is missing")
        value = outputs[key]
        if value.shape != (1,) or value.dtype != np.int8 or int(value[0]) != 1:
            raise ValueError(f"{capability}: committed {label}-case evidence did not execute")


def _validate_prm_semantics(
    outputs: np.lib.npyio.NpzFile, inputs: np.lib.npyio.NpzFile, backend: str
) -> dict[str, dict[str, Any]]:
    """Validate outcome-equivalent PRM evidence without comparing roadmaps."""
    expected = {
        "success": np.asarray([True, True, False, False, False, True]),
        "status_code": np.asarray([0, 1, 2, 3, 4, 0], np.int8),
        "endpoint_ok": np.ones(6, np.bool_),
        "swept_valid": np.ones(6, np.bool_),
    }
    report: dict[str, dict[str, Any]] = {}
    for key, value in expected.items():
        passed = key in outputs.files and outputs[key].shape == value.shape and bool(
            np.array_equal(outputs[key], value)
        )
        report[key] = {"passed": passed, "semantic": True, "backend": backend}
    for key in ("batch_observed", "deterministic_repeat"):
        passed = (
            key in outputs.files and outputs[key].shape == (1,)
            and outputs[key].dtype == np.int8 and int(outputs[key][0]) == 1
        )
        report[key] = {"passed": passed, "semantic": True, "backend": backend}

    counts = outputs["point_count"] if "point_count" in outputs.files else np.array([])
    count_ok = (
        counts.shape == (6,) and np.issubdtype(counts.dtype, np.integer)
        and bool(np.all(counts[[0, 1]] >= 2)) and int(counts[5]) == 2
        and bool(np.all(counts[2:5] == 0))
    )
    report["point_count"] = {"passed": count_ok, "semantic": True, "backend": backend}

    costs = outputs["path_cost"] if "path_cost" in outputs.files else np.array([])
    starts, goals = inputs["graph_starts"], inputs["graph_goals"]
    direct = np.linalg.norm(goals - starts, axis=-1)
    cost_ok = (
        costs.shape == (6,) and np.issubdtype(costs.dtype, np.floating)
        and np.isfinite(costs[[0, 1, 5]]).all()
        and bool(np.isinf(costs[2:5]).all())
        and bool(costs[0] >= direct[0] - 1e-5)
        and bool(costs[1] > direct[1] + 1e-4)
        and abs(float(costs[5])) <= 1e-7
    )
    report["path_cost"] = {"passed": cost_ok, "semantic": True, "backend": backend}
    return report


def _validate_lbfgs_semantics(
    outputs: np.lib.npyio.NpzFile, inputs: np.lib.npyio.NpzFile, backend: str
) -> dict[str, dict[str, Any]]:
    """Validate optimizer outcomes without requiring identical iteration paths."""
    report: dict[str, dict[str, Any]] = {}

    def record(key: str, passed: bool) -> None:
        report[key] = {"passed": bool(passed), "semantic": True, "backend": backend}

    initial = inputs["lbfgs_initial"]
    target = inputs["lbfgs_target"]
    weight = inputs["lbfgs_weight"]
    lower, upper = inputs["lbfgs_lower"], inputs["lbfgs_upper"]
    solution = outputs["solution"] if "solution" in outputs.files else np.array([])
    expected_shape = initial.shape
    record("solution_shape", solution.shape == expected_shape and "solution_shape" in outputs.files
           and np.array_equal(outputs["solution_shape"], np.asarray(expected_shape, np.int64)))
    finite_solution = solution.shape == expected_shape and np.isfinite(solution).all()
    record("solution_finite", finite_solution)
    record("bounds_satisfied", finite_solution and np.all(solution >= lower - 1e-6)
           and np.all(solution <= upper + 1e-6) and "bounds_satisfied" in outputs.files
           and outputs["bounds_satisfied"].shape == (1,) and int(outputs["bounds_satisfied"][0]) == 1)
    projected = np.minimum(np.maximum(initial, lower), upper)
    record("fixed_terminal_satisfied", finite_solution
           and np.array_equal(solution[:, -1], projected[:, -1])
           and "fixed_terminal_satisfied" in outputs.files
           and outputs["fixed_terminal_satisfied"].shape == (1,)
           and int(outputs["fixed_terminal_satisfied"][0]) == 1)

    expected_initial = (0.5 * weight * (projected - target) ** 2).sum(axis=(-2, -1))
    expected_final = ((0.5 * weight * (solution - target) ** 2).sum(axis=(-2, -1))
                      if finite_solution else np.array([]))
    initial_value = outputs["initial_objective"] if "initial_objective" in outputs.files else np.array([])
    final_value = outputs["final_objective"] if "final_objective" in outputs.files else np.array([])
    record("initial_objective", initial_value.shape == expected_initial.shape
           and np.allclose(initial_value, expected_initial, rtol=2e-5, atol=2e-6))
    record("final_objective", finite_solution and final_value.shape == expected_final.shape
           and np.allclose(final_value, expected_final, rtol=2e-5, atol=2e-6))
    improvement = expected_final < expected_initial if finite_solution else np.array([], dtype=bool)
    record("objective_improved", improvement.shape == (initial.shape[0],) and improvement.all()
           and "objective_improved" in outputs.files
           and np.array_equal(outputs["objective_improved"], improvement))

    expected_norm = (np.linalg.norm((weight * (solution - target))[:, :-1].reshape(initial.shape[0], -1), axis=-1)
                     if finite_solution else np.array([]))
    gradient_norm = outputs["free_gradient_norm"] if "free_gradient_norm" in outputs.files else np.array([])
    record("free_gradient_norm", gradient_norm.shape == expected_norm.shape
           and np.allclose(gradient_norm, expected_norm, rtol=2e-5, atol=2e-6)
           and np.all(gradient_norm <= 2e-3))
    codes = outputs["convergence_code"] if "convergence_code" in outputs.files else np.array([])
    record("convergence_code", codes.shape == (initial.shape[0],) and codes.dtype == np.int8
           and np.all(codes == 0))
    for key in ("batch_observed", "reset_equivalent"):
        record(key, key in outputs.files and outputs[key].shape == (1,)
               and outputs[key].dtype == np.int8 and int(outputs[key][0]) == 1)
    record("nonfinite_status", "nonfinite_status" in outputs.files
           and outputs["nonfinite_status"].shape == (1,)
           and outputs["nonfinite_status"].dtype == np.int8
           and int(outputs["nonfinite_status"][0]) == 2)
    return report


def _validate_particle_semantics(
    outputs: np.lib.npyio.NpzFile, inputs: np.lib.npyio.NpzFile, backend: str
) -> dict[str, dict[str, Any]]:
    """Validate ES outcomes without requiring identical device RNG streams."""
    report: dict[str, dict[str, Any]] = {}

    def record(key: str, passed: bool) -> None:
        report[key] = {"passed": bool(passed), "semantic": True, "backend": backend}

    initial = inputs["particle_initial"]
    target = inputs["particle_target"]
    lower, upper = inputs["particle_lower"], inputs["particle_upper"]
    solution = outputs["solution"] if "solution" in outputs.files else np.array([])
    expected_shape = initial.shape
    shape_ok = (
        solution.shape == expected_shape
        and "solution_shape" in outputs.files
        and np.array_equal(outputs["solution_shape"], np.asarray(expected_shape, np.int64))
    )
    record("solution_shape", shape_ok)
    finite = shape_ok and bool(np.isfinite(solution).all())
    record("solution_finite", finite and "solution_finite" in outputs.files
           and outputs["solution_finite"].shape == (1,)
           and outputs["solution_finite"].dtype == np.int8
           and int(outputs["solution_finite"][0]) == 1)
    bounded = finite and bool(np.all(solution >= lower - 1e-6)) and bool(np.all(solution <= upper + 1e-6))
    record("bounds_satisfied", bounded and "bounds_satisfied" in outputs.files
           and outputs["bounds_satisfied"].shape == (1,)
           and int(outputs["bounds_satisfied"][0]) == 1)

    expected_initial = ((initial - target) ** 2).sum(axis=(-2, -1))
    expected_final = ((solution - target) ** 2).sum(axis=(-2, -1)) if finite else np.array([])
    initial_value = outputs["initial_objective"] if "initial_objective" in outputs.files else np.array([])
    final_value = outputs["final_objective"] if "final_objective" in outputs.files else np.array([])
    record("initial_objective", initial_value.shape == expected_initial.shape
           and np.allclose(initial_value, expected_initial, rtol=2e-5, atol=2e-6))
    record("final_objective", finite and final_value.shape == expected_final.shape
           and np.allclose(final_value, expected_final, rtol=2e-5, atol=2e-6))
    improved = expected_final < expected_initial if finite else np.array([], dtype=bool)
    record("objective_improved", improved.shape == (initial.shape[0],) and bool(improved.all())
           and "objective_improved" in outputs.files
           and np.array_equal(outputs["objective_improved"], improved))

    for key in (
        "deterministic_repeat", "batch_independent", "shift_observed", "fixed_sample_repeat"
    ):
        record(key, key in outputs.files and outputs[key].shape == (1,)
               and outputs[key].dtype == np.int8 and int(outputs[key][0]) == 1)

    initial_mean = outputs["multi_seed_initial_mean"] if "multi_seed_initial_mean" in outputs.files else np.array([])
    final_mean = outputs["multi_seed_final_mean"] if "multi_seed_final_mean" in outputs.files else np.array([])
    final_std = outputs["multi_seed_final_std"] if "multi_seed_final_std" in outputs.files else np.array([])
    rate = outputs["multi_seed_improvement_rate"] if "multi_seed_improvement_rate" in outputs.files else np.array([])
    record("multi_seed_initial_mean", initial_mean.shape == expected_initial.shape
           and np.allclose(initial_mean, expected_initial, rtol=2e-5, atol=2e-6))
    record("multi_seed_final_mean", final_mean.shape == expected_initial.shape
           and np.isfinite(final_mean).all() and bool(np.all(final_mean < expected_initial)))
    record("multi_seed_final_std", final_std.shape == expected_initial.shape
           and np.isfinite(final_std).all() and bool(np.all(final_std >= 0.0)))
    record("multi_seed_improvement_rate", rate.shape == expected_initial.shape
           and np.isfinite(rate).all() and bool(np.all(rate >= 0.8))
           and bool(np.all(rate <= 1.0)))
    return report


def compare_capability(
    metal_root: Path, cuda_root: Path, capability: str
) -> dict[str, Any]:
    """Validate provenance and compare every output tensor for one capability."""
    if capability not in ADAPTERS:
        raise ValueError(f"{capability}: no real CUDA adapter is registered")
    case = BY_ID[capability]
    metal_folder = metal_root / capability
    cuda_folder = cuda_root / capability
    metal_manifest = _load_json(metal_folder / "metal-manifest.json")
    cuda_manifest = _load_json(cuda_folder / "cuda-manifest.json")
    _validate_common(metal_manifest, capability)
    _validate_common(cuda_manifest, capability)
    if metal_manifest.get("backend") != "metal" or metal_manifest.get("device") != "mps":
        raise ValueError(f"{capability}: reference side is not fallback-disabled Metal")
    if cuda_manifest.get("backend") != "cuda" or cuda_manifest.get("device") != "cuda":
        raise ValueError(f"{capability}: candidate side is not CUDA")
    if cuda_manifest.get("status") != "complete":
        raise ValueError(f"{capability}: CUDA replay is not complete")
    runtime = cuda_manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("upstream_revision") != PIN
        or not isinstance(runtime.get("torch_cuda"), str)
        or not runtime["torch_cuda"]
        or not isinstance(runtime.get("cuda_device_name"), str)
        or not runtime["cuda_device_name"]
        or runtime.get("cuda_device_count", 0) < 1
        or not (
            isinstance(runtime.get("cuda_capability"), list)
            and len(runtime["cuda_capability"]) == 2
            and all(isinstance(value, int) for value in runtime["cuda_capability"])
        )
    ):
        raise ValueError(f"{capability}: incomplete CUDA runtime provenance")

    input_path = _owned_file(metal_folder, metal_manifest["input"]["file"])
    metal_output = _owned_file(metal_folder, metal_manifest["output"]["file"])
    cuda_output = _owned_file(cuda_folder, cuda_manifest["output"]["file"])
    if sha256(input_path) != metal_manifest["input"]["sha256"]:
        raise ValueError(f"{capability}: committed input hash mismatch")
    if cuda_manifest.get("input_sha256") != sha256(input_path):
        raise ValueError(f"{capability}: CUDA output consumed a different input")
    with np.load(input_path, allow_pickle=False) as inputs:
        if cuda_manifest.get("input_tensor_count") != len(inputs.files):
            raise ValueError(f"{capability}: CUDA input tensor count mismatch")
    if sha256(metal_output) != metal_manifest["output"]["sha256"]:
        raise ValueError(f"{capability}: committed Metal output hash mismatch")
    if sha256(cuda_output) != cuda_manifest["output"]["sha256"]:
        raise ValueError(f"{capability}: CUDA output hash mismatch")

    report: dict[str, Any] = {
        "capability": capability,
        "operation": case.operation,
        "rtol": case.rtol,
        "atol": case.atol,
        "tensors": {},
        "passed": True,
    }
    with (
        np.load(metal_output, allow_pickle=False) as metal,
        np.load(cuda_output, allow_pickle=False) as cuda,
    ):
        # Invalid-input and edge-case probes are executed and retained on the
        # Metal side as coverage evidence.  They deliberately are not part of
        # the upstream CUDA adapter payload: several pinned public APIs do
        # not expose an equivalent low-level probe without constructing
        # unrelated CUDA/Warp state.  Validate that evidence above, then
        # compare only the declared numerical operation outputs.
        evidence_outputs = {"invalid_rejected", "edge_observed"}
        metal_keys = set(metal.files) - evidence_outputs
        cuda_keys = set(cuda.files) - evidence_outputs
        if metal_keys != cuda_keys:
            raise ValueError(
                f"{capability}: output keys differ: "
                f"{sorted(metal_keys ^ cuda_keys)}"
            )
        _validate_required_evidence(metal_manifest, metal, capability)
        declared = cuda_manifest["output"].get("tensors")
        actual = {
            key: {"shape": list(cuda[key].shape), "dtype": str(cuda[key].dtype)}
            for key in sorted(cuda.files)
        }
        if declared != actual:
            raise ValueError(f"{capability}: CUDA tensor schema does not match manifest")
        if capability in {"graph.prm_planner", "optim.lbfgs", "optim.particle_evolution"}:
            # Planner roadmaps and optimizer iteration histories need not be
            # identical across devices. Require both backends to satisfy the
            # capability's observable outcome invariants instead.
            _validate_required_evidence(cuda_manifest, cuda, capability)
            with np.load(input_path, allow_pickle=False) as inputs:
                validator = {
                    "graph.prm_planner": _validate_prm_semantics,
                    "optim.lbfgs": _validate_lbfgs_semantics,
                    "optim.particle_evolution": _validate_particle_semantics,
                }[capability]
                metal_semantics = validator(metal, inputs, "metal")
                cuda_semantics = validator(cuda, inputs, "cuda")
            for key in sorted(metal_semantics):
                left, right = metal_semantics[key], cuda_semantics[key]
                item = {
                    "passed": bool(left["passed"] and right["passed"]),
                    "semantic": True,
                    "metal_passed": bool(left["passed"]),
                    "cuda_passed": bool(right["passed"]),
                }
                report["tensors"][key] = item
                report["passed"] = report["passed"] and item["passed"]
            return report
        for key in sorted(metal_keys):
            left, right = metal[key], cuda[key]
            item: dict[str, Any]
            if left.shape != right.shape or left.dtype != right.dtype:
                item = {
                    "passed": False,
                    "reason": "schema",
                    "metal_shape": list(left.shape),
                    "cuda_shape": list(right.shape),
                    "metal_dtype": str(left.dtype),
                    "cuda_dtype": str(right.dtype),
                }
            elif np.issubdtype(left.dtype, np.number):
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                passed = bool(
                    np.allclose(
                        left, right, rtol=case.rtol, atol=case.atol, equal_nan=True
                    )
                )
                item = {
                    "passed": passed,
                    "max_abs": float(delta.max(initial=0)),
                    "mean_abs": float(delta.mean()) if delta.size else 0.0,
                }
            else:
                passed = bool(np.array_equal(left, right))
                item = {"passed": passed, "exact": True}
            report["tensors"][key] = item
            report["passed"] = report["passed"] and item["passed"]
    return report


def compare_ready(metal_root: Path, cuda_root: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for capability in sorted(ADAPTERS):
        try:
            results.append(compare_capability(metal_root, cuda_root, capability))
        except ValueError as error:
            errors.append({"capability": capability, "error": str(error)})
    return {
        "format": "curobo-metal-paired-report",
        "version": 1,
        "upstream_revision": PIN,
        "required_capabilities": sorted(ADAPTERS),
        "results": results,
        "errors": errors,
        "passed": not errors
        and len(results) == len(ADAPTERS)
        and all(item["passed"] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metal-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_ready(args.metal_root, args.cuda_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
