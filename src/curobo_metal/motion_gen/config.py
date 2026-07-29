"""Portable MotionGen configuration loading and compilation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from curobo_metal.backend import resolve_device
from curobo_metal.ops.costs import CollisionModel
from curobo_metal.ops.kinematics import KinematicChain
from curobo_metal.ops.trajectory import TrajectoryWeights
from curobo_metal.ops.whole_body import WholeBodyModel
from curobo_metal.reference import SerialRobot

from .types import UnsupportedMotionGenFeature


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


@dataclass(frozen=True)
class MotionGenConfig:
    chain: KinematicChain
    lower: torch.Tensor
    upper: torch.Tensor
    joint_names: tuple[str, ...]
    collision_model: CollisionModel | None = None
    steps: int = 32
    dt: float = 0.05
    interpolation_dt: float = 0.025
    trajectory_weights: TrajectoryWeights = TrajectoryWeights()
    max_trajectory_iterations: int = 100
    num_ik_seeds: int = 16
    max_ik_iterations: int = 300
    position_tolerance: float = 5e-3
    rotation_tolerance: float = 5e-2
    graph_sample_count: int = 128
    graph_seed: int = 0
    graph_k_neighbors: int = 12
    graph_edge_step: float = 0.05
    dynamics_aware: bool = False
    dynamics_model: WholeBodyModel | None = None
    dynamics_aware_options: Mapping[str, Any] | None = None

    @property
    def device(self) -> torch.device:
        return self.chain.device

    @property
    def dtype(self) -> torch.dtype:
        return self.chain.dtype

    @classmethod
    def load_from_robot_config(
        cls,
        robot: str | Path | Mapping[str, Any],
        world: Mapping[str, Any] | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        **overrides: Any,
    ) -> "MotionGenConfig":
        """Load the documented portable schema or an existing replay fixture.

        This intentionally does not parse URDF/USD/XRDF or upstream cuRobo YAML.
        """
        if isinstance(robot, (str, Path)):
            path = Path(robot)
            if path.suffix.lower() not in {".json"}:
                raise UnsupportedMotionGenFeature(
                    "only portable JSON configs are supported; URDF/USD/XRDF/YAML require cuRobo"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
        elif isinstance(robot, Mapping):
            data = dict(robot)
        else:
            raise TypeError("robot must be a JSON path or mapping")
        if "robot" in data:
            fixture = data
            data = dict(fixture["robot"])
            inputs = fixture.get("inputs", {})
            options = fixture.get("options", {})
            overrides = {
                "lower": inputs.get("lower"),
                "upper": inputs.get("upper"),
                "steps": inputs.get("steps"),
                "dt": inputs.get("dt"),
                "collision": inputs.get("collision"),
                "trajectory_weights": options.get("weights"),
                "max_trajectory_iterations": options.get("max_iterations"),
                **{k: v for k, v in overrides.items() if v is not None},
            }
        else:
            inputs = data.pop("motion_gen", {})
            overrides = {**inputs, **overrides}
        resolved = resolve_device(device)
        actual_dtype = dtype or (torch.float32 if resolved.type == "mps" else torch.float64)
        if resolved.type == "mps" and actual_dtype != torch.float32:
            raise TypeError("MPS MotionGen supports only float32")
        serial = SerialRobot.from_dict(data)
        chain = KinematicChain(serial, device=resolved, dtype=actual_dtype)
        lower_raw, upper_raw = overrides.pop("lower", None), overrides.pop("upper", None)
        if lower_raw is None or upper_raw is None:
            raise ValueError("portable MotionGen config requires lower and upper joint limits")
        lower = _tensor(lower_raw, device=resolved, dtype=actual_dtype)
        upper = _tensor(upper_raw, device=resolved, dtype=actual_dtype)
        if lower.shape != (serial.dof,) or upper.shape != (serial.dof,):
            raise ValueError("lower and upper must match robot DOF")
        collision_raw = overrides.pop("collision", None)
        collision = _collision_model(
            world if world is not None else collision_raw,
            device=resolved,
            dtype=actual_dtype,
        )
        weights_raw = overrides.pop("trajectory_weights", None)
        weights = (
            TrajectoryWeights(**weights_raw)
            if isinstance(weights_raw, Mapping)
            else weights_raw or TrajectoryWeights()
        )
        allowed = set(cls.__dataclass_fields__) - {
            "chain", "lower", "upper", "joint_names", "collision_model", "trajectory_weights"
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise TypeError(f"unsupported MotionGen configuration fields: {sorted(unknown)}")
        clean = {key: value for key, value in overrides.items() if value is not None}
        names = tuple(j.name for j in serial.joints if j.q_index is not None)
        return cls(chain, lower, upper, names, collision, trajectory_weights=weights, **clean)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], **kwargs: Any) -> "MotionGenConfig":
        return cls.load_from_robot_config(value, **kwargs)


def _collision_model(
    raw: Mapping[str, Any] | None, *, device: torch.device, dtype: torch.dtype
) -> CollisionModel | None:
    if raw is None:
        return None
    unsupported = set(raw) & {"mesh", "meshes", "voxel", "voxels", "esdf", "depth"}
    if unsupported:
        raise UnsupportedMotionGenFeature(
            f"MotionGen world supports primitive cuboids only, not {sorted(unsupported)}"
        )
    required = {"local_spheres", "link_indices"}
    if not required.issubset(raw):
        raise ValueError("collision config requires local_spheres and link_indices")
    tensor = lambda value: _tensor(value, device=device, dtype=dtype)
    return CollisionModel(
        local_spheres=tensor(raw["local_spheres"]),
        link_indices=torch.as_tensor(raw["link_indices"], device=device, dtype=torch.int64),
        self_pairs=(
            torch.as_tensor(raw["self_pairs"], device=device, dtype=torch.int64)
            if raw.get("self_pairs") is not None else None
        ),
        cuboid_centers=tensor(raw["cuboid_centers"]) if raw.get("cuboid_centers") is not None else None,
        cuboid_rotations=tensor(raw["cuboid_rotations"]) if raw.get("cuboid_rotations") is not None else None,
        cuboid_half_extents=tensor(raw["cuboid_half_extents"]) if raw.get("cuboid_half_extents") is not None else None,
        padding=float(raw.get("padding", 0.0)),
        activation_distance=float(raw.get("activation_distance", 0.0)),
        weight=float(raw.get("weight", 1.0)),
    )
