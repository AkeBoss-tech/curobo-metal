"""Portable self-collision inspection and artifact export for robot configurations.

The pinned debugger is a CUDA/Warp diagnostic shell.  This implementation
keeps its useful model-inspection workflow on CPU/MPS with production portable
kinematics and sphere collision data; Viser, USD/Isaac, and raw Warp buffers
remain intentionally outside this module's contract.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.robot.kinematics import Kinematics, KinematicsCfg
from curobo._src.robot.types import SelfCollisionKinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


class RobotDebugger:
    """Inspect a portable robot's configured self-collision sphere model."""

    def __init__(self, config_path: str, device_cfg: Optional[DeviceCfg] = None) -> None:
        if not isinstance(config_path, str) or not config_path:
            raise ValueError("config_path must be a nonempty robot YAML path")
        self.config_path = config_path
        self.device_cfg = device_cfg or DeviceCfg()
        self._robot_config = KinematicsCfg.from_robot_yaml_file(config_path, device_cfg=self.device_cfg)
        self._robot_model = Kinematics(self._robot_config)
        self._self_collision_config = self._compile_self_collision_config()
        self._collision_cost = SelfCollisionCost(SelfCollisionCostCfg(
            weight=1.0,
            device_cfg=self.device_cfg,
            self_collision_kin_config=self._self_collision_config,
            store_pair_distance=True,
        ))
        self._last_result: Optional[dict[str, Any]] = None
        self._check_count = 0

    @classmethod
    def from_xrdf(cls, *args, **kwargs) -> "RobotDebugger":
        del args, kwargs
        raise NotImplementedError(
            "direct XRDF debugging requires the external XRDF-to-YAML conversion path"
        )

    def _compile_self_collision_config(self) -> SelfCollisionKinematicsCfg:
        """Compile link-level ignores/buffers into portable indexed sphere pairs."""
        params = self._robot_config.kinematics_config
        robot = params.robot_cfg
        sphere_count = params.total_spheres
        if sphere_count == 0:
            return SelfCollisionKinematicsCfg(
                num_spheres=0,
                sphere_padding=torch.empty((0,), **self.device_cfg.as_torch_dict()),
                collision_pairs=torch.empty((0, 2), device=self.device_cfg.device, dtype=torch.int64),
            )
        names = list(dict.fromkeys(
            robot.collision_link_names or [sphere.link_name for sphere in robot.collision_spheres]
        ))
        link_map = params.link_name_to_idx_map
        # Configurations can retain named attachment placeholders that have no
        # current link in the compiled portable tree.  They own no active
        # sphere and therefore cannot form a pair in this debugger session.
        names = [name for name in names if name in link_map]
        return SelfCollisionKinematicsCfg.create_from_link_pairs(
            names,
            {name: link_map[name] for name in names},
            robot.self_collision_ignore,
            robot.self_collision_buffer,
            params.link_spheres[0],
            params.link_sphere_idx_map,
            self.device_cfg,
        )

    def _joint_tensor(self, joint_position: Union[List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(joint_position, torch.Tensor):
            value = joint_position.to(**self.device_cfg.as_torch_dict())
        else:
            value = self.device_cfg.to_device(joint_position)
        if value.numel() != self._robot_config.dof:
            raise ValueError(
                f"joint_position must have {self._robot_config.dof} elements, got {value.numel()}"
            )
        value = value.reshape(1, -1)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("joint_position must contain only finite values")
        return value

    def _spheres_for_joint_batch(self, joint_position: torch.Tensor) -> torch.Tensor:
        if joint_position.ndim != 2 or joint_position.shape[-1] != self._robot_config.dof:
            raise ValueError("joint_position batch must have shape [batch, dof]")
        state = self._robot_model.compute_kinematics(
            JointState.from_position(joint_position, joint_names=self._robot_model.joint_names)
        )
        assert state.robot_spheres is not None
        return state.robot_spheres

    def _pair_clearance(self, spheres: torch.Tensor) -> torch.Tensor:
        """Return signed clearances for configured pairs, in deterministic pair order."""
        pairs = self._self_collision_config.collision_pairs
        if pairs is None or pairs.numel() == 0:
            return spheres.new_empty((*spheres.shape[:2], 0))
        pair = pairs.to(device=spheres.device, dtype=torch.long)
        first, second = pair[:, 0], pair[:, 1]
        positions = spheres[..., :3]
        center_distance = torch.linalg.vector_norm(
            positions[..., first, :] - positions[..., second, :], dim=-1
        )
        padding = self._self_collision_config.sphere_padding.to(spheres)
        effective_radius = (
            spheres[..., first, 3] + spheres[..., second, 3]
            + padding[first] + padding[second]
        )
        return center_distance - effective_radius

    def _link_pair(self, first: int, second: int) -> Tuple[str, str]:
        params = self._robot_config.kinematics_config
        names = {index: name for name, index in params.link_name_to_idx_map.items()}
        links = params.link_sphere_idx_map
        return (names[int(links[first].item())], names[int(links[second].item())])

    def _result_from_spheres(self, spheres: torch.Tensor) -> dict[str, Any]:
        """Produce the pinned detailed result schema for one joint state."""
        clearance = self._pair_clearance(spheres)[0, 0]
        pairs = self._self_collision_config.collision_pairs
        violation = (-clearance).clamp_min(0)
        active = violation > 0
        distances: dict[Tuple[str, str], float] = {}
        colliding_pairs: list[Tuple[str, str]] = []
        if pairs is not None:
            for pair_index in torch.nonzero(active, as_tuple=False).flatten().detach().cpu().tolist():
                first, second = (int(value) for value in pairs[pair_index].tolist())
                link_pair = self._link_pair(first, second)
                magnitude = float(violation[pair_index].detach().cpu())
                if link_pair not in distances:
                    colliding_pairs.append(link_pair)
                    distances[link_pair] = magnitude
                else:
                    distances[link_pair] = max(distances[link_pair], magnitude)
        has_collision = bool(active.any().item())
        return {
            "has_collision": has_collision,
            # The early portable facade exposed this spelling.  Retain it as
            # an alias while the pinned spelling above is canonical.
            "collision": has_collision,
            "num_colliding_pairs": len(colliding_pairs),
            "colliding_pairs": colliding_pairs,
            "max_penetration": 0.0 if not has_collision else float(violation.max().detach().cpu()),
            "distances": distances,
            "num_spheres": int(spheres.shape[-2]),
            "num_checked_pairs": int(clearance.numel()),
            "minimum_radius": 0.0 if spheres.numel() == 0 else float(spheres[..., 3].min().detach().cpu()),
        }

    def check_default_joint_configuration_collision(self) -> dict[str, Any]:
        return self.check_collision_at_config(self._robot_model.default_joint_position)

    def check_collision_at_config(
        self, joint_position: Union[List[float], np.ndarray, torch.Tensor]
    ) -> dict[str, Any]:
        q = self._joint_tensor(joint_position)
        spheres = self._spheres_for_joint_batch(q)
        # Exercise the production self-collision cost as well as preserving
        # pair-resolved debugger evidence below.
        self._collision_cost.setup_batch_tensors(1, 1)
        self._collision_cost(spheres)
        result = self._result_from_spheres(spheres)
        self._last_result = deepcopy(result)
        self._check_count += 1
        return result

    def _sample_joint_positions(self, count: int, *, seed: int) -> torch.Tensor:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("sample count must be a positive integer")
        limits = self._robot_model.get_joint_limits()
        low = limits.position_lower_limits.reshape(-1).detach().cpu()
        high = limits.position_upper_limits.reshape(-1).detach().cpu()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = low + torch.rand((count, self._robot_config.dof), generator=generator) * (high - low)
        return values.to(**self.device_cfg.as_torch_dict())

    def sample_collision_checks(self, num_samples: int = 1000, batch_size: int = 100, seed: int = 42) -> dict[str, Any]:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        samples = self._sample_joint_positions(num_samples, seed=seed)
        pair_counts: dict[Tuple[str, str], int] = {}
        total_collisions = 0
        for start in range(0, num_samples, batch_size):
            spheres = self._spheres_for_joint_batch(samples[start : start + batch_size])
            clearance = self._pair_clearance(spheres).squeeze(1)
            active = clearance < 0
            total_collisions += int(active.any(dim=-1).sum().item()) if active.numel() else 0
            pairs = self._self_collision_config.collision_pairs
            if pairs is None:
                continue
            for batch_index in range(active.shape[0]):
                pair_indices = torch.nonzero(active[batch_index], as_tuple=False).flatten().detach().cpu().tolist()
                # Count each link pair at most once per sampled configuration,
                # matching V2's unique link-pair audit rather than inflating a
                # frequency for multiple overlapping sphere pairs.
                link_pairs = {
                    self._link_pair(int(pairs[index, 0]), int(pairs[index, 1]))
                    for index in pair_indices
                }
                for pair in link_pairs:
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
        frequent = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        return {
            "total_samples": num_samples,
            "collision_count": total_collisions,
            "collision_rate": 100.0 * total_collisions / num_samples,
            "frequent_collisions": frequent,
            "seed": seed,
            "batch_size": batch_size,
        }

    def find_never_colliding_pairs(self, num_samples: int = 10000, batch_size: int = 10000, seed: int = 345) -> List[Tuple[str, str]]:
        stats = self.sample_collision_checks(num_samples, batch_size, seed)
        observed = {pair for pair, _ in stats["frequent_collisions"]}
        pairs = self._self_collision_config.collision_pairs
        if pairs is None:
            return []
        ordered = list(dict.fromkeys(
            self._link_pair(int(first), int(second)) for first, second in pairs.detach().cpu().tolist()
        ))
        return [pair for pair in ordered if pair not in observed]

    def collision_matrix_stats(self) -> dict[str, int | float]:
        count = self._self_collision_config.num_spheres
        checked = self._self_collision_config.num_collision_checks
        possible = count * (count - 1) // 2
        return {
            "total_spheres": count,
            "total_possible_pairs": possible,
            "checked_pairs": checked,
            "ignored_pairs": possible - checked,
            "checking_percent": 0.0 if possible == 0 else 100.0 * checked / possible,
        }

    def print_collision_matrix_stats(self) -> None:
        stats = self.collision_matrix_stats()
        print("Collision Matrix Statistics:")
        print(f"  Total spheres: {stats['total_spheres']}")
        print(f"  Total possible pairs: {stats['total_possible_pairs']:,}")
        print(f"  Checked pairs: {stats['checked_pairs']:,}")
        print(f"  Ignored pairs: {stats['ignored_pairs']:,}")
        print(f"  Checking: {stats['checking_percent']:.1f}% of all possible pairs")

    def inspection_report(self) -> dict[str, Any]:
        """Return JSON-safe static model and last-run inspection data."""
        robot = self._robot_config.kinematics_config.robot_cfg
        last_result = self.last_result
        if last_result is not None:
            last_result["distances"] = {
                "::".join(pair): value for pair, value in last_result["distances"].items()
            }
        return {
            "config_path": self.config_path,
            "device": str(self.device_cfg.device),
            "dtype": str(self.device_cfg.dtype).removeprefix("torch."),
            "joint_names": list(self._robot_model.joint_names),
            "tool_frames": list(self._robot_model.tool_frames),
            "base_link": self._robot_model.base_link,
            "collision_links": list(robot.collision_link_names),
            "matrix": self.collision_matrix_stats(),
            "check_count": self._check_count,
            "last_result": last_result,
        }

    def export_inspection_report(self, file_path: str | Path) -> Path:
        """Write a deterministic portable inspection JSON artifact."""
        destination = Path(file_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.inspection_report(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    def visualize_collision_at_config(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("visualization requires the optional external Viser backend")

    @property
    def robot_config(self) -> KinematicsCfg:
        return self._robot_config

    @property
    def robot_model(self) -> Kinematics:
        return self._robot_model

    @property
    def last_result(self) -> Optional[dict[str, Any]]:
        return None if self._last_result is None else deepcopy(self._last_result)


__all__ = ["RobotDebugger"]
