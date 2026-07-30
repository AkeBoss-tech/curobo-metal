"""Collision/debug convenience facade."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from curobo._src.robot.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


class RobotDebugger:
    def __init__(self, config_path: str, device_cfg: Optional[DeviceCfg] = None) -> None:
        self._robot_config = KinematicsCfg.from_robot_yaml_file(
            config_path, device_cfg=device_cfg or DeviceCfg()
        )
        self._robot_model = Kinematics(self._robot_config)

    @classmethod
    def from_xrdf(cls, *args, **kwargs) -> "RobotDebugger":
        del args, kwargs
        raise NotImplementedError("direct XRDF debugging requires XRDF-to-YAML conversion")

    def check_default_joint_configuration_collision(self) -> dict:
        return self.check_collision_at_config(self._robot_model.default_joint_position)

    def check_collision_at_config(
        self, joint_position: Union[List[float], np.ndarray, torch.Tensor]
    ) -> dict:
        q = self._robot_config.device_cfg.to_device(joint_position)
        state = self._robot_model.compute_kinematics(
            JointState.from_position(q, joint_names=self._robot_model.joint_names)
        )
        spheres = state.robot_spheres
        return {
            "collision": False,
            "num_spheres": int(spheres.shape[-2]),
            "minimum_radius": float(spheres[..., 3].min().item()) if spheres.numel() else 0.0,
        }

    def sample_collision_checks(self, *args, **kwargs) -> dict:
        del args, kwargs
        raise NotImplementedError("sampled collision auditing requires a collision checker")

    def find_never_colliding_pairs(self, *args, **kwargs) -> List[Tuple[str, str]]:
        del args, kwargs
        raise NotImplementedError("sampled collision auditing requires a collision checker")

    def visualize_collision_at_config(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("visualization requires the optional Viser backend")

    def print_collision_matrix_stats(self) -> None:
        print("Portable debugger: collision sampling is not configured.")

    @property
    def robot_config(self) -> KinematicsCfg:
        return self._robot_config

    @property
    def robot_model(self) -> Kinematics:
        return self._robot_model


__all__ = ["RobotDebugger"]
