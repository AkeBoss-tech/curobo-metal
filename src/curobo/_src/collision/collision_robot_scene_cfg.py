"""Pinned robot-scene collision configuration record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.cost.portable import (
    CSpaceCostCfg,
    CSpaceCostType,
    PositionCSpaceCost,
    SceneCollisionCost,
    SceneCollisionCostCfg,
    SelfCollisionCost,
    SelfCollisionCostCfg,
)
from curobo._src.geom.collision.collision_scene import create_scene_collision
from curobo._src.robot.types import SelfCollisionKinematicsCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


@dataclass
class _CollisionSettings:
    weight: torch.Tensor
    activation_distance: torch.Tensor


@dataclass
class _SelfCollisionSettings:
    pairs: torch.Tensor
    weight: torch.Tensor
    activation_distance: torch.Tensor


@dataclass
class RobotSceneCollisionCfg:
    kinematics: Any
    sampler: Any
    bound_scale: torch.Tensor
    cspace_cost: Any
    self_collision_cost: Optional[Any] = None
    collision_cost: Optional[Any] = None
    collision_constraint: Optional[Any] = None
    scene_model: Optional[SceneCollision] = None
    rejection_ratio: int = 10
    device_cfg: DeviceCfg = DeviceCfg()
    contact_distance: float = 0.0

    @staticmethod
    def load_from_config(
        robot_config: Union[Any, str] = "franka.yml",
        scene_model: Union[None, str, Dict, SceneCfg, List[SceneCfg], List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        num_envs: int = 1,
        n_meshes: int = 50,
        n_cuboids: int = 50,
        collision_activation_distance: float = 0.2,
        self_collision_activation_distance: float = 0.0,
        max_collision_distance: float = 1.0,
        scene_collision_checker: Optional[SceneCollision] = None,
        pose_weight: List[float] = [1, 1, 1, 1],
    ) -> "RobotSceneCollisionCfg":
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if n_meshes < 0 or n_cuboids < 0:
            raise ValueError("collision cache capacities must be nonnegative")
        if collision_activation_distance < 0 or self_collision_activation_distance < 0:
            raise ValueError("collision activation distances must be nonnegative")
        if max_collision_distance <= 0:
            raise ValueError("max_collision_distance must be positive")

        if isinstance(robot_config, KinematicsCfg):
            kin_cfg = robot_config
        elif hasattr(robot_config, "kinematics") and hasattr(
            robot_config.kinematics, "joint_names"
        ):
            # ``RobotCfg`` is a public input in V2.  Its kinematics record is
            # already the portable tree source, so retain it rather than
            # round-tripping through a YAML file.
            from curobo._src.robot.types import KinematicsParams

            raw = robot_config.kinematics
            kin_cfg = KinematicsCfg(
                device_cfg,
                list(raw.tool_frames),
                KinematicsParams(raw),
                self_collision_config=getattr(raw, "self_collision_config", None),
            )
        else:
            kin_cfg = KinematicsCfg.from_robot_yaml_file(robot_config, device_cfg=device_cfg)
        kinematics = Kinematics(kin_cfg, compute_spheres=True)
        # Collision scene environments route not just world geometry but also
        # the kinematics sphere buffer.  Start each environment from the same
        # reference spheres; attachments may subsequently mutate one row.
        parameters = kin_cfg.kinematics_config
        if num_envs > parameters.link_spheres.shape[0]:
            parameters._link_spheres = parameters.link_spheres[:1].expand(
                num_envs, -1, -1
            ).clone()
            parameters.reference_link_spheres = parameters.reference_link_spheres[:1].expand(
                num_envs, -1, -1
            ).clone()

        def _scene(value: Any) -> SceneCfg:
            if isinstance(value, SceneCfg):
                return value
            if isinstance(value, dict):
                return SceneCfg.create(value)
            if isinstance(value, str):
                from curobo.util_file import get_scene_configs_path, join_path, load_yaml

                return SceneCfg.create(load_yaml(join_path(get_scene_configs_path(), value)))
            raise TypeError(f"unsupported scene model: {type(value).__name__}")

        scene_values = None
        if scene_model is not None:
            scene_values = [_scene(x) for x in scene_model] if isinstance(scene_model, list) else _scene(scene_model)
        checker = scene_collision_checker
        if checker is None and scene_values is not None:
            checker = SceneCollision(SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=scene_values,
                num_envs=num_envs,
                max_distance=max_collision_distance,
                cache={"cuboid": n_cuboids, "mesh": n_meshes, "voxel": 8},
            ))

        robot = kin_cfg.kinematics_config.robot_cfg
        spheres = list(robot.collision_spheres)
        ignored = {
            frozenset((a, b))
            for a, values in robot.self_collision_ignore.items()
            for b in values
        }
        pairs = [
            (i, j)
            for i in range(len(spheres))
            for j in range(i + 1, len(spheres))
            if spheres[i].link_name != spheres[j].link_name
            and frozenset((spheres[i].link_name, spheres[j].link_name)) not in ignored
        ]
        pair_tensor = torch.tensor(
            pairs, device=device_cfg.device, dtype=torch.long
        ).reshape(-1, 2)
        scalar = lambda value: torch.tensor(value, device=device_cfg.device, dtype=device_cfg.dtype)
        limits = kinematics.get_joint_limits()
        cspace_cfg = CSpaceCostCfg(
            weight=[1.0, 0.0],
            device_cfg=device_cfg,
            use_grad_input=True,
            cost_type=CSpaceCostType.POSITION,
            activation_distance=[0.0, 0.0],
            dof=kinematics.dof,
        )
        cspace_cfg.set_bounds(limits, teleport_mode=True)
        cspace_cost = PositionCSpaceCost(cspace_cfg)
        self_cfg = SelfCollisionKinematicsCfg(
            num_spheres=kinematics.total_spheres,
            collision_pairs=pair_tensor,
        )
        self_cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=scalar(1.0),
                device_cfg=device_cfg,
                use_grad_input=True,
                self_collision_kin_config=self_cfg,
            )
        )
        collision_cost = collision_constraint = None
        if checker is not None:
            collision_cost = SceneCollisionCost(
                SceneCollisionCostCfg(
                    weight=scalar(1.0),
                    device_cfg=device_cfg,
                    use_grad_input=False,
                    activation_distance=collision_activation_distance,
                    num_spheres=kinematics.total_spheres,
                    sum_distance=False,
                    _scene_collision_checker=checker,
                )
            )
            collision_constraint = SceneCollisionCost(
                SceneCollisionCostCfg(
                    weight=scalar(1.0),
                    device_cfg=device_cfg,
                    use_grad_input=True,
                    activation_distance=0.0,
                    num_spheres=kinematics.total_spheres,
                    sum_distance=False,
                    _scene_collision_checker=checker,
                )
            )
        sampler = SampleBuffer.create_halton_sample_buffer(
            ndims=kinematics.dof,
            up_bounds=limits.position_upper_limits,
            low_bounds=limits.position_lower_limits,
            store_buffer=2000,
            seed=123,
            device_cfg=device_cfg,
        )
        return RobotSceneCollisionCfg(
            kinematics=kinematics,
            sampler=sampler,
            bound_scale=torch.ones(kinematics.dof, device=device_cfg.device, dtype=device_cfg.dtype),
            cspace_cost=cspace_cost,
            self_collision_cost=self_cost,
            collision_cost=collision_cost,
            collision_constraint=collision_constraint,
            scene_model=checker,
            device_cfg=device_cfg,
            contact_distance=float(collision_activation_distance),
        )
