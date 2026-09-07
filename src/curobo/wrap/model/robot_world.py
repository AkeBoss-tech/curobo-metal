"""Established RobotWorld names backed by portable scene collision."""

from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg

RobotWorld = RobotSceneCollision
RobotWorldConfig = RobotSceneCollisionCfg
__all__ = ["RobotWorld", "RobotWorldConfig"]
