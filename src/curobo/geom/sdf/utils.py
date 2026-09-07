"""Established collision-checker factory import path."""

from curobo._src.geom.collision.collision_scene import create_scene_collision

create_collision_checker = create_scene_collision
__all__ = ["create_collision_checker"]
