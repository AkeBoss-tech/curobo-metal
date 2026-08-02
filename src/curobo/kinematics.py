"""Differentiable forward kinematics and robot geometry."""

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState


def _get_robot_as_mesh(self, joint_position):
    """Expose the pinned mesh-query signature with a precise portable boundary."""
    del joint_position
    return self.get_robot_link_meshes()


if not getattr(Kinematics, "_portable_public_mesh_signature", False):
    # The portable backend has no CUDA/Isaac mesh construction path.  Keeping
    # the public argument makes an otherwise ordinary downstream call fail at
    # the documented feature boundary rather than with a Python arity error.
    Kinematics.get_robot_as_mesh = _get_robot_as_mesh
    Kinematics._portable_public_mesh_signature = True

__all__ = ["Kinematics", "KinematicsCfg", "KinematicsState"]
