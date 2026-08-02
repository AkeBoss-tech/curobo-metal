"""Robot-scene collision checking public API.

The compatibility adapters below only normalize public parameter names.  They
delegate all collision calculation, gradients, buffers, and CPU/MPS dispatch
to :class:`RobotSceneCollision`.
"""

from inspect import Parameter, Signature

from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg


def _get_kinematics(self, joint_position, *private_args):
    # Private production helpers optionally pass an environment-index tensor;
    # retain that extension even though the pinned public signature has only
    # ``joint_position``.
    return self._portable_public_get_kinematics(joint_position, *private_args)


def _get_bound(self, q, q_tau=None):
    return self._portable_public_get_bound(q, q_tau)


def _sample(self, n, mask_valid=True, env_query_idx=None):
    return self._portable_public_sample(n, mask_valid, env_query_idx)


def _validate(self, q, env_query_idx=None):
    return self._portable_public_validate(q, env_query_idx)


def _sample_trajectory(self, batch, horizon, mask_valid=True, env_query_idx=None):
    return self._portable_public_sample_trajectory(batch, horizon, mask_valid, env_query_idx)


def _validate_trajectory(self, q, env_query_idx=None):
    return self._portable_public_validate_trajectory(q, env_query_idx)


def _get_active_js(self, full_js):
    return self._portable_public_get_active_js(full_js)


# Keep private callers' optional environment argument operational while making
# public-facade introspection match the pinned Python signature.
_get_kinematics.__signature__ = Signature(
    [
        Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("joint_position", Parameter.POSITIONAL_OR_KEYWORD),
    ]
)


if not getattr(RobotSceneCollision, "_portable_public_signature_adapter", False):
    # Save the backend methods exactly once.  This is intentionally installed
    # at the public import boundary so private consumers retain their existing
    # portable extensions (for example ``idxs_env`` on ``get_kinematics``).
    RobotSceneCollision._portable_public_get_kinematics = RobotSceneCollision.get_kinematics
    RobotSceneCollision._portable_public_get_bound = RobotSceneCollision.get_bound
    RobotSceneCollision._portable_public_sample = RobotSceneCollision.sample
    RobotSceneCollision._portable_public_validate = RobotSceneCollision.validate
    RobotSceneCollision._portable_public_sample_trajectory = RobotSceneCollision.sample_trajectory
    RobotSceneCollision._portable_public_validate_trajectory = RobotSceneCollision.validate_trajectory
    RobotSceneCollision._portable_public_get_active_js = RobotSceneCollision.get_active_js
    RobotSceneCollision.get_kinematics = _get_kinematics
    RobotSceneCollision.get_bound = _get_bound
    RobotSceneCollision.sample = _sample
    RobotSceneCollision.validate = _validate
    RobotSceneCollision.sample_trajectory = _sample_trajectory
    RobotSceneCollision.validate_trajectory = _validate_trajectory
    RobotSceneCollision.get_active_js = _get_active_js
    RobotSceneCollision._portable_public_signature_adapter = True

RobotCollisionChecker = RobotSceneCollision
RobotCollisionCheckerCfg = RobotSceneCollisionCfg

__all__ = ["RobotCollisionChecker", "RobotCollisionCheckerCfg"]
