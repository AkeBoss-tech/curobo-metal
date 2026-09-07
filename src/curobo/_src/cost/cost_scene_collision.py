"""Portable pinned cuRobo V2 world-collision cost.

The upstream implementation routes this class directly to Warp kernels.  This
facade keeps its public lifecycle on CPU and Metal, but delegates geometry
queries to the production :class:`~curobo._src.geom.collision.collision_scene.SceneCollision`
checker.  Swept checking is the checker's documented fixed-resolution query;
it is not an analytic continuous-collision certificate or a Warp kernel ABI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg

from curobo._src.cost.cost_base import BaseCost
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.torch_util import get_torch_jit_decorator

from .portable import SceneCollisionCost as _PortableSceneCollisionCost


class _SceneCollisionCostPortable(_PortableSceneCollisionCost):
    """Aggregate portable scene-clearance queries into a trajectory cost.

    A checker returning ``[batch, horizon, spheres]`` is understood to return
    signed clearances.  This class applies the V2 activation-distance quadratic
    penalty and then the configured sum/max reduction.  A checker which
    returns ``[batch, horizon]`` is treated as an already-aggregated custom
    loss, retaining the useful duck-typed extension point of the original
    facade.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        if config.scene_collision_checker is None:
            log_and_raise("scene_collision_checker must be set before using world collision cost")
        # Config factories consult this attribute when assembling a rollout.
        # Point an instantiated record at the concrete facade even though the
        # portable dataclass cannot import this module without a cycle.
        config.class_type = type(self)
        self._last_swept = False

    @staticmethod
    def _spheres(state: Any) -> torch.Tensor:
        spheres = getattr(state, "robot_spheres", None)
        if spheres is None:
            spheres = getattr(state, "link_spheres_tensor", state)
        return spheres

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        super().setup_batch_tensors(batch_size, horizon)
        # Allocate even for zero configured spheres.  That matches the V2
        # lifecycle and lets callers inspect a valid empty gradient workspace.
        self._collision_buffer = CollisionBuffer.from_shape(
            torch.Size((batch_size, horizon, self.config.num_spheres, 4)), self.device_cfg
        )
        return True

    def update_num_spheres(
        self,
        num_spheres: int,
        batch_size: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> None:
        if isinstance(num_spheres, bool) or not isinstance(num_spheres, int) or num_spheres < 0:
            raise ValueError("num_spheres must be a non-negative integer")
        self.config.update_num_spheres(num_spheres)
        resolved_batch = self._batch_size if batch_size is None else int(batch_size)
        resolved_horizon = self._horizon if horizon is None else int(horizon)
        # It is valid to set topology before a rollout allocates its batch.
        if resolved_batch >= 0 and resolved_horizon >= 0:
            self.setup_batch_tensors(resolved_batch, resolved_horizon)

    def validate_input(
        self,
        state: Any,
        idxs_env_query: Optional[torch.Tensor] = None,
        trajectory_dt: Optional[torch.Tensor] = None,
    ) -> bool:
        spheres = self._spheres(state)
        if not isinstance(spheres, torch.Tensor) or spheres.ndim != 4 or spheres.shape[-1] != 4:
            raise ValueError("scene collision expects [batch,horizon,spheres,4] xyzw-radius tensors")
        if not self.device_cfg.is_same_torch_device(spheres.device):
            raise ValueError("scene collision spheres must be on the configured device")
        if spheres.dtype != self.device_cfg.dtype:
            raise TypeError("scene collision spheres dtype must match device_cfg.dtype")
        if spheres.device.type == "mps" and spheres.dtype != torch.float32:
            raise TypeError("MPS scene collision supports float32 only")
        if not bool(torch.isfinite(spheres).all().item()):
            raise ValueError("scene collision spheres must contain only finite values")
        # Exactly -100 is the pinned disabled-attachment sentinel. Other
        # negative radii remain malformed geometry and fail closed.
        invalid_radius = (spheres[..., 3] < 0) & (spheres[..., 3] != -100.0)
        if bool(invalid_radius.any().item()):
            raise ValueError("scene collision sphere radii must be non-negative")
        if self.config.num_spheres and spheres.shape[-2] != self.config.num_spheres:
            raise ValueError("sphere count does not match configured num_spheres")
        if self._batch_size >= 0 and spheres.shape[0] != self._batch_size:
            raise ValueError("robot spheres batch size does not match setup_batch_tensors")
        if self._horizon >= 0 and spheres.shape[1] != self._horizon:
            raise ValueError("robot spheres horizon does not match setup_batch_tensors")
        if idxs_env_query is not None:
            if not isinstance(idxs_env_query, torch.Tensor) or idxs_env_query.shape != (spheres.shape[0],):
                raise ValueError("idxs_env_query must have shape [batch]")
            if idxs_env_query.device != spheres.device or idxs_env_query.dtype not in (torch.int32, torch.int64):
                raise ValueError("idxs_env_query must be an int32/int64 tensor on the query device")
        if self.config.use_sweep:
            if trajectory_dt is None:
                raise ValueError("trajectory_dt is required when use_sweep=True")
            if not isinstance(trajectory_dt, torch.Tensor) or trajectory_dt.shape != (spheres.shape[0],):
                raise ValueError("trajectory_dt must have shape [batch] when use_sweep=True")
            if trajectory_dt.device != spheres.device:
                raise ValueError("trajectory_dt must be on the query device")
            if not torch.is_floating_point(trajectory_dt) or not bool(torch.isfinite(trajectory_dt).all().item()):
                raise ValueError("trajectory_dt must be finite and floating point")
            if bool((trajectory_dt <= 0).any().item()):
                raise ValueError("trajectory_dt must be positive")
        return True

    def reset(self, reset_problem_ids=None, **kwargs) -> None:
        del kwargs
        if self._collision_buffer is None:
            return None
        if reset_problem_ids is None:
            self._collision_buffer.zero_()
            return None
        problem_ids = torch.as_tensor(reset_problem_ids, device=self._collision_buffer.distance.device)
        if problem_ids.ndim != 1 or problem_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("reset_problem_ids must be a rank-1 integer tensor")
        if bool(((problem_ids < 0) | (problem_ids >= self._collision_buffer.distance.shape[0])).any().item()):
            raise ValueError("reset_problem_ids contains an out-of-range batch index")
        self._collision_buffer.distance[problem_ids] = 0
        self._collision_buffer.gradient[problem_ids] = 0
        return None

    def get_gradient_buffer(self):
        if self._collision_buffer is not None:
            return self._collision_buffer.gradient
        checker = self.config.scene_collision_checker
        return getattr(checker, "collision_buffer", None) if checker is not None else None

    def _query_state(self, state: Any, spheres: torch.Tensor) -> Any:
        return state if hasattr(state, "robot_spheres") else SimpleNamespace(robot_spheres=spheres)

    def _unit_weight(self, spheres: torch.Tensor) -> torch.Tensor:
        # SceneCollision's low-level interface multiplies its signed clearance
        # by this argument.  Keep raw clearance in the buffer and apply this
        # cost's weight exactly once after the sum/max reduction.
        return torch.ones_like(self._weight, device=spheres.device, dtype=spheres.dtype)

    def _call_checker(
        self,
        state: Any,
        spheres: torch.Tensor,
        idxs_env_query: Optional[torch.Tensor],
        trajectory_dt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        checker = self.config.scene_collision_checker
        if checker is None:
            raise ValueError("scene_collision_checker is required")
        swept = bool(self.config.use_sweep)
        suffix = "collision" if self.config.convert_to_binary else "distance"
        method_name = f"get_{'swept_' if swept else ''}sphere_{suffix}"
        method = getattr(checker, method_name, None)
        if method is None and self.config.convert_to_binary:
            method_name = f"get_{'swept_' if swept else ''}sphere_distance"
            method = getattr(checker, method_name, None)
        if method is None:
            raise NotImplementedError(
                f"scene checker does not provide {method_name}; raw Warp collision kernels are unavailable"
            )
        query_state = self._query_state(state, spheres)
        if self._collision_buffer is None:
            self.setup_batch_tensors(spheres.shape[0], spheres.shape[1])
        base = dict(
            activation_distance=self.config.activation_distance,
            env_query_idx=idxs_env_query,
            return_loss=self.use_grad_input,
        )
        if swept:
            base.update(
                trajectory_dt=trajectory_dt,
                enable_speed_metric=bool(self.config.use_speed_metric),
            )
        # Native V2-like checkers take state, a reusable workspace, a query
        # weight and the named options.  Small user checkers commonly accept
        # only raw spheres; retain that extension point as a second attempt.
        try:
            value = method(query_state, self._collision_buffer, self._unit_weight(spheres), **base)
        except TypeError as native_error:
            try:
                if swept:
                    value = method(spheres, trajectory_dt=trajectory_dt, env_query_idx=idxs_env_query)
                else:
                    value = method(spheres, env_query_idx=idxs_env_query)
            except TypeError:
                raise native_error
        return torch.as_tensor(value, device=spheres.device, dtype=spheres.dtype)

    def _apply_scene_weight(self, value: torch.Tensor) -> torch.Tensor:
        weight = self._weight if self._weight.numel() == 1 else self._weight.mean()
        result = value * weight.to(value)
        return result if self.enabled else result * 0

    def _aggregate_clearance(self, value: torch.Tensor, spheres: torch.Tensor) -> torch.Tensor:
        if value.shape == spheres.shape[:-1]:
            activation = self.config.activation_distance.reshape(-1)[0].to(value)
            penalty = 0.5 * (activation - value).clamp_min(0).square()
            self._record_gradient_buffer(value, penalty, spheres)
            if self.config.convert_to_binary:
                return torch.where(penalty > 0, penalty + 1.0, penalty)
            # The pinned SceneCollisionCost returns one value per robot sphere.
            # ``sum_distance`` only selects the native query's gradient mode;
            # its jit_weight_* helpers are separate utilities and are not
            # applied by forward().
            return penalty
        if value.shape != spheres.shape[:2]:
            raise ValueError(
                "scene checker must return signed [batch,horizon,spheres] clearances "
                "or an aggregated [batch,horizon] loss"
            )
        return value

    def _record_gradient_buffer(
        self,
        clearance: torch.Tensor,
        penalty: torch.Tensor,
        spheres: torch.Tensor,
    ) -> None:
        """Store a detached gradient of this *cost* in the reusable workspace.

        ``SceneCollision`` populates ``CollisionBuffer.gradient`` with the
        gradient of signed clearance.  That is insufficient for V2 callers
        that use ``get_gradient_buffer`` for a collision-cost linearization:
        the activation hinge, reduction, binary offset, and configured cost
        weight must be accounted for.  When ``use_grad_input`` is requested,
        take an exact first-order VJP through the active portable query.  The
        extra VJP is intentionally detached and retains the outer graph, so a
        later user ``backward()`` remains valid.  When gradients were not
        requested, retain the checker's inexpensive raw diagnostic buffer.

        Raw Warp/CUDA gradient-buffer ABI is not exposed; this is the
        documented CPU/MPS tensor equivalent.
        """
        buffer = self._collision_buffer
        if (
            buffer is None
            or buffer.gradient.shape != spheres.shape
            or not self.use_grad_input
            or not spheres.requires_grad
            or not penalty.requires_grad
        ):
            return
        weight = self._weight.reshape(-1)[0].to(penalty)
        scalar = (penalty * weight).sum()
        gradient = torch.autograd.grad(
            scalar,
            spheres,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if gradient is None:
            buffer.gradient.zero_()
        else:
            buffer.gradient.copy_(gradient.detach())

    def _discrete_fn(self, state: Any, env_query_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        spheres = self._spheres(state)
        value = self._call_checker(state, spheres, env_query_idx, None)
        return self._apply_scene_weight(self._aggregate_clearance(value, spheres))

    def _sweep_fn(
        self,
        state: Any,
        env_query_idx: Optional[torch.Tensor] = None,
        trajectory_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spheres = self._spheres(state)
        value = self._call_checker(state, spheres, env_query_idx, trajectory_dt)
        return self._apply_scene_weight(self._aggregate_clearance(value, spheres))

    def forward(
        self,
        state: KinematicsState | torch.Tensor | Any,
        idxs_env_query: Optional[torch.Tensor] = None,
        trajectory_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.validate_input(state, idxs_env_query, trajectory_dt)
        spheres = self._spheres(state)
        self._last_swept = bool(self.config.use_sweep)
        # Preserve an autograd-connected zero while avoiding needless world
        # queries for disabled cost terms.
        if not self.enabled:
            return spheres[..., 0] * 0
        if not self.config.sum_distance:
            # Match upstream's full-gradient query mode for this option.
            log_info("sum_distance=False will be slower than sum_distance=True")
            self.use_grad_input = True
        if self.config.use_sweep:
            return self._sweep_fn(state, idxs_env_query, trajectory_dt)
        return self._discrete_fn(state, idxs_env_query)

    __call__ = forward


class SceneCollisionCost(BaseCost):
    """Pinned cuRoboV2 declaration surface for portable scene collision.

    The live implementation is bound below.  It preserves the same useful
    Python lifecycle while performing CPU/MPS tensor queries rather than
    exposing the unavailable Warp kernel ABI.
    """

    def __init__(self, config: SceneCollisionCostCfg):
        raise NotImplementedError

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        raise NotImplementedError

    def update_num_spheres(
        self, num_spheres: int, batch_size: Optional[int] = None, horizon: Optional[int] = None
    ):
        raise NotImplementedError

    def forward(
        self,
        state: KinematicsState,
        idxs_env_query: Optional[torch.Tensor] = None,
        trajectory_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def validate_input(
        self,
        robot_spheres_in: torch.Tensor,
        idxs_env_query: Optional[torch.Tensor] = None,
        trajectory_dt: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError

    def get_gradient_buffer(self) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @get_torch_jit_decorator()
    def jit_weight_distance(dist: torch.Tensor, sum_cost: bool) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @get_torch_jit_decorator()
    def jit_weight_collision(dist: torch.Tensor, sum_cost: bool) -> torch.Tensor:
        raise NotImplementedError


# Keep the complete V2 declaration visible to static consumers while the
# runtime remains the portable implementation used throughout this package.
if not TYPE_CHECKING:
    SceneCollisionCost = _SceneCollisionCostPortable


__all__ = ["BaseCost", "CollisionBuffer", "KinematicsState", "SceneCollisionCost"]
