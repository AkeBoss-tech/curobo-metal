"""Portable support-polygon cost used by whole-body balance rollouts.

This mirrors the small pure-PyTorch implementation in pinned cuRobo V2.  In
particular, the contact polygon is intentionally detached from autograd while
the centre-of-mass query remains differentiable.  That is the useful contract
for trajectory optimization: contact geometry is a fixed support set for a
rollout, and gradients tell the optimizer how to move the CoM back into it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from curobo._src.cost.cost_support_polygon_cfg import CostSupportPolygonCfg

from curobo._src.cost.cost_base import BaseCost
from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper
from curobo._src.util.logging import log_and_raise


class CostSupportPolygon(BaseCost):
    """Penalty for a robot centre of mass outside its feet's convex hull.

    ``robot_com`` is ``[batch, horizon, 3]`` and ``robot_spheres`` is
    ``[batch, horizon, spheres, 4]``.  The cached polygon is formed from the
    first rollout time step, matching cuRobo V2's fixed-contact assumption.
    A cache is automatically rebuilt if its batch/device/dtype no longer
    matches a later rollout; this prevents stale CPU hulls and stale batch
    layouts from leaking into a Metal solve.
    """

    _DEFAULT_CONTACT_PADDING = 0.05
    _INSIDE_MARGIN = 0.1

    def __init__(self, config: CostSupportPolygonCfg):
        super().__init__(config)
        self._polygon_helper = ConvexPolygon2DHelper(config.device_cfg)
        self.vertices: Optional[torch.Tensor] = None

    def build_convex_hull(self, vertices: torch.Tensor, padding: Optional[float] = None):
        """Build and cache a hull for each batch element.

        The helper detaches the vertices before its discrete hull operation.
        This is deliberate: differentiating through changes in contact order
        is not a defined cuRobo operation.
        """
        if not isinstance(vertices, torch.Tensor):
            raise TypeError("vertices must be a torch.Tensor")
        if vertices.ndim != 3 or vertices.shape[-1] != 2:
            raise ValueError("vertices must have shape [batch, vertices, 2]")
        if vertices.shape[1] == 0:
            raise ValueError("vertices must include at least one contact point")
        if not vertices.is_floating_point():
            raise TypeError("vertices must use a floating-point dtype")
        self._polygon_helper.build_convex_hull(vertices, padding)
        self.vertices = self._polygon_helper._cached_convex_hulls
        # The helper only returns None for an invalid call, which we normalize
        # into a direct, portable error instead of allowing a later index error.
        if self.vertices is None:
            raise RuntimeError("convex hull construction did not produce a hull")
        # The pinned API does not declare a return value.  Returning this
        # portable convenience value remains backward compatible while the
        # exact callable shape stays unchanged.
        return self.vertices

    def _validate_input(self, robot_com: torch.Tensor, robot_spheres: torch.Tensor) -> torch.Tensor:
        if not isinstance(robot_com, torch.Tensor) or not isinstance(robot_spheres, torch.Tensor):
            raise TypeError("robot_com and robot_spheres must be torch.Tensor instances")
        if robot_com.ndim != 3 or robot_com.shape[-1] != 3:
            raise ValueError("robot_com must have shape [batch, horizon, 3]")
        if robot_spheres.ndim != 4 or robot_spheres.shape[-1] != 4:
            raise ValueError("robot_spheres must have shape [batch, horizon, spheres, 4]")
        if robot_com.shape[:2] != robot_spheres.shape[:2]:
            raise ValueError("robot_com and robot_spheres batch/horizon dimensions must match")
        if robot_com.device != robot_spheres.device:
            raise ValueError("robot_com and robot_spheres must be on the same device")
        if robot_com.dtype != robot_spheres.dtype:
            raise ValueError("robot_com and robot_spheres must use the same dtype")
        if not robot_com.is_floating_point() or not robot_spheres.is_floating_point():
            raise TypeError("robot_com and robot_spheres must use floating-point dtypes")
        if robot_com.shape[0] == 0:
            # There is no hull to build for an empty batch; preserving the
            # shape makes composition with empty batched solves well-defined.
            return torch.empty(0, device=robot_spheres.device, dtype=torch.long)

        raw_indices = self.config.foot_sphere_indices
        if raw_indices is None:
            if self._polygon_helper._cached_convex_hulls is None:
                raise ValueError(
                    "foot_sphere_indices must be configured unless a convex hull was built explicitly"
                )
            return torch.empty(0, device=robot_spheres.device, dtype=torch.long)
        indices = torch.as_tensor(raw_indices, device=robot_spheres.device, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("foot_sphere_indices must be a non-empty rank-1 index tensor")
        if torch.any(indices < 0) or torch.any(indices >= robot_spheres.shape[2]):
            raise ValueError("foot_sphere_indices contains an out-of-range sphere index")
        return indices

    def _cache_matches(self, robot_com: torch.Tensor) -> bool:
        hulls = self._polygon_helper._cached_convex_hulls
        return (
            hulls is not None
            and hulls.shape[0] == robot_com.shape[0]
            and hulls.device == robot_com.device
            and hulls.dtype == robot_com.dtype
        )

    def _compute_support_polygon_cost_vectorized(self, com_pos: torch.Tensor) -> torch.Tensor:
        """Evaluate signed hull distances and the pinned inside-margin loss."""
        if self._polygon_helper._cached_convex_hulls is None:
            log_and_raise("No convex hull cached, call build_convex_hull first")
        batch_indices = torch.arange(com_pos.shape[0], device=com_pos.device)
        signed_distances = self._polygon_helper.compute_point_hull_distance(
            com_pos.unsqueeze(2), batch_indices
        ).squeeze(-1)
        inside_weight = float(self.config.inside_cost_weight)
        if inside_weight > 0.0:
            inside_cost = inside_weight * (self._INSIDE_MARGIN + signed_distances).clamp_min(0.0)
            cost = torch.where(signed_distances < 0.0, inside_cost, signed_distances)
        else:
            cost = signed_distances.clamp_min(0.0)
        # V2 expects a scalar support-polygon weight.  Let PyTorch retain its
        # normal broadcasting behaviour for exotic legacy configurations, just
        # as the upstream ``self._weight.squeeze()`` expression does.
        return cost * self._weight.squeeze().to(device=cost.device, dtype=cost.dtype)

    def forward(self, robot_com: torch.Tensor, robot_spheres: torch.Tensor) -> torch.Tensor:
        indices = self._validate_input(robot_com, robot_spheres)
        if robot_com.shape[0] == 0:
            return robot_com.new_zeros(robot_com.shape[:2])
        if not self._cache_matches(robot_com):
            if indices.numel() == 0:
                raise ValueError("cached convex hull is incompatible with this robot batch")
            foot_spheres = robot_spheres[:, 0, indices, :2].detach()
            self.build_convex_hull(foot_spheres, padding=self._DEFAULT_CONTACT_PADDING)
        return self._compute_support_polygon_cost_vectorized(robot_com[..., :2])

    # ``BaseCost`` is deliberately framework-neutral in this Metal port, so
    # retain PyTorch module-style invocation as a portable extension.
    __call__ = forward
