"""Solver problem-shape metadata.

This module is deliberately tensor-free.  ``SolveState`` travels between the
public solve APIs, goal-buffer managers, and optimizer backends, so its values
must describe a problem identically on CPU and MPS.  CUDA graph handles and
other CUDA-only execution state do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .solve_mode import SolveMode


@dataclass
class SolveState:
    """Metadata describing one optimization problem layout.

    ``batch_size`` counts independent requested problems while the seed fields
    count candidates per problem.  ``num_seeds`` is the generic candidate
    count; when it is omitted, the V2 precedence is IK, then TrajOpt, then
    graph seeds.  ``get_batch_size`` intentionally follows the upstream
    contract and therefore requires a resolved generic seed count.

    Args:
        solve_type: Requested :class:`~.solve_mode.SolveMode`.
        batch_size: Number of requested problems.
        num_envs: Number of collision environments.
        num_goalset: Goal poses per requested problem.
        num_seeds: Generic candidates per problem.
        num_ik_seeds: Candidates used by inverse kinematics.
        num_graph_seeds: Candidates used by graph search.
        num_trajopt_seeds: Candidates used by trajectory optimization.
        tool_frames: Target link names, in the caller-supplied order.
    """

    solve_type: SolveMode
    batch_size: int
    num_envs: int
    num_goalset: int = 1
    multi_env: bool = False
    batch_mode: bool = False
    num_seeds: Optional[int] = None
    num_ik_seeds: Optional[int] = None
    num_graph_seeds: Optional[int] = None
    num_trajopt_seeds: Optional[int] = None
    tool_frames: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Derive V2 mode flags and a generic seed count when possible."""
        self.multi_env = self.num_envs != 1
        # V2 preserves an explicitly requested ``batch_mode=True`` for a
        # single-item caller, but all multi-item layouts are necessarily batch
        # mode.  Do not reset the explicit flag in the single-item case.
        if self.batch_size > 1:
            self.batch_mode = True
        if self.num_seeds is None:
            self.num_seeds = self.num_ik_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_trajopt_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_graph_seeds

    def clone(self) -> "SolveState":
        """Rebuild a V2-compatible copy of this problem descriptor.

        The upstream type treats ``tool_frames`` as immutable caller metadata;
        clone therefore preserves that list reference rather than silently
        changing alias semantics.
        """
        return SolveState(
            solve_type=self.solve_type,
            num_envs=self.num_envs,
            batch_size=self.batch_size,
            num_goalset=self.num_goalset,
            multi_env=self.multi_env,
            batch_mode=self.batch_mode,
            num_seeds=self.num_seeds,
            num_ik_seeds=self.num_ik_seeds,
            num_graph_seeds=self.num_graph_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            tool_frames=self.tool_frames,
        )

    def get_batch_size(self) -> int:
        """Return generic optimizer candidates: ``batch_size * num_seeds``."""
        return self.num_seeds * self.batch_size

    def get_ik_batch_size(self) -> int:
        """Return IK candidates, or zero when IK is absent from this solve."""
        if self.num_ik_seeds is None:
            return 0
        return self.num_ik_seeds * self.batch_size

    def get_trajopt_batch_size(self) -> int:
        """Return TrajOpt candidates, or zero when TrajOpt is absent."""
        if self.num_trajopt_seeds is None:
            return 0
        return self.num_trajopt_seeds * self.batch_size


@dataclass
class MotionPlanSolveState:
    """Composite metadata for an IK-plus-TrajOpt motion-planning request."""

    solve_type: SolveMode
    ik_solve_state: SolveState
    trajopt_solve_state: SolveState


__all__ = ["SolveState", "MotionPlanSolveState"]
