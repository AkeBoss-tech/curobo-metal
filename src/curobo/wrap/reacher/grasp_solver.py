"""Established grasp-planning names backed by MotionPlanner."""

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg

GraspSolver = MotionPlanner
GraspSolverConfig = MotionPlannerCfg
__all__ = ["GraspSolver", "GraspSolverConfig"]
