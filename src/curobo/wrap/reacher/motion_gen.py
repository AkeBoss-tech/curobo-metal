"""Established MotionGen API translated to the portable motion planner."""

from dataclasses import dataclass
import inspect

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.motion.motion_planner_result import MotionPlannerResult
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo.rollout.cost.pose_cost import PoseCostMetric


@dataclass
class MotionGenPlanConfig:
    max_attempts: int = 5
    enable_graph: bool = True
    enable_graph_attempt: int = 1
    timeout: float = 10.0
    enable_opt: bool = True
    parallel_finetune: bool = True


class MotionGenConfig:
    @staticmethod
    def load_from_robot_config(robot_config, world_config=None, tensor_args=None, **kwargs):
        if tensor_args is not None:
            kwargs["device_cfg"] = tensor_args
        accepted = inspect.signature(MotionPlannerCfg.create).parameters
        translated = {key: value for key, value in kwargs.items() if key in accepted}
        return MotionPlannerCfg.create(
            robot_config, scene_model=world_config, **translated
        )


class MotionGen(MotionPlanner):
    def _legacy_goal(self, goal_pose):
        if isinstance(goal_pose, GoalToolPose):
            return goal_pose
        if not isinstance(goal_pose, Pose):
            raise TypeError("goal_pose must be Pose or GoalToolPose")
        frame = self.ik_solver.kinematics.tool_frames[0]
        return GoalToolPose.from_poses({frame: goal_pose})

    def plan_single(self, start_state, goal_pose, plan_config=None):
        config = MotionGenPlanConfig() if plan_config is None else plan_config
        return self.plan_pose(
            self._legacy_goal(goal_pose), start_state,
            max_attempts=config.max_attempts,
            enable_graph_attempt=(config.enable_graph_attempt if config.enable_graph else
                                  config.max_attempts + 1),
        )

    def get_retract_config(self):
        cspace = self.ik_solver.kinematics.config.kinematics_config.cspace
        return cspace.default_joint_position


MotionGenResult = MotionPlannerResult
__all__ = [
    "MotionGen", "MotionGenConfig", "MotionGenPlanConfig", "MotionGenResult",
    "PoseCostMetric",
]
