"""Portable cuRobo goal registry."""
from dataclasses import dataclass
from typing import Optional
import torch
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose

@dataclass
class GoalRegistry:
    name: str = "goal"
    batch_size: int = -1
    num_goalset: int = 1
    num_seeds: int = 1
    goal_js: Optional[JointState] = None
    seed_goal_js: Optional[JointState] = None
    link_goal_poses: Optional[GoalToolPose] = None
    current_js: Optional[JointState] = None
    current_state_dt: Optional[torch.Tensor] = None
    idxs_link_pose: Optional[torch.Tensor] = None
    idxs_goal_js: Optional[torch.Tensor] = None
    idxs_current_js: Optional[torch.Tensor] = None
    idxs_seed_goal_js: Optional[torch.Tensor] = None
    idxs_enable: Optional[torch.Tensor] = None
    idxs_env: Optional[torch.Tensor] = None
    seed_enable_implicit_goal_js: Optional[torch.Tensor] = None
    update_idxs_buffers: bool = True
    def __post_init__(self):
        candidates = [x for x in (self.goal_js, self.current_js) if x is not None]
        if self.batch_size < 0 and candidates: self.batch_size = candidates[0].position.shape[0]
    @property
    def link_goal_pose_dict(self):
        return None if self.link_goal_poses is None else self.link_goal_poses.to_dict()
    def repeat_seeds(self, num_seeds, repeat_seed_idx_buffers=False):
        out = self.clone(); out.num_seeds = num_seeds
        return out
    def clone(self):
        import copy
        return copy.deepcopy(self)
    def apply_kernel(self, kernel_mat):
        out = self.clone()
        if out.goal_js is not None: out.goal_js.position = kernel_mat @ out.goal_js.position
        return out
    def copy_(self, goal, update_idx_buffers=True, allow_clone=True):
        self.__dict__.update(goal.clone().__dict__); return self
    def get_batch_goal_state(self): return self.goal_js
    @classmethod
    def create_idx(cls, pose_batch_size, multi_env, num_seeds, device_cfg,
                   seed_goal_state=None, repeat_seed_idx_buffers=False):
        result = cls(batch_size=pose_batch_size, num_seeds=num_seeds, seed_goal_js=seed_goal_state)
        result.create_index_buffers(pose_batch_size, multi_env, num_seeds, device_cfg)
        return result
    def create_index_buffers(self, batch_size, multi_env, num_seeds, device_cfg):
        self.batch_size, self.num_seeds = batch_size, num_seeds
        self.idxs_goal_js = torch.arange(batch_size, device=device_cfg.device).repeat_interleave(num_seeds)
        self.idxs_env = self.idxs_goal_js.clone() if multi_env else torch.zeros_like(self.idxs_goal_js)
        return self
    def get_index_size(self): return self.batch_size * self.num_seeds

__all__ = ["GoalRegistry", "GoalToolPose", "JointState", "Pose"]
