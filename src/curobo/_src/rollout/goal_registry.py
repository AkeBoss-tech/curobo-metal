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
        self._update_batch_size()
        if self.link_goal_poses is not None:
            self.batch_size = self.link_goal_poses.batch_size
            self.num_goalset = self.link_goal_poses.num_goalset
            if self.idxs_link_pose is None:
                self.idxs_link_pose = self._indices(self.batch_size, self.link_goal_poses.device)
        if self.current_js is not None and self.idxs_current_js is None:
            self.idxs_current_js = self._indices(self.current_js.position.shape[0], self.current_js.position.device)
        if self.seed_goal_js is not None:
            if self.seed_goal_js.position.ndim != 3:
                raise ValueError("seed_goal_js must have shape [batch, seeds, dof]")
            batch, seeds = self.seed_goal_js.position.shape[:2]
            if self.idxs_seed_goal_js is None:
                self.idxs_seed_goal_js = self._indices(batch * seeds, self.seed_goal_js.position.device)
            if self.seed_enable_implicit_goal_js is None:
                self.seed_enable_implicit_goal_js = torch.zeros(
                    (batch, seeds), device=self.seed_goal_js.position.device, dtype=torch.uint8
                )

    @staticmethod
    def _indices(size, device):
        return torch.arange(size, device=device, dtype=torch.int32).unsqueeze(-1)

    def _update_batch_size(self):
        if self.link_goal_poses is not None:
            self.batch_size = self.link_goal_poses.batch_size
        elif self.goal_js is not None:
            self.batch_size = self.goal_js.position.shape[0]
    @property
    def link_goal_pose_dict(self):
        return None if self.link_goal_poses is None else self.link_goal_poses.to_dict()
    def repeat_seeds(self, num_seeds, repeat_seed_idx_buffers=False):
        if num_seeds < 1:
            raise ValueError("num_seeds must be positive")
        out = self.clone()
        def repeat(value):
            if value is None:
                return None
            return value.repeat_interleave(num_seeds, dim=0)
        for name in ("idxs_link_pose", "idxs_goal_js", "idxs_enable", "idxs_env", "idxs_current_js"):
            setattr(out, name, repeat(getattr(self, name)))
        if repeat_seed_idx_buffers:
            out.idxs_seed_goal_js = repeat(self.idxs_seed_goal_js)
        out.num_seeds = self.num_seeds * num_seeds
        return out
    def clone(self):
        # State/index buffers are intentionally shared by the CUDA design;
        # only mutable pose and elapsed-time payloads are independent.
        values = dict(self.__dict__)
        values["link_goal_poses"] = (
            None if self.link_goal_poses is None else self.link_goal_poses.clone()
        )
        values["current_state_dt"] = (
            None if self.current_state_dt is None else self.current_state_dt.clone()
        )
        return type(self)(**values)
    def apply_kernel(self, kernel_mat):
        out = self.clone()
        for name in ("idxs_enable", "idxs_goal_js", "idxs_current_js", "idxs_link_pose", "idxs_env"):
            value = getattr(out, name)
            if value is not None:
                setattr(out, name, (kernel_mat @ value.to(torch.float32)).to(torch.int32))
        return out
    def copy_(self, goal, update_idx_buffers=True, allow_clone=True):
        for name in ("goal_js", "seed_goal_js", "current_js"):
            source = getattr(goal, name)
            target = getattr(self, name)
            if source is not None:
                if target is None:
                    if not allow_clone:
                        raise ValueError(f"{name} has no preallocated buffer")
                    setattr(self, name, source.clone())
                else:
                    target.copy_(source, allow_clone=allow_clone)
        if goal.link_goal_poses is not None:
            if self.link_goal_poses is None:
                self.link_goal_poses = goal.link_goal_poses.clone()
            else:
                self.link_goal_poses.copy_(goal.link_goal_poses)
        for name in ("current_state_dt", "seed_enable_implicit_goal_js"):
            source = getattr(goal, name)
            if source is not None:
                target = getattr(self, name)
                if target is None or target.shape != source.shape:
                    if not allow_clone:
                        raise ValueError(f"{name} has no matching preallocated buffer")
                    setattr(self, name, source.clone())
                else:
                    target.copy_(source)
        if update_idx_buffers and goal.update_idxs_buffers:
            for name in ("idxs_link_pose", "idxs_goal_js", "idxs_current_js", "idxs_seed_goal_js", "idxs_enable", "idxs_env"):
                source = getattr(goal, name)
                if source is not None:
                    target = getattr(self, name)
                    if target is None or target.shape != source.shape:
                        if not allow_clone:
                            raise ValueError(f"{name} has no matching preallocated buffer")
                        setattr(self, name, source.clone())
                    else:
                        target.copy_(source)
        self._update_batch_size()
        return self
    def get_batch_goal_state(self):
        if self.goal_js is None:
            return None
        if self.idxs_link_pose is None:
            return self.goal_js
        return self.goal_js[self.idxs_link_pose[:, 0].to(torch.long)]
    @classmethod
    def create_idx(cls, pose_batch_size, multi_env, num_seeds, device_cfg,
                   seed_goal_state=None, repeat_seed_idx_buffers=False):
        if pose_batch_size < 0 or num_seeds < 1:
            raise ValueError("pose_batch_size must be non-negative and num_seeds positive")
        base = cls._indices(pose_batch_size, device_cfg.device)
        registry = cls(
            batch_size=pose_batch_size,
            num_seeds=1,
            seed_goal_js=seed_goal_state,
            idxs_link_pose=base,
            idxs_goal_js=base.clone(),
            idxs_current_js=base.clone(),
            idxs_env=base.clone() if multi_env else torch.zeros_like(base),
        )
        return registry.repeat_seeds(num_seeds, repeat_seed_idx_buffers)
    def create_index_buffers(self, batch_size, multi_env, num_seeds, device_cfg):
        indexed = type(self).create_idx(batch_size, multi_env, num_seeds, device_cfg)
        indexed.copy_(self, update_idx_buffers=False)
        return indexed
    def get_index_size(self):
        if self.idxs_link_pose is not None:
            return self.idxs_link_pose.shape[0]
        if self.idxs_goal_js is not None:
            return self.idxs_goal_js.shape[0]
        return None

__all__ = ["GoalRegistry", "GoalToolPose", "JointState", "Pose"]
