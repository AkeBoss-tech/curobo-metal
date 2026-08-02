"""Portable goal registry with cuRobo V2 batch/seed indexing semantics.

The registry itself has no CUDA dependency: index tensors are ordinary
``int32`` PyTorch tensors and state payloads remain on their caller-selected
device.  CUDA graph ownership and packed kernel buffers are deliberately left
to the solver layer rather than being faked here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import copy_or_clone, tensor_repeat_seeds

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
    def _indices(size: int, device: torch.device) -> torch.Tensor:
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("index size must be a non-negative integer")
        return torch.arange(size, device=device, dtype=torch.int32).unsqueeze(-1)

    def _update_batch_size(self):
        if self.link_goal_poses is not None:
            self.batch_size = self.link_goal_poses.batch_size
        elif self.goal_js is not None:
            self.batch_size = self.goal_js.position.shape[0]
    @property
    def link_goal_pose_dict(self) -> Optional[Dict[str, Pose]]:
        return None if self.link_goal_poses is None else self.link_goal_poses.to_dict()

    @profiler.record_function("GoalRegistry/repeat_seeds")
    def repeat_seeds(self, num_seeds, repeat_seed_idx_buffers=False):
        if not isinstance(num_seeds, int) or isinstance(num_seeds, bool) or num_seeds < 1:
            raise ValueError("num_seeds must be positive")
        out = self.clone()

        def repeat(value):
            if value is None:
                return None
            return tensor_repeat_seeds(value, num_seeds)

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
    def apply_kernel(self, kernel_mat: torch.Tensor):
        """Map index buffers through a caller-provided batch selection matrix.

        Payloads intentionally remain shared (or cloned according to
        :meth:`clone`); only selection buffers are transformed.  This matches
        V2's use in seed/batch expansion without claiming a packed CUDA
        kernel implementation.
        """
        if not isinstance(kernel_mat, torch.Tensor) or kernel_mat.ndim != 2:
            raise TypeError("kernel_mat must be a rank-2 torch.Tensor")
        index_size = self.get_index_size()
        if index_size is not None and kernel_mat.shape[1] != index_size:
            raise ValueError("kernel_mat width must equal the registry index size")
        first_index = next(
            (getattr(self, name) for name in ("idxs_link_pose", "idxs_goal_js", "idxs_current_js", "idxs_env") if getattr(self, name) is not None),
            None,
        )
        if first_index is not None and kernel_mat.device != first_index.device:
            raise ValueError("kernel_mat and registry indices must be on the same device")
        out = self.clone()
        for name in ("idxs_enable", "idxs_goal_js", "idxs_current_js", "idxs_link_pose", "idxs_env"):
            value = getattr(out, name)
            if value is not None:
                setattr(out, name, (kernel_mat @ value.to(torch.float32)).to(torch.int32))
        return out
    @profiler.record_function("GoalRegistry/copy_")
    def copy_(self, goal: "GoalRegistry", update_idx_buffers=True, allow_clone=True):
        if not isinstance(goal, GoalRegistry):
            raise TypeError("goal must be a GoalRegistry")
        for name in ("goal_js", "seed_goal_js", "current_js"):
            source = getattr(goal, name)
            target = getattr(self, name)
            if source is not None:
                if target is None:
                    if not allow_clone:
                        continue
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
                        continue
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
                            continue
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
        if not isinstance(device_cfg, DeviceCfg):
            raise TypeError("device_cfg must be a DeviceCfg")
        if (not isinstance(pose_batch_size, int) or isinstance(pose_batch_size, bool)
                or pose_batch_size < 0 or not isinstance(num_seeds, int)
                or isinstance(num_seeds, bool) or num_seeds < 1):
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
