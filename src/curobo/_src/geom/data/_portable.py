"""Shared storage implementation for portable obstacle datasets."""
from __future__ import annotations
from typing import Any
import torch
from curobo._src.types.device_cfg import DeviceCfg

def pose_vector(value, cfg):
    """Return one ``xyzw``-position / ``wxyz``-quaternion pose row.

    The public data stores accept both cuRobo :class:`Pose` values and the
    serialisable seven-value world representation.  Keeping this conversion in
    tensor space is important: pose updates remain on MPS and preserve a
    caller's autograd graph until it reaches a mutable cache buffer.
    """
    if hasattr(value, "get_pose_vector"):
        value = value.get_pose_vector()
    elif hasattr(value, "position") and hasattr(value, "quaternion"):
        value = torch.cat((value.position, value.quaternion), dim=-1)
    p = torch.as_tensor(value, **cfg.as_torch_dict())
    if p.shape[-1:] != (7,):
        raise ValueError("pose must end in [x, y, z, qw, qx, qy, qz]")
    return p.reshape(-1, 7)[0]


def inverse_pose(value, cfg):
    """Return the inverse of one portable pose row without CPU/Warp calls."""
    p = pose_vector(value, cfg)
    q = p[3:7] / p[3:7].norm().clamp_min(1e-12)
    qi = q * q.new_tensor([1, -1, -1, -1])
    w, x, y, z = qi
    r = torch.stack((torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w))),
                     torch.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w))),
                     torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))))
    return torch.cat((-(r @ p[:3]), qi))

class PortableWarpStruct:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("raw Warp structs are unavailable; use the portable torch dataset")

class PortableObstacleData:
    kind = ""
    @classmethod
    def _base(cls, max_n, num_envs, cfg):
        obj = cls.__new__(cls)
        obj.enable = torch.zeros((num_envs,max_n), dtype=torch.uint8, device=cfg.device)
        obj.count = torch.zeros(num_envs, dtype=torch.int32, device=cfg.device)
        obj.inv_pose = torch.zeros((num_envs,max_n,8), **cfg.as_torch_dict()); obj.inv_pose[...,3]=1
        obj.names = [[None]*max_n for _ in range(num_envs)]
        obj.max_n, obj.num_envs, obj.device_cfg = max_n, num_envs, cfg
        return obj
    def get_idx(self, name, env_idx=0):
        try: return self.names[env_idx].index(name)
        except ValueError as e: raise ValueError(f"obstacle {name!r} not found in environment {env_idx}") from e
    def has_name(self, name, env_idx=0): return name in self.names[env_idx]
    def get_active_count(self, env_idx=0): return int(self.count[env_idx].item())
    def get_names(self, env_idx=0): return [x for x in self.names[env_idx] if x is not None]
    def _check_env(self, env_idx):
        if not 0 <= int(env_idx) < self.num_envs:
            raise IndexError(f"environment index {env_idx} is outside [0, {self.num_envs})")

    def set_enabled(self, name, enabled, env_idx=0):
        self._check_env(env_idx)
        self.enable[env_idx,self.get_idx(name,env_idx)] = int(bool(enabled))
    def clear(self, env_idx=None):
        ids = range(self.num_envs) if env_idx is None else [env_idx]
        for i in ids:
            self._check_env(i)
            self.enable[i].zero_(); self.count[i]=0; self.names[i]=[None]*self.max_n
    def update_pose(self, name, w_obj_pose=None, obj_w_pose=None, env_idx=0):
        self._check_env(env_idx)
        idx=self.get_idx(name,env_idx)
        if obj_w_pose is not None:
            p=pose_vector(obj_w_pose, self.device_cfg)
        elif w_obj_pose is not None:
            p=inverse_pose(w_obj_pose,self.device_cfg)
        else: raise ValueError("w_obj_pose or obj_w_pose is required")
        self.inv_pose[env_idx,idx,:7]=p
    def to_warp(self, *args, **kwargs): raise NotImplementedError("Warp is unavailable on the portable backend")

def raw_warp(*args, **kwargs):
    raise NotImplementedError("raw Warp collision functions are unavailable; use SceneCollision")
