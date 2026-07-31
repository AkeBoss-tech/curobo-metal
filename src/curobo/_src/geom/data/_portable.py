"""Shared storage implementation for portable obstacle datasets."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional
import torch
from curobo._src.types.device_cfg import DeviceCfg

def inverse_pose(value, cfg):
    p = torch.as_tensor(value, **cfg.as_torch_dict())
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
    def set_enabled(self, name, enabled, env_idx=0): self.enable[env_idx,self.get_idx(name,env_idx)] = int(enabled)
    def clear(self, env_idx=None):
        ids = range(self.num_envs) if env_idx is None else [env_idx]
        for i in ids:
            self.enable[i].zero_(); self.count[i]=0; self.names[i]=[None]*self.max_n
    def update_pose(self, name, w_obj_pose=None, obj_w_pose=None, env_idx=0):
        idx=self.get_idx(name,env_idx)
        if obj_w_pose is not None:
            p=torch.cat((obj_w_pose.position.reshape(-1,3)[0],obj_w_pose.quaternion.reshape(-1,4)[0]))
        elif w_obj_pose is not None:
            p=inverse_pose(torch.cat((w_obj_pose.position.reshape(-1,3)[0],w_obj_pose.quaternion.reshape(-1,4)[0])),self.device_cfg)
            self.inv_pose[env_idx,idx,:7]=p; return
        else: raise ValueError("w_obj_pose or obj_w_pose is required")
        self.inv_pose[env_idx,idx,:7]=p
    def to_warp(self, *args, **kwargs): raise NotImplementedError("Warp is unavailable on the portable backend")

def raw_warp(*args, **kwargs):
    raise NotImplementedError("raw Warp collision functions are unavailable; use SceneCollision")
