from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from ._portable import PortableObstacleData, PortableWarpStruct, inverse_pose, raw_warp

class CuboidDataWarp(PortableWarpStruct): pass
@dataclass(init=False)
class CuboidData(PortableObstacleData):
    @classmethod
    def create_cache(cls,max_n:int,num_envs:int,device_cfg:DeviceCfg):
        o=cls._base(max_n,num_envs,device_cfg)
        o.dims=torch.full((num_envs,max_n,4),.01,**device_cfg.as_torch_dict()); return o
    @classmethod
    def from_scene_cfg(cls,scene_cfg,device_cfg,env_idx=0,num_envs=1,max_n=None):
        o=cls.create_cache(max_n or max(len(scene_cfg.cuboid),1),num_envs,device_cfg); o.load_batch(scene_cfg.cuboid,env_idx); return o
    @classmethod
    def from_batch_scene_cfg(cls,scene_cfg_list,device_cfg,max_n=None):
        o=cls.create_cache(max_n or max([len(x.cuboid) for x in scene_cfg_list]+[1]),len(scene_cfg_list),device_cfg)
        for i,s in enumerate(scene_cfg_list): o.load_batch(s.cuboid,i)
        return o
    def load_batch(self,cuboids:List[Cuboid],env_idx:int):
        if len(cuboids)>self.max_n: raise ValueError("cuboid cache capacity exceeded")
        self.clear(env_idx)
        for c in cuboids: self.add(c,env_idx)
    def add(self,cuboid:Cuboid,env_idx=0):
        i=self.get_active_count(env_idx)
        if i>=self.max_n: raise ValueError("cuboid cache capacity exceeded")
        self.names[env_idx][i]=cuboid.name; self.dims[env_idx,i,:3]=torch.as_tensor(cuboid.dims,**self.device_cfg.as_torch_dict())
        self.inv_pose[env_idx,i,:7]=inverse_pose(cuboid.pose,self.device_cfg); self.enable[env_idx,i]=1; self.count[env_idx]+=1; return i
    def add_from_raw(self,name,dims,env_idx,w_obj_pose=None,obj_w_pose=None):
        c=Cuboid(name=name,pose=[0,0,0,1,0,0,0],dims=dims.detach().cpu().tolist()); i=self.add(c,env_idx); self.update_pose(name,w_obj_pose,obj_w_pose,env_idx); return i
    def update_dims(self,name,dims,env_idx=0): self.dims[env_idx,self.get_idx(name,env_idx),:3]=dims
is_obs_enabled=load_obstacle_transform=compute_local_sdf=compute_local_sdf_with_grad=raw_warp
__all__=["CuboidData","CuboidDataWarp","is_obs_enabled","load_obstacle_transform","compute_local_sdf","compute_local_sdf_with_grad"]
