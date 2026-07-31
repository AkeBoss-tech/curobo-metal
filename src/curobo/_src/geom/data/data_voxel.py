from __future__ import annotations
from dataclasses import dataclass
import torch
from curobo._src.geom.types import VoxelGrid
from ._portable import PortableObstacleData, PortableWarpStruct, inverse_pose, raw_warp

class VoxelDataWarp(PortableWarpStruct): pass
@dataclass(init=False)
class VoxelData(PortableObstacleData):
    @classmethod
    def create_cache(cls,max_n,num_envs,device_cfg,max_voxels=1):
        o=cls._base(max_n,num_envs,device_cfg); o.max_voxels=max_voxels
        o.features=torch.zeros((num_envs,max_n,max_voxels),**device_cfg.as_torch_dict())
        o.xyzr=torch.zeros((num_envs,max_n,max_voxels,4),**device_cfg.as_torch_dict())
        o.params=torch.zeros((num_envs,max_n,4),**device_cfg.as_torch_dict())
        o.dims=torch.zeros((num_envs,max_n,4),**device_cfg.as_torch_dict()); o.max_esdf_distance=100.;o._grids={}; return o
    @classmethod
    def from_voxel_grid(cls,voxel_grid,device_cfg,env_idx=0,num_envs=1,max_n=1):
        n=voxel_grid.feature_tensor.numel() if voxel_grid.feature_tensor is not None else 1
        o=cls.create_cache(max_n,num_envs,device_cfg,n); o.load_batch([voxel_grid],env_idx); return o
    @classmethod
    def from_scene_cfg(cls,scene_cfg,device_cfg,env_idx=0,num_envs=1,max_n=None):
        grids=scene_cfg.voxel; n=max([g.feature_tensor.numel() if g.feature_tensor is not None else 1 for g in grids]+[1])
        o=cls.create_cache(max_n or max(len(grids),1),num_envs,device_cfg,n); o.load_batch(grids,env_idx); return o
    @classmethod
    def from_batch_scene_cfg(cls,scene_cfg_list,device_cfg,max_n=None):
        grids=[g for s in scene_cfg_list for g in s.voxel]; n=max([g.feature_tensor.numel() if g.feature_tensor is not None else 1 for g in grids]+[1])
        o=cls.create_cache(max_n or max([len(s.voxel) for s in scene_cfg_list]+[1]),len(scene_cfg_list),device_cfg,n)
        for i,s in enumerate(scene_cfg_list):o.load_batch(s.voxel,i)
        return o
    def load_batch(self,grids,env_idx):
        if len(grids)>self.max_n:raise ValueError("voxel cache capacity exceeded")
        self.clear(env_idx)
        for g in grids:self.update_data(g,env_idx)
    def update_data(self,g:VoxelGrid,env_idx=0,name=None):
        key=name or g.name
        if self.has_name(key,env_idx):i=self.get_idx(key,env_idx)
        else:
            i=self.get_active_count(env_idx)
            if i>=self.max_n:raise ValueError("voxel cache capacity exceeded")
            self.names[env_idx][i]=key;self.count[env_idx]+=1
        if g.feature_tensor is not None:self.features[env_idx,i,:g.feature_tensor.numel()]=g.feature_tensor.reshape(-1).to(self.device_cfg.device)
        if g.xyzr_tensor is not None:self.xyzr[env_idx,i,:g.xyzr_tensor.reshape(-1,4).shape[0]]=g.xyzr_tensor.reshape(-1,4).to(self.device_cfg.device)
        self.dims[env_idx,i,:3]=torch.as_tensor(g.dims,**self.device_cfg.as_torch_dict());self.dims[env_idx,i,3]=g.voxel_size
        self.params[env_idx,i]=self.dims[env_idx,i]
        self.inv_pose[env_idx,i,:7]=inverse_pose(g.pose or [0,0,0,1,0,0,0],self.device_cfg);self.enable[env_idx,i]=1;self._grids[(env_idx,key)]=g
    def update_features(self,features,name,env_idx=0):
        i=self.get_idx(name,env_idx);self.features[env_idx,i,:features.numel()]=features.reshape(-1)
    def get_voxel_grid(self,name,env_idx=0):return self._grids[(env_idx,name)]
    def get_grid_shape(self,env_idx=0,name=None,idx=0):
        if name is not None:idx=self.get_idx(name,env_idx)
        d=self.dims[env_idx,idx];return torch.Size([round(float(d[j]/d[3])) for j in range(3)])
is_obs_enabled=load_obstacle_transform=compute_local_sdf=compute_local_sdf_with_grad=raw_warp
__all__=["VoxelData","VoxelDataWarp","is_obs_enabled","load_obstacle_transform","compute_local_sdf","compute_local_sdf_with_grad"]
