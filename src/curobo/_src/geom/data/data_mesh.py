from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from curobo._src.geom.types import Mesh
from ._portable import PortableObstacleData, PortableWarpStruct, inverse_pose, raw_warp

@dataclass(frozen=True)
class WarpMeshCache:
    name: str
    mesh_id: Optional[int]
    vertices: torch.Tensor
    faces: torch.Tensor
    mesh: object = None
    def get_bounds(self): return self.vertices.amin(0), self.vertices.amax(0)

class MeshDataWarp(PortableWarpStruct): pass
@dataclass(init=False)
class MeshData(PortableObstacleData):
    @classmethod
    def create_cache(cls,max_n,num_envs,device_cfg,max_dist=0.1):
        if max_n < 1 or num_envs < 1:
            raise ValueError("max_n and num_envs must be positive")
        o=cls._base(max_n,num_envs,device_cfg)
        o.mesh_ids=torch.zeros((num_envs,max_n),dtype=torch.int64,device=device_cfg.device)
        o.dims=torch.zeros((num_envs,max_n,4),**device_cfg.as_torch_dict())
        o._mesh_cache={};o.wp_cache=o._mesh_cache;o.max_dist=float(max_dist);o._wp_device=None; return o
    @classmethod
    def from_scene_cfg(cls,scene_cfg,device_cfg,env_idx=0,num_envs=1,max_n=None,max_dist=0.1):
        o=cls.create_cache(max_n or max(len(scene_cfg.mesh),1),num_envs,device_cfg,max_dist); o.load_batch(scene_cfg.mesh,env_idx); return o
    @classmethod
    def from_batch_scene_cfg(cls,scene_cfg_list,device_cfg,max_n=None,max_dist=0.1):
        o=cls.create_cache(max_n or max([len(x.mesh) for x in scene_cfg_list]+[1]),len(scene_cfg_list),device_cfg,max_dist)
        for i,s in enumerate(scene_cfg_list): o.load_batch(s.mesh,i)
        return o
    def _load_mesh_into_cache(self,mesh):
        if mesh.vertices is None or mesh.faces is None: raise NotImplementedError("file-backed mesh loading requires trimesh data to be supplied")
        entry=WarpMeshCache(mesh.name,None,torch.as_tensor(mesh.vertices,**self.device_cfg.as_torch_dict()),torch.as_tensor(mesh.faces,dtype=torch.int64,device=self.device_cfg.device))
        self._mesh_cache[mesh.name]=entry; return entry
    _load_mesh_to_warp=_load_mesh_into_cache
    def load_batch(self,meshes,env_idx):
        if len(meshes)>self.max_n: raise ValueError("mesh cache capacity exceeded")
        self.clear(env_idx)
        for m in meshes:self.add(m,env_idx)
    def add(self,mesh:Mesh,env_idx=0):
        i=self.get_active_count(env_idx)
        if i>=self.max_n: raise ValueError("mesh cache capacity exceeded")
        self._load_mesh_into_cache(mesh); self.names[env_idx][i]=mesh.name
        self.mesh_ids[env_idx,i]=i; lo,hi=self._mesh_cache[mesh.name].get_bounds();self.dims[env_idx,i,:3]=hi-lo
        self.inv_pose[env_idx,i,:7]=inverse_pose(mesh.pose or [0,0,0,1,0,0,0],self.device_cfg)
        self.enable[env_idx,i]=1; self.count[env_idx]+=1; return i
    def update_pose(self, name, w_obj_pose=None, obj_w_pose=None, env_idx=0):
        return super().update_pose(name, w_obj_pose, obj_w_pose, env_idx)
    def get_cached_mesh_names(self): return list(self._mesh_cache)
    def update_from_warp_id(self,warp_mesh_id,name,w_obj_pose=None,obj_w_pose=None,env_idx=0,mesh_idx=None):
        raise NotImplementedError(
            "Warp mesh ids are unavailable on the portable backend; create Mesh with vertices/faces instead"
        )
    def clear(self,env_idx=None,clear_warp_cache=False):
        super().clear(env_idx)
        if clear_warp_cache:self._mesh_cache.clear()
is_obs_enabled=load_obstacle_transform=compute_local_sdf=compute_local_sdf_with_grad=raw_warp
__all__=["MeshData","MeshDataWarp","WarpMeshCache","is_obs_enabled","load_obstacle_transform","compute_local_sdf","compute_local_sdf_with_grad"]
