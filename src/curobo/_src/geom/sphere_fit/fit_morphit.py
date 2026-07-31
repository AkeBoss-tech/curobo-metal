from __future__ import annotations
from dataclasses import dataclass,field
from typing import Optional,Tuple
import numpy as np
import torch
from .fit_voxel import voxel_fit_mesh

@dataclass
class MorphItLossWeights:
    coverage:float=1000.;protrusion:float=10.;tangency:float=1.;overlap:float=.1;halfplane:float=1000.;protrusion_samples:int=128
@dataclass
class MorphItConfig:
    num_spheres:int=25;device:torch.device=torch.device("cpu");num_inside_samples:int=1000;iterations:int=200
    center_lr:float=.005;radius_lr:float=.001;grad_clip_norm:float=1.;loss_weights:MorphItLossWeights=field(default_factory=MorphItLossWeights)
    density_control_interval:int=20;radius_threshold_ratio:float=.01;coverage_threshold_ratio:float=.05;max_spheres:int=0
    clip_plane:Optional[Tuple[Tuple[float,float,float],float]]=None;clip_plane_buffer:float=.02;verbose_frequency:int=0
    def __post_init__(self):
        if self.max_spheres==0:self.max_spheres=max(self.num_spheres,300)
    def get_radius_threshold(self,mesh):return self.radius_threshold_ratio*float(np.linalg.norm(np.diff(mesh.bounds,axis=0)))
    def get_coverage_threshold(self,mesh):return self.coverage_threshold_ratio*float(np.linalg.norm(np.diff(mesh.bounds,axis=0)))

def morphit_sphere_fit(mesh,num_spheres=None,iterations=200,max_attempts=10,loss_weights=None,device=torch.device("cpu"),init_centers=None,init_radii=None,max_spheres=0,clip_plane=None):
    count=int(num_spheres or (len(init_centers) if init_centers is not None else 25))
    if init_centers is None: centers,radii=voxel_fit_mesh(mesh,count,device)
    else: centers,radii=np.asarray(init_centers,float),np.asarray(init_radii,float)
    if centers is None:return None,None,[]
    centers,radii=centers[:count],radii[:count]
    if clip_plane is not None:
        n,offset=clip_plane;n=np.asarray(n,float); radii=np.minimum(radii,np.maximum(0,centers@n-offset))
    return centers,radii,[(centers.copy(),radii.copy())]
__all__=["MorphItConfig","MorphItLossWeights","morphit_sphere_fit"]
