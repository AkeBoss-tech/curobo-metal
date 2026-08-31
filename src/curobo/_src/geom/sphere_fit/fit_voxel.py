from __future__ import annotations
from typing import Optional,Tuple
import numpy as np
import torch
from . import _trimesh_compat as trimesh
from .wp_mesh_query import WarpMeshQuery
from curobo._src.util.logging import log_warn

def sample_even_fit_mesh(mesh: trimesh.Trimesh,num_spheres:int,sphere_radius:float)->Tuple[np.ndarray,np.ndarray]:
    if num_spheres<1 or sphere_radius<0:raise ValueError("num_spheres must be positive and radius nonnegative")
    # Deterministic farthest-point sampling avoids trimesh's global RNG.
    pts=np.asarray(mesh.vertices,float); chosen=[int(np.lexsort((pts[:,2],pts[:,1],pts[:,0]))[0])]
    near=np.full(len(pts),np.inf)
    while len(chosen)<min(num_spheres,len(pts)):
        near=np.minimum(near,np.linalg.norm(pts-pts[chosen[-1]],axis=1));chosen.append(int(np.argmax(near)))
    out=pts[chosen]
    if len(out)<num_spheres:out=np.concatenate((out,np.repeat(out[-1:],num_spheres-len(out),0)))
    return out,np.full(num_spheres,sphere_radius)

def _build_bbox_grid(mesh,num_spheres):
    if num_spheres<1:raise ValueError("num_spheres must be positive")
    lo,hi=np.asarray(mesh.bounds); ext=hi-lo; volume=float(np.prod(ext))
    if volume<=0:return np.zeros((0,3))
    pitch=(volume/num_spheres)**(1/3)
    axes=[np.linspace(lo[i]+pitch/2,hi[i]-pitch/2,max(int(np.ceil(ext[i]/pitch)),1)) for i in range(3)]
    return np.stack(np.meshgrid(*axes,indexing="ij"),-1).reshape(-1,3)

def voxel_fit_mesh(
    mesh: trimesh.Trimesh,
    num_spheres: int,
    device: torch.device = torch.device("cuda", 0),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pts=_build_bbox_grid(mesh,num_spheres)
    if not len(pts):return None,None
    query=WarpMeshQuery(mesh,device); d,_=query.query_sdf(torch.as_tensor(pts,dtype=torch.float32,device=device))
    d=d.detach().cpu().numpy(); keep=d<0; pts,r=pts[keep],-d[keep]
    if not len(pts):return None,None
    order=np.argsort(-r,kind="stable")[:num_spheres];return pts[order],r[order]
__all__=["sample_even_fit_mesh","voxel_fit_mesh"]
