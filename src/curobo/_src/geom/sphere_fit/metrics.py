from __future__ import annotations
import numpy as np
import torch
from .types import SphereFitMetrics,SphereFitResult
from .wp_mesh_query import WarpMeshQuery

def _directions(n,device):
    i=torch.arange(n,dtype=torch.float32,device=device)+.5
    z=1-2*i/n; phi=i*(np.pi*(3-np.sqrt(5)));r=torch.sqrt((1-z*z).clamp_min(0))
    return torch.stack((r*torch.cos(phi),r*torch.sin(phi),z),-1)
def compute_sphere_fit_metrics(mesh,centers,radii,n_interior=10000,n_surface=5000,n_sphere_surface=200,device=torch.device("cpu")):
    c=torch.as_tensor(centers,dtype=torch.float32,device=device);r=torch.as_tensor(radii,dtype=torch.float32,device=device).flatten()
    if not len(c):return SphereFitMetrics(num_spheres=0,protrusion=1.,surface_gap_mean=float("inf"),surface_gap_p95=float("inf"),max_uncovered_gap=float("inf"))
    # Fixed grid gives reproducible interior coverage.
    lo,hi=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in mesh.bounds]
    side=max(2,int(round(n_interior**(1/3)))); axes=[torch.linspace(lo[j],hi[j],side,device=device) for j in range(3)]
    samples=torch.stack(torch.meshgrid(*axes,indexing="ij"),-1).reshape(-1,3)[:n_interior]
    q=WarpMeshQuery(mesh,device); sd,_=q.query_sdf(samples);inside=samples[sd<0]
    coverage=float(((torch.cdist(inside,c)-r).amin(-1)<=0).float().mean()) if len(inside) else 0.
    verts=torch.as_tensor(mesh.vertices,dtype=torch.float32,device=device)
    surface=verts[torch.arange(n_surface,device=device)%len(verts)]
    gaps=(torch.cdist(surface,c)-r).amin(-1).clamp_min(0)
    sphere_points=(c[:,None]+r[:,None,None]*_directions(n_sphere_surface,device)[None]).reshape(-1,3)
    out_sd,_=q.query_sdf(sphere_points);outside=out_sd>0;od=out_sd[outside]
    sphere_vol=float((4/3*np.pi*r.pow(3).sum()).item());mesh_vol=max(float(abs(getattr(mesh,"volume",0))),1e-12)
    return SphereFitMetrics(num_spheres=len(c),coverage=coverage,protrusion=float(outside.float().mean()),
        protrusion_dist_mean=float(od.mean()) if len(od) else 0.,protrusion_dist_p95=float(torch.quantile(od,.95)) if len(od) else 0.,
        surface_gap_mean=float(gaps.mean()),surface_gap_p95=float(torch.quantile(gaps,.95)),max_uncovered_gap=float(gaps.max()),volume_ratio=sphere_vol/mesh_vol)
def populate_metrics(result:SphereFitResult,mesh,**kwargs):
    result.metrics=compute_sphere_fit_metrics(mesh,result.centers,result.radii,**kwargs);return result.metrics
__all__=["compute_sphere_fit_metrics","populate_metrics"]
