"""Torch replacement for the upstream Warp mesh-query wrapper."""
from __future__ import annotations
import torch
from curobo_metal.ops.world_collision import Mesh as TorchMesh, mesh_distance

class WarpMeshQuery:
    def __init__(self, mesh, device=torch.device("cpu")):
        self.device=torch.device(device)
        self.mesh=TorchMesh(torch.as_tensor(mesh.vertices,dtype=torch.float32,device=self.device),
                            torch.as_tensor(mesh.faces,dtype=torch.int64,device=self.device),
                            bool(getattr(mesh,"is_watertight",False)))
    def query_sdf(self, points):
        eye=torch.eye(3,dtype=points.dtype,device=points.device).reshape(1,1,3,3)
        zero=torch.zeros((1,1,3),dtype=points.dtype,device=points.device)
        result=mesh_distance(points,[self.mesh],zero,eye,signed=self.mesh.watertight)
        d=result.reduced_distance.squeeze(0) if result.reduced_distance.ndim>1 and points.ndim==2 else result.reduced_distance
        g=result.reduced_gradient.squeeze(0) if result.reduced_gradient.ndim>2 and points.ndim==2 else result.reduced_gradient
        return d,g
    def query_outside_mask(self,points): return self.query_sdf(points)[0] > 0
    def query_closest_point(self,points):
        distance,gradient=self.query_sdf(points)
        return points-distance.unsqueeze(-1)*gradient,distance

class WarpSphereSDFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, query):
        distance,gradient=query.query_sdf(points);ctx.save_for_backward(gradient);return distance
    @staticmethod
    def backward(ctx, grad):
        (gradient,)=ctx.saved_tensors;return grad.unsqueeze(-1)*gradient,None
__all__=["WarpMeshQuery","WarpSphereSDFFunction"]
