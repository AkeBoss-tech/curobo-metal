import torch


def extract_mesh_block_sparse(tsdf, level=0.0, surface_only=False, refine_iterations=0, minimum_tsdf_weight=0.1):
    if hasattr(tsdf, "extract_mesh"):
        mesh = tsdf.extract_mesh()
        normals = torch.zeros_like(mesh.vertices)
        return mesh.vertices, mesh.faces, normals
    raise NotImplementedError("use Mapper.extract_mesh for the portable dense backend")
