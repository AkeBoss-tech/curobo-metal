# Geometry values and transforms

`curobo._src.geom.types` and `curobo._src.geom.transform` provide the pinned
cuRobo V2 value-level geometry interface on CPU and Apple MPS through ordinary
PyTorch operations.  Rigid transforms, point transforms, quaternion/matrix
conversion, analytic primitive meshes, voxel values, world cloning, and
first-order autograd are supported.

The matrix helpers retain their distinct signatures: `get_inv_transform(rot,
trans)` and `transform_point_inverse(point, rot, trans)` operate on rotation
matrices, while `pose_inverse(position, quaternion)` operates on `wxyz` pose
values.  Direct uses of the pinned autograd Function facades also have
first-order PyTorch backward support.

Raw Warp kernel entry points, CUDA output/adjoint buffer ABI details,
trimesh-based asset loading, and CUDA/NVIDIA numerical equivalence remain out
of scope.  The portable aliases are mathematical PyTorch helpers, not Warp
kernel objects.
