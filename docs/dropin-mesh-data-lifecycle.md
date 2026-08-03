# MeshData lifecycle

`MeshData` stores triangular meshes as ordinary PyTorch tensors and uses the
production vectorized triangle-distance operator on CPU and MPS. Mesh cache
IDs are stable portable identifiers, not Warp IDs or BVH handles. The cache
shares immutable local geometry by name across environments; poses and enable
bits remain per-environment.

`load_batch` validates every mesh, transform, and triangle before replacing
the active environment, so malformed later input cannot expose a partially
updated world. Clearing the shared cache is rejected while another environment
still references mesh names; clear all consumers first. Mesh source tensors
and pose tensors must already be on the cache device, preventing hidden CPU to
MPS copies. Poses require finite translation/quaternion values and a nonzero
quaternion norm.

World transforms, enable masks, scene reconstruction, and differentiable
point queries are supported. Raw Warp mesh IDs/structs, BVH construction, and
Warp kernel calls remain explicit unsupported boundaries. Queries use an exact
vectorized triangle scan, so they do not claim Warp BVH performance or its
incidental closest-face tie behavior.
