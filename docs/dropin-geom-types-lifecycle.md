# Portable Geometry Value-Model Lifecycle

`curobo._src.geom.types` provides portable CPU/MPS geometry records for scene
collision, mapping, and serialization.  Primitive records validate finite
poses, nonzero quaternions, finite dimensions/radii, and mesh index ranges at
construction.  Integer point/vertex lists are promoted to the portable floating
tensor type while preserving floating tensor graphs and devices.

`tensor_sphere` and `tensor_capsule` accept scalar, `[B]`, or `[B, 1]` radii
and broadcast them with vector inputs.  Their optional output tensors are
updated in place with exact shape checks.  Cuboid helpers validate and normalize
quaternions before emitting the inverse-pose buffers used by collision code.

`SceneCfg.to_dict()` / `as_dict()` emits a deterministic JSON-safe `world_cfg`
mapping, and `SceneCfg.create()` accepts that mapping or a list-of-named-records
form.  Runtime device configuration and derived voxel dtype are intentionally
not serialized; feature dtype is reconstructed from the tensor on load.
`clone()` remains deep for mutable scene data, so collision-world mutation does
not alter a source scene.

Raw Warp structs, mesh IDs, BVH construction/traversal, and trimesh/asset scene
graphs are not emulated.  Portable mesh queries remain vectorized tensor
operations with explicit optional-visualization boundaries.
