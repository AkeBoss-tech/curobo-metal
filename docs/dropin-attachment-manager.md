# Attachment Manager compatibility

`curobo._src.collision.attachment_manager.AttachmentManager` supports the
pinned V2 `fit_spheres`, `update`, `attach`, `attach_from_scene`, and `detach`
methods on CPU and Apple Metal.  It fits primitive/serialized mesh vertex data
to deterministic PyTorch spheres, resolves a payload's link-local pose through
batched production FK, and mutates the portable `KinematicsParams` sphere bank
used by collision-aware planning.

One attachment manager owns one payload lifecycle.  Re-attaching first resets
the previous link's reference spheres and re-enables scene obstacles disabled by
that manager.  World-object names are validated in every requested environment
before any world enable bit is changed.  A single object pose is broadcast over
a batched joint state; alternatively provide one pose per environment.

The object fitting path intentionally does not require `trimesh`, Warp, or a
CUDA sphere-fitting kernel.  Its deterministic primitive/vertex approximation
is useful for portable planning but is not a claim of numerical equivalence to
V2's optional MorphIt/Warp mesh pipeline.  `_obstacles_to_trimesh` remains an
explicit optional-dependency boundary.
