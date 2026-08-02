# Portable projective texture mapping

`curobo._src.perception.mapper.projector_texture.ProjectiveTextureProjector`
provides CPU and Metal implementations of the useful high-level RGB-D texture
workflow: source-shaped camera batch validation, pinhole projection,
depth-consistency visibility tests, deterministic nearest-view selection,
atlas packing, mesh colors/UVs, and occupied-voxel colors.

The portable path accepts `CameraObservation` records with uint8 RGB images,
floating intrinsics, `Pose` camera transforms, and optional camera-frame depth
in metres. If depth is absent it uses the supplied mapper renderer. Texture
sampling is nearest pixel, with the source adaptive tolerance of the larger of
two voxels or one percent of projected depth (or an explicit tolerance).

It intentionally does not expose CUDA texture objects, Warp launches, raw
sparse-TSDF buffers, or the upstream per-triangle Warp score/fallback-atlas
kernel. Those low-level ABIs raise through their owning CUDA/Warp components;
the high-level projector is an ordinary PyTorch CPU/MPS implementation.
