# Portable JointState operation lifecycle

`curobo._src.state.state_joint_ops` and
`curobo._src.state.state_joint_trajectory_ops` now provide the pinned V2
packing, blending, reordering, augmentation, time scaling, finite-difference,
batch/seed gathering, copying, horizon trimming, and DOF indexing surfaces
with ordinary differentiable PyTorch operations.  All materialized channels
and timing/knot metadata stay device-resident on CPU or Apple Metal.

The only intentional boundary is implementation ABI: CUDA TorchScript helper
kernels and packed CUDA buffer internals are not exposed.  The public tensor
semantics use native PyTorch instead, including MPS with CPU fallback disabled.
