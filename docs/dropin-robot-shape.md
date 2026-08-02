# Portable robot-model API closure

The `curobo._src.robot` runtime namespace runs its kinematics and rigid-body
dynamics through the native PyTorch CPU/MPS tree backend.  This closure wave
adds mutable tensor views to `KinematicsParams`, including joint limits, link
and tool index maps, fixed transforms, tree depth/width, sphere activation,
and independent cloning.  `Kinematics.get_link_poses` now accepts any link in
the compiled tree, rather than only configured tool links, and remains
differentiable for batched CPU/MPS tensors.

`KinematicsCfg.from_config` accepts a `KinematicsLoaderCfg`, and a loader with
collision spheres generates deterministic self-collision pair metadata.  The
explicit boundaries remain unchanged: raw CUDA-packed tensors, Isaac/USD
loading, mesh fitting/mesh visualization, arbitrary-axis unsupported joints,
and CUDA ABI kernels are not provided.  They raise precise errors rather than
silently falling back or claiming CUDA equivalence.

The executable contract is
`tests/dropin/robot_shape/test_robot_shape.py`; it covers mutation,
independent cloning, arbitrary-link FK/autograd, loader-to-config construction,
and fallback-disabled MPS when available.
