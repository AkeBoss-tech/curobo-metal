# Optimizer and inverse-kinematics compatibility

The `curobo.optim` and `curobo.inverse_kinematics` imports now resolve without
CUDA, Warp, or Isaac dependencies. The pinned V2 configuration names and
constructor signatures are preserved for L-BFGS, MPPI, evolution strategies,
PyTorch/SciPy wrappers, multi-stage optimization, solve modes, IK configuration,
and IK results.

The implementations keep tensors resident on CPU or MPS. L-BFGS and particle
optimizers use the production `curobo_metal.optim` engines and retain reusable
shape-keyed state across calls. The IK facade loads the packaged Franka YAML,
uses the production differentiable whole-body kinematics implementation, and
performs deterministic multi-seed projected optimization.

CUDA Graph capture and fused CUDA optimizer kernels are platform-specific
implementation details, not portable semantics. A low-level request to capture
a CUDA graph raises `UnsupportedOptimizerFeature`. The high-level
`InverseKinematicsCfg.create` accepts the upstream default
`use_cuda_graph=True` but compiles it to the persistent portable execution
cache, allowing dependency-only application changes on Apple systems.

Current bounded gaps are collision-aware IK through arbitrary external
`SceneCollision` instances, multi-tool/goalset IK, velocity-aware IK, CUDA
debug-dump internals, and exact CUDA numerical replay. These gaps must remain
visible in the release inventory until implemented and paired against the
pinned NVIDIA runner.
