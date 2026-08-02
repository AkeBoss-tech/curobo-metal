# Portable utility aliases

The legacy `curobo._src.util` import paths are maintained for configuration,
tensor-buffer, state-filter, deterministic sampling, trajectory, XRDF, and
logging workflows. Their public type aliases and helper imports are exposed
from the same module paths as pinned cuRoboV2, while implementations route to
portable PyTorch CPU/MPS operations.

`LINEAR_CUDA` remains a compatibility name for the differentiable portable
linear interpolator. `BSPLINE_KNOTS_CUDA` and raw Warp interpolation retain
their names but raise `NotImplementedError`, because their pinned behavior
requires CUDA kernels. CUDA graph/stream and Warp objects are likewise not
emulated; callers can use the portable direct executor and tensor operations.

OpenUSD and Viser modules load lazily. Their operations require the relevant
optional dependency and unsupported Isaac/Viser adapter paths fail explicitly.
