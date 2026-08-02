# Seed IK compatibility

`curobo._src.solver.seed_ik` provides a deterministic Levenberg--Marquardt
seed solver over the portable tree FK/Jacobian backend.  It supports single and
batched one-timestep tool-pose IK, multiple seed returns, goal sets, bounded
joint updates, and optional velocity/acceleration regularisation.  CPU and
float32 MPS use the same composed PyTorch math and preserve device residency.

The implementation intentionally does not expose CUDA graph capture, Warp
tile-kernel ABI, CUDA streams, or exact NVIDIA numerical parity.  A requested
`use_cuda_graph=True` remains accepted as an upstream-compatible configuration
default but executes as ordinary portable tensor work; result metrics identify
this as `backend: "torch-lm"` and `cuda_graph: false`.

Goal-set ranking is deterministic: every candidate is solved, then ties select
the lower goal-set index and lower seed index.  Raw cost-gradient buffer APIs
from the CUDA implementation are not emulated; the public error calculator
uses the geometric Jacobian directly.
