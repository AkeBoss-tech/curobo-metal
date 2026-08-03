# State base and filter coefficients

`curobo._src.state.state_base.State` is the pinned V2 empty dataclass that
also acts as the abstract `Sequence` base for portable state records.  Concrete
objects such as `JointState` own regular PyTorch tensors and therefore retain
normal CPU/MPS placement and autograd behavior.

`FilterCoeff` retains the pinned four mutable scalar fields (`position`,
`velocity`, `acceleration`, and `jerk`) with zero defaults.  Scalars are used
directly in PyTorch arithmetic, so a coefficient record adds no hidden CPU or
CUDA allocation when filtering an MPS state.  Coefficients are intentionally
not clamped: values outside `[0, 1]`, including negative values, preserve the
upstream extrapolation behavior.

This portable surface does not expose a CUDA packed coefficient buffer or CUDA
graph capture.  Those are implementation details rather than part of the
state value-object contract.
