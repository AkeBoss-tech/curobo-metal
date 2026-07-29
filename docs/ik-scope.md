# IK reference scope

Wave 3B establishes independent correctness contracts for pose, limit,
smoothness, and primitive-collision costs plus a replayable serial-chain IK
oracle. It proves composition of the earlier FK and collision references for a
hand-checkable two-link arm and a Panda-class seven-DoF chain.

This wave does not claim cuRobo IK API parity or useful runtime performance. It
does not port LBFGS, CUDA Graph execution, production batching, PyTorch
autograd, MPS kernels, trajectory optimization, mesh/voxel collision, or Isaac
integration. The projected finite-difference solver is deliberately unsuitable
for latency measurements; future portable optimizers should be tested against
its problem/result semantics and replay evidence rather than copy its method.

The checked-in cases distinguish reachable, geometrically unreachable,
joint-limit-constrained, and collision-constrained outcomes. A later production
solver may add richer diagnostics, but it must preserve these stable categories
and must never convert a residual or constraint failure into success.
