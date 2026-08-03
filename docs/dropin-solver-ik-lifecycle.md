# Portable IK Solver Lifecycle

`curobo._src.solver.solver_ik.IKSolver` is a real CPU/MPS inverse-kinematics
solver.  It uses deterministic Torch LM seeds followed by the portable
autograd Adam refinement, and returns ranked `[batch, seed, dof]` results.

The facade owns a `SolverCore` instance.  Goal registry updates, world
replacement, tool-pose/joint tracking, inertial updates, seed-manager calls,
and configured eager rollout notification therefore share the same lifecycle
as the other solver facades.  `prepare_action_seeds` returns SolverCore's
`[batch * seed, 1, dof]` transport layout; `solve_pose` keeps its public
result layout unchanged.

`_solve_impl` accepts an already-created `SolveState` for solver composition.
It validates its batch, goal-set and tool-frame metadata before executing the
same portable solve.  `get_unique_solution()` returns first-occurrence,
rounded unique successful configurations from the latest result.

`reset_cuda_graph()` is an explicit unsupported CUDA Graph control surface;
ordinary shape and goal updates still refresh their portable `SolverCore`
state.  The implementation does not claim CUDA Graph handles, Warp kernels,
packed CUDA result buffers, or NVIDIA numerical equivalence.
