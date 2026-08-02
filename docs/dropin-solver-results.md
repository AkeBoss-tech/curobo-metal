# Solver result lifecycle compatibility

`TrajOptSolverResult` and `MPCSolverResult` are portable CPU/MPS value
objects built on `BaseSolverResult`.  They retain the pinned V2 dataclass
fields, ordinary tensor layouts, cloning, device conversion, and the result
merge operations used by solver retries.

- TrajOpt uses `[batch, seed, horizon, dof]` plan data.  Full-batch and
  successful-seed merges copy both the normal rollout and the interpolated
  trajectory, including every materialised derivative/timing channel and the
  recorded final interpolation step.  `motion_time()` is derived from the
  selected rollout `JointState.dt`; a maximum dt is only used when no rollout
  timing is available.
- MPC uses `[batch, horizon, dof]` action plans.  Full-batch and successful
  merges include `next_action`, current/full action sequences, robot-state
  sequences, and action buffers.  `action_at()` returns a standalone command
  without advancing solver state and retains known per-command timing.
- `clone()` and `to(DeviceCfg(...))` do not alias tensor payloads; floating
  data adopts the requested dtype while bool/index tensors retain theirs.

Raw CUDA graph state, packed rollout/action result buffers, and opaque CUDA
robot-model buffer transfer are intentionally unavailable.  The portable
interfaces fail explicitly rather than claiming those ABI semantics on Metal.
