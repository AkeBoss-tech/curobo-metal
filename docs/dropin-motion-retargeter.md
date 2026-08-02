# Motion retargeter compatibility

`curobo.motion_retargeter` provides the pinned V2 stateful retargeting
surface on PyTorch CPU and Apple MPS.  A retargeter uses global multi-seed IK
for its first frame, then uses velocity-aware warm-started local IK.  With
`use_mpc=True`, the first global solution initializes a portable
receding-horizon MPC instance and following frames emit the executed endpoint
stream as `RetargetResult.trajectory`.

The configuration preserves ordered tool criteria, environment batch capacity,
collision-sphere loading invariants, step/time tolerances, and reset behavior.
`solve_sequence` resets first and returns states arranged as
`[environment, frame, dof]`.

The portable MPC retargeting adapter currently supports one tracked tool link;
multi-link clips work through warm-started IK.  Its Cartesian target is
resolved by the production portable IK solver before trajectory optimization.
That is functional CPU/MPS planning, not a claim of identical CUDA rollout
control, raw CUDA graph capture, Warp kernels, or Isaac humanoid integration.
