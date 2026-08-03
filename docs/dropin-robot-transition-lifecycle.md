# Portable Robot State Transition Lifecycle

`curobo._src.transition.robot_state_transition` and
`robot_state_transition_cfg` provide the V2 transition/configuration surface
using differentiable PyTorch tensor arithmetic on CPU and Apple MPS.

The implementation supports position (including teleport), velocity,
acceleration, and portable interpolated B-spline control spaces; state-table
selection; command-buffer sizing; typed `TimeTrajCfg` construction; schedule
updates; command filtering; optional whole-body kinematics and inverse
dynamics augmentation; bounds; and mutable inertial updates when a portable
dynamics configuration is supplied. Public `traj_dt`, `dt`, `state_seq`,
`joint_limits`, and action-order lifecycle fields remain device-resident
ordinary tensors/records. Velocity and acceleration action integration is
first- and second-order respectively and preserves first-order autograd.

Configuration validates positive finite timesteps, a `base_ratio` in `[0, 1]`,
positive batch/horizon sizes, control-space values, and device/state/action
agreement. This prevents accidental CPU/MPS copies rather than allowing an
implicit fallback.

CUDA streams, graph capture, packed state buffers, and raw CUDA/Warp kernels
are intentionally not emulated. The persistent state sequence is portable
metadata/buffer lifecycle only; a differentiable forward returns new PyTorch
state values and does not retain a prior optimizer graph.
