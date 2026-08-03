# Kinematics reducer compatibility

`curobo._src.robot.kinematics.kinematics_reducer.KinematicsReducer` reduces a
portable `KinematicsParams` tree for CPU and Metal execution.  It keeps the
requested links and their full base-to-link ancestry, subsets C-space/joint
limits in active-joint order, drops collision-only branches, and records
discarded active joints in `lock_jointstate` at their configured default
positions.  `reconstruct_joint_state` combines an optimized reduced state with
those locks while preserving all available derivative channels and trajectory
metadata.

The reducer accepts any link in the source portable tree and makes it an output
tool frame in the requested order.  It does not mutate the input configuration.
Collision spheres attached to discarded links are necessarily removed.  Passing
`remove_collision_spheres=False` is accepted only when there are no such
spheres; otherwise it raises `NotImplementedError` instead of returning a
configuration with dangling geometry.

The portable boundary is explicit: a reduced mimic joint whose source joint is
not retained, fixed-only reduced trees, and composition of arbitrary non-zero
revolute locks are not supported.  These require a transform rewrite beyond the
portable tree model; configure/combine those transforms before reduction.
