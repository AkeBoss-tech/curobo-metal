# Portable RobotCfg lifecycle

`curobo._src.types.robot.RobotCfg` owns the same three public fields as pinned
cuRobo V2: `kinematics`, optional `dynamics`, and `device_cfg`.  It accepts
the usual YAML/mapping input and an already compiled portable `KinematicsCfg`.
`load_dynamics: true` produces the typed CPU/MPS `DynamicsCfg` used by the
differentiable whole-body RNEA implementation.

Direct `RobotCfg(portable_tree)` construction remains supported for lower-level
solver factories.  The additive `clone()`/`copy()` helpers make an independent
configuration, including mutable c-space and collision records.  `write_config`
serializes portable tree models; CUDA/Isaac model objects deliberately do not
serialize through this facade.

The compatibility boundary is explicit: raw CUDA packed model buffers, Isaac
USD parsing, Warp model construction, and paired NVIDIA numerical equivalence
are not represented by this CPU/MPS value model.
