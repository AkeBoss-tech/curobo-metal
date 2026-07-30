# Robot construction, parsing, and dynamics compatibility

The `curobo.robot_builder`, `curobo.robot_parser`, and internal robot
loader/type/dynamics import paths match pinned cuRoboV2. URDF parsing preserves
fixed, revolute, prismatic, mimic, limit, inertial, and parent-tree data.
Packaged YAML/URDF configurations compile to the same portable kinematics and
differentiable RNEA backend used elsewhere in curobo-metal.

The portable dynamics facade accepts batched and horizon-shaped `JointState`
records on CPU and fallback-disabled MPS. Standard cuRobo URDF inertias that
are non-physical are projected onto the positive-semidefinite cone before
RNEA compilation.

Mesh sphere fitting, sampled self-collision pruning, Viser visualization,
standalone XRDF authoring/conversion, USD/Isaac assets, and nonzero external
spatial forces raise explicit `NotImplementedError` boundaries.
