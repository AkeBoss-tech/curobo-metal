# Portable Robot Debugger Lifecycle

`RobotDebugger` loads portable robot YAML through `KinematicsCfg`, compiles the
configured collision spheres plus link-level ignore/padding rules into indexed
self-collision pairs, and evaluates those pairs with production CPU/MPS
kinematics and collision operations.

`check_collision_at_config()` accepts one finite joint configuration and
returns the pinned detailed collision schema: `has_collision`,
`num_colliding_pairs`, unique link `colliding_pairs`, `max_penetration`, and
per-link-pair `distances`.  The legacy `collision` key remains an alias.
`last_result` is a copy, so callers cannot mutate the retained inspection
snapshot.

Sampling uses a CPU-seeded uniform stream over configured position limits and
then transfers each batch to the selected CPU/MPS device.  Link-pair frequency
is counted once per sampled configuration, even when multiple sphere pairs for
that link pair overlap.  `collision_matrix_stats()`, `inspection_report()`,
and `export_inspection_report()` provide deterministic, JSON-safe evidence
without a visualizer.

Direct XRDF conversion, Viser sessions, USD/Isaac inspection, Warp mesh data,
and CUDA graph/kernel diagnostics remain explicit external boundaries.  This
tool inspects the portable collision-sphere model; it does not claim raw CUDA
debugger parity.
