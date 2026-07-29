# Whole-body reference scope

Wave 5D extends the correctness layer from serial end-effector FK to complete
fixed-base trees. The new code is deliberately independent NumPy: it specifies
topology, mimic behavior, full-link transforms/Jacobians, and rigid-body
inverse dynamics before any Torch or Metal production operator is selected.

The pinned cuRobo V2 source provides `link_map`, joint maps/types, BFS tree
levels, link COM/mass arrays, six inertia coefficients, and mimic metadata.
Its public kinematics state can generate mimic joint state, but the census pin
does not provide a supported rigid-body inverse-dynamics API. Consequently,
tree metadata is aligned with V2 while RNEA and its decomposition are a new
portable contract rather than a claim of numerical parity with a cuRobo CUDA
dynamics kernel.

Included are fixed-base parent-index trees, fixed/revolute/prismatic joints,
multiple end effectors, direct active-joint mimic mappings, batched analytical
transform/geometric Jacobians, RNEA, mass matrix, bias and gravity terms,
effort limits, and quadratic effort/violation costs.

Deferred are floating bases, loop constraints, contacts/external wrenches,
friction and actuator/transmission dynamics, rotor inertia, identified
parameters, forward dynamics/integration, collision coupling, and all
Torch/MPS/Metal production implementations. Mimic chains are also deferred;
the pinned loader resolves mimic joints from independently actuated joints and
the contract rejects ambiguous recursive mappings.
