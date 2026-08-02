# Portable IK and high-level motion behavior

`curobo.inverse_kinematics.InverseKinematics` and the internal
`curobo._src.solver.solver_ik.IKSolver` run a deterministic projected PyTorch
IK solve on CPU or MPS.  A `GoalToolPose` can contain one timestep, every
configured tool frame, and up to `IKSolverCfg.max_goalset` candidates.  Each
seed is evaluated against every goal-set candidate; ranking chooses its lowest
cost candidate and returns the selected value in `result.goalset_index` with
shape `[batch, returned_seeds, links]`.

`run_optimizer=False` is a metrics-only evaluation: it ranks the supplied
seeds (or current joint state) without modifying them and still returns pose
errors, selected goal indices, world clearance, feasibility, and normal
top-k result fields.  `optimization_dt` produces a finite-difference solution
velocity relative to `current_state`.

`update_world` accepts a portable `SceneCfg`, a list of `SceneCfg` for an
existing matching multi-environment collision adapter, or a `SceneCollision`.
It updates the live adapter used for IK scoring.  World clearance is composed
through the public vectorized `SceneCollision` query; active world penetration
makes a solution infeasible and adds an activation-distance penalty while
optimizing.

This is intentionally not a CUDA/Warp drop-in at the kernel level.  Raw CUDA
graphs, captured streams, Warp/BVH collision internals, YAML/USD scene asset
loading, and multi-timestep pose goals are explicit unsupported boundaries.
Use `TrajOptSolver` for a pose trajectory goal.
