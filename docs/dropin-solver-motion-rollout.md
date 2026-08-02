# Solver, motion, and rollout lifecycle compatibility

The portable solver stack accepts the pinned cuRoboV2 lifecycle entry points for
IK, trajectory optimization, MPC, planner warmup, result cloning/ranking, goal
buffers, and motion-planner scene attachment management. CPU and MPS execution
uses the production PyTorch FK, IK, trajectory, and scene components.

`TrajOptSolver.solve_pose` now composes portable IK with c-space trajectory
optimization. A supplied `goal_state` skips IK, matching the upstream endpoint
override. `MotionPlanner.update_world` accepts a concrete `SceneCfg`, a list of
`SceneCfg`, or `SceneCollisionCfg`; `attachment_manager` is backed by the same
portable kinematics and scene object.

`MotionPlanner` also compiles the optional PRM configuration into the portable
PRM implementation, uses its valid sampled paths as retry seeds, preserves
seed-ranked trajectory payloads while producing interpolation, and resets the
roadmap alongside world/cache/seed lifecycle calls. Link-sphere collision
enable/disable and tool-pose criteria now mutate the same production kinematics
and solver configuration that planning consumes. Grasp planning performs real
approach, grasp, and lift pose stages when requested; it does not claim the
CUDA rollout's exact linear-motion-cost or contact semantics.

The following remain explicit boundaries: CUDA Graph reset/capture, raw Warp
rollout kernels, automatic YAML/USD/Isaac scene-asset loading, and collision
cost integration into the portable IK/trajectory solve objective. A retained
scene supports lifecycle and attachment APIs; it is not represented as a hidden
CUDA collision rollout.

## SolverCore lifecycle

`SolverCore` now retains the V2 component contract independently of a concrete
IK, TrajOpt, or MPC facade. `prepare_goal_buffer` returns `(GoalRegistry,
structural_change)`: value-only goal updates preserve the existing shape,
whereas a batch, seed, environment, goal-set, solve-mode, or tool-frame change
invalidates shape-keyed portable state. Criteria updates, joint-target toggles,
seed resets, supplied rollout propagation, deterministic bounded sampling, and
link inertial mutation all execute on CPU or MPS tensors. The core factory
resolves bundled robot YAML and mapping inputs into portable optimizer,
rollout, and scene records.

`use_cuda_graph=True` remains accepted at the configuration boundary for
source-compatible application defaults, but is recorded then disabled; an
explicit `reset_cuda_graph()` or `debug_dump()` raises `NotImplementedError`.
This is intentional: eager Metal execution is not represented as CUDA capture,
and raw CUDA streams, Warp packed buffers, and NVIDIA numerical parity remain
outside this portable contract.
