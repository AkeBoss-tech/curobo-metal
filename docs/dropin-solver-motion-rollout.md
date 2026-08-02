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

The following remain explicit boundaries: CUDA Graph reset/capture, raw Warp
rollout kernels, automatic YAML/USD/Isaac scene-asset loading, and collision
cost integration into the portable IK/trajectory solve objective. A retained
scene supports lifecycle and attachment APIs; it is not represented as a hidden
CUDA collision rollout.
