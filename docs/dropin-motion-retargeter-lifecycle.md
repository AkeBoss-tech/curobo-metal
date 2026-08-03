# Portable motion-retargeter lifecycle

`MotionRetargeter` keeps V2's global-IK-first, local-IK-or-MPC-followup model
on CPU and Apple Metal.  It materializes ordinary PyTorch state instead of a
CUDA graph, so its reusable state has explicit, inspectable lifecycle rules.

- `update_world(SceneCfg | SceneCollisionCfg | SceneCollision)` validates the
  world through global IK and installs the same collision adapter in global IK,
  local IK, and MPC.  The operation increments `world_generation` and clears
  warm-start state: a configuration found feasible before an obstacle change
  must never seed the first frame after it.  Construct the retargeter with
  an initial concrete `SceneCfg` (an empty `SceneCfg()` is sufficient) if the
  world may change later; that retains `load_collision_spheres=True`.  A solver
  that intentionally skipped robot spheres cannot acquire that fixed robot
  geometry after setup.
- `update_tool_pose_criteria(...)` accepts replacement weights for the same
  ordered tool-frame list.  It recompiles criteria for the configured CPU/MPS
  device and clears warm-start state.  Adding, removing, or reordering links
  is a fixed-shape solver change and requires a new retargeter.
- `last_result` and `last_failure_mask` make the latest materialized output and
  per-environment status observable.  A failed local IK row is held at its
  prior solution while other batch rows advance; the failure mask remains the
  signal to retry, skip, or stop the source clip.  This avoids silently
  propagating a failed optimizer candidate as a new warm start.
- `destroy()` is idempotent and releases portable solver caches.  `solve_*`,
  `reset`, and mutation methods reject use after destruction.

The retained boundaries are deliberate: raw CUDA graph capture, Warp kernels,
Isaac retargeting assets, and CUDA-specific solver-buffer ABI are not emulated.
World asset YAML/USD parsing is not performed by runtime updates; pass an
already compiled portable scene record instead.
