# RobotSceneCollisionCfg portable lifecycle

`curobo._src.collision.collision_robot_scene_cfg.RobotSceneCollisionCfg` now
owns a portable robot/scene collision setup rather than a CUDA launch-buffer
bundle. `load_from_config` still accepts the pinned robot and scene inputs, and
the config also supports three explicit lifecycle operations:

- `clone()` makes independent kinematics, costs, deterministic sampler state,
  and built-in `SceneCollision` world state.
- `to(DeviceCfg(...))` recreates those value objects on CPU or Apple Metal.
  It does not reuse CUDA/Warp buffers. A same-device sampler copy continues
  the current deterministic stream; a CPU↔MPS move restarts its target-device
  index stream from the public seed because PyTorch's generator-state encoding
  is backend-specific.
- `update_scene_model(...)` reloads a `SceneCfg`, mapping, or per-environment
  scene list. With the same environment count it preserves the existing
  checker and cost identities; otherwise it atomically installs a new
  portable checker. `set_scene_collision_checker` accepts an already-built
  portable checker, and `update_scene_model(None)` detaches the world.

`update_collision_parameters(distance)` changes both scene-cost activation
distances and defaults `contact_distance` to half that distance, matching the
pinned configuration formula.

Caller-supplied query objects are external resources. They remain shared in a
same-device clone, but moving one across devices is rejected rather than
silently accessing CPU or CUDA state. Raw Warp collision objects, CUDA graph
buffers, and CUDA/Warp ABI launch paths remain unsupported; use
`SceneCollision` for CPU/MPS tensor queries.
