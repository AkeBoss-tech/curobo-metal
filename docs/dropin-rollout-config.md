# Portable `RobotRolloutCfg`

`curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg` retains the pinned V2
configuration boundary on CPU and Metal: it validates component types, compiles
YAML-shaped transition and cost-manager sections through the supplied `create`
methods, leaves caller mappings unmodified, and returns cost/constraint managers
in V2 order.

`object` is retained as an explicit portable assembly sentinel for generic solver
configuration. In that mode precompiled mappings are copied but deliberately not
treated as CUDA/Warp transition or manager implementations. Scene construction,
CUDA graph capture, streams, and packed CUDA buffers remain owned by their
respective layers and are not emulated by this config record.
