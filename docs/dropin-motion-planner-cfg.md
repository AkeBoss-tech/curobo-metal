# Motion planner configuration compatibility

`curobo._src.motion.motion_planner_cfg.MotionPlannerCfg` compiles the pinned
V2 high-level planner request into the production CPU/MPS IK, trajectory, PRM,
and scene-collision configurations. Normal applications use
`MotionPlannerCfg.create(...)` and can keep the upstream robot, task, world,
batch, tolerance, seed, and interpolation settings.

The direct dataclass form is also supported for applications that compose
solver configurations themselves. It validates the constraints the planner
needs before a solve:

- IK and TrajOpt use the same device, batch capacity, multi-environment flag,
  and goal-set capacity.
- A supplied scene has the expected environment count and is the same scene
  passed to each child solver and PRM planner.
- Typed `SceneCollisionCfg` inputs are reused; their device and cache are not
  silently replaced by factory arguments.

`clone()`/`copy()` make an independent configuration tree and preserve the
shared-scene relationships within that new tree. `update()` validates an
entire replacement before mutating the existing object. They intentionally
do not move an already compiled robot/world from CPU to MPS (or the reverse):
use `create(..., device_cfg=DeviceCfg(...))` for the target device so every
underlying tensor is rebuilt coherently.

`use_cuda_graph=True` remains accepted and is recorded on the child core
records as `requested_use_cuda_graph`; it uses portable persistent execution
state, not an emulated CUDA Graph. Raw CUDA graph manipulation remains
unsupported on Apple Metal.
