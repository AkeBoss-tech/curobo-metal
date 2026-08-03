# Robot cost-manager lifecycle

`RobotCostManager` eagerly composes portable c-space, tool-pose,
self-collision, scene-collision, and start/target c-space-distance costs on
CPU or MPS. Batch allocation is cached per `(batch, horizon)` and all state,
goal, index, collision, and timing tensors are checked before cost evaluation;
the manager never copies a CPU tensor into an MPS rollout implicitly.

Goal index buffers may have the pinned `[batch]` or `[batch, 1]` form. A
per-problem `current_state_dt` supplied as `[batch]` is treated as a time
value per rollout problem and is expanded internally to `[batch, 1]` before
the cost adds its final DOF axis. This avoids accidental alignment with the
horizon dimension while leaving the shared `GoalRegistry` unchanged.

`RobotCostManagerCfg.create` accepts serializable nested configuration maps or
already-built matching config values, including an explicit matching
`DeviceCfg`. Mutable weight/activation updates accept scalars and same-device
tensors only. CUDA stream overlap, CUDA Graph capture, Warp cost kernels, and
packed collision-buffer ABI remain intentionally unavailable; normal PyTorch
CPU/MPS autograd is the supported execution path.
