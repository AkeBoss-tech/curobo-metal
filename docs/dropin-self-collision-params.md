# Self-collision parameter compatibility

`curobo._src.robot.types.self_collision_params.SelfCollisionKinematicsCfg`
compiles robot link spheres into the pair list consumed by the portable CPU and
Metal self-collision cost.

- The V2 distance matrix contract is retained: `-inf` disables a pair, while
  every other upper-triangular value, including `+inf`, enables it.
- Same-link pairs and a link-pair ignore in either direction are disabled.
  Missing link padding is zero. Inputs remain on the configured CPU or MPS
  device; the compiler never constructs a hidden CPU collision buffer.
- Configuration validates padding shape/device, canonical unique pair indices,
  link mapping coverage, radii, and padding values before a planner starts.
- `num_checks_per_thread`, `max_threads_per_block`, and
  `num_blocks_per_batch` preserve V2's pair-count launch heuristics for callers
  that size diagnostic buffers. They are metadata only on Metal.

The pinned CUDA implementation writes `int16` launch buffers and consumes them
through Warp. The portable compiler uses `int64` PyTorch indices instead; raw
Warp/CUDA pointer layout and kernel launch ABI are intentionally unsupported.
