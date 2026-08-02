# Portable PRM configuration

`curobo._src.graph_planner.graph_planner_prm_cfg.PRMGraphPlannerCfg` compiles
the useful configuration boundary of cuRobo V2's PRM planner for CPU and MPS.
It accepts a robot mapping/path/object, graph/rollout/transition mappings or
YAML paths, a scene mapping/path/configuration, cache capacities, and the
usual deterministic seeds. Short bundled robot and scene names such as
`franka.yml` and `collision_test.yml` resolve exactly as they do from the
high-level planner factory.

The three V2 factory-default task paths are usable even though this package
does not bundle NVIDIA's task YAML corpus: they compile to the documented
portable class defaults. Any other absent task path raises `FileNotFoundError`
instead of being silently ignored. Supplied task mappings are checked for
factory-owned overrides, the reduced robot c-space produces finite
device-local lower/upper bounds, and all capacities, radii, growth factors,
seeds, and projection methods are validated at construction.

`use_cuda_graph_for_rollout` remains an accepted setting. CPU/MPS planners use
persistent shape-keyed execution state; no CUDA graph is created, and direct
CUDA graph manipulation continues to raise an explicit unsupported error.
The three pinned projection names (`svd`, `householder`, `approximate`) are
accepted for configuration compatibility, but portable sampling uses the
documented PyTorch approximation rather than a Warp/CUDA kernel.
