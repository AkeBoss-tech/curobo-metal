# Portable getting-started examples

The V2 source pinned at `8e734f3ced1df898990bcd92de40abce475907db`
contains eight CUDA-oriented tutorials in `examples/getting_started/`.  The
portable repository has an executable, asset-independent equivalent for the
five planning workflows that can run on CPU and Apple Metal:

```bash
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python examples/getting_started_portable.py --device mps
```

Use `--device cpu` on systems without MPS. `--json` produces a machine-readable
record of output shapes, solver success, device residency, and mapper state.
`tests/integration/test_portable_getting_started_examples.py` runs the same
flows in a clean subprocess on CPU and, when usable, MPS.

| Pinned tutorial | Portable evidence | Intentional boundary |
| --- | --- | --- |
| `forward_kinematics.py` | Batched Franka FK, robot spheres, and reverse-mode joint gradients. | The tutorial's `cuda` tensors and CUDA-event timing must be replaced with `cpu`/`mps` tensors and ordinary synchronized timing. |
| `inverse_kinematics.py` | A reachable pose is solved through `InverseKinematicsCfg`/`InverseKinematics`. | Viser interaction and CUDA graph capture are unavailable; changing worlds uses the portable `Scene`/collision interfaces. |
| `motion_planning.py` | A real pose-plan route runs IK plus trajectory optimization and returns an interpolated trajectory. | The smoke uses an asset-free, no-graph route; GPU CUDA graphs and exact Warp/CUDA trajectories are not claimed. |
| `build_robot_model.py` | A URDF with native sphere collision geometry is fitted exactly, collision ignores are generated, and YAML is reloaded. | The alpha includes the attributed Franka mesh subset, but the pinned self-test still selects CUDA and its full MorphIt/Warp mesh sphere-fitting behavior has no unchanged Metal execution evidence. |
| `volumetric_mapping.py` | Synthetic RGB-D depth fusion, static cuboid stamping, ESDF construction, and mesh extraction run on CPU/MPS. | The full Sun3D tutorial needs a downloaded dataset. Dense portable maps do not reproduce CUDA/Warp block hashes, PBA scheduling, or large-map capacity. Viser and GLB/texture export remain optional/external surfaces. |
| `feature_mapping.py` | Not exercised by the smoke. | It requires the external Sun3D dataset, C-RADIO checkpoint/download, feature-volume integration, and optional Viser. Portable feature-volume fusion deliberately raises a precise unsupported error. |
| `reactive_control.py` | The underlying portable MPC route is covered elsewhere by `tests/dropin/mpc_runtime/`. | The tutorial's viewer and plotting are optional; exact CUDA-graph real-time behavior is not equivalent. |
| `humanoid_retargeting.py` | Not exercised by the smoke. | It needs external humanoid assets and the Viser/Isaac-oriented visualization and retargeting setup. |

These checks prove portable execution and API routing only. They do **not**
establish numerical CUDA equivalence: that still requires paired replay on the
pinned upstream commit using an NVIDIA runner.

The authoritative per-module release classification is
`artifacts/api_compat/upstream-execution-census.json`. All 14 bundled example
modules have been reviewed. None is currently labeled `applicable_unchanged`:
six have an executed or tested portable workload but require a documented
device/runtime substitution, seven require unavailable datasets, hardware, or
interactive services, and the package marker itself is not an executable
example.
