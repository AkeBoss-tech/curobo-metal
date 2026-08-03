# Portable kinematics-loader lifecycle

`curobo._src.robot.loader.KinematicsLoaderCfg` and `KinematicsLoader` compile
URDF plus cuRobo-shaped collision/C-space configuration into the production
CPU/MPS kinematics metadata.

Supported portable behavior includes:

- packaged or absolute URDF resolution, tool/collision/mesh link validation,
  YAML collision-sphere maps, disabled reserved spheres, and per-environment
  sphere banks;
- device-correct tensor C-space input materialized into the serializable robot
  model without retaining caller-owned tensor aliases;
- self-collision pair compilation with configured ignores/buffers, including
  configurations that reserve a future `attached_object` link;
- deterministic cache rebuilding through `initialize_tensors()` and after
  `add_link()` / `add_fixed_link()`; and
- fixed/prismatic lock-joint application.  Locking a nonzero revolute joint is
  rejected because it requires transform composition rather than a scalar
  metadata edit.

The loader intentionally does not expose upstream packed CUDA FK/RNEA buffers,
CUDA graph state, Warp kernels, USD/Isaac assets, or external asset providers.
Those paths raise explicit errors instead of producing a CPU fallback or a
misleading compatibility result.
