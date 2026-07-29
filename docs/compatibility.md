# cuRoboV2 compatibility boundary

## Supported pin and public API targets

The compatibility seam targets only NVlabs/cuRobo commit
`8e734f3ced1df898990bcd92de40abce475907db`. Updating the pin requires a
reviewed change to the adapter, tests, audit artifact, and this document.

The upstream public entry points in scope are:

- `curobo.kinematics.KinematicsCfg.from_robot_yaml_file`,
  `from_config_file`, `from_data_dict`, and `from_basic_urdf`, which generate a
  `KinematicsCfg`;
- `curobo.types.robot.RobotCfg.create`, whose `kinematics` member is a
  `KinematicsCfg`;
- `curobo.kinematics.Kinematics`, specifically the configuration consumed by
  its constructor and later by `compute_kinematics`;
- the public tensor fields on
  `curobo._src.robot.types.kinematics_params.KinematicsParams`: fixed
  transforms, link/parent maps, joint maps/types/offsets, tool-frame map,
  collision spheres and their link map, names, base link, and degree count.

`curobo_metal.compat.convert_kinematics_config` accepts a generated
`KinematicsCfg`, its `KinematicsParams`, or a mapping serialized with those
field names. It copies the data to C-contiguous NumPy arrays. NumPy is the
backend-neutral interchange representation; an FK or collision backend owns
the later one-time transfer to CPU, MPS, or another device.

The adapter deliberately does not call the YAML/URDF constructors. Applications
may generate the tensor config in an existing cuRobo environment and serialize
the narrow fields, or pass an already generated object. This prevents import of
cuRobo's CUDA backend and avoids the eager Warp initialization in
`RobotSceneCollisionCfg.load_from_config`.

## Output contract

`BackendRobotConfig` contains:

| Field | Shape | Meaning |
|---|---:|---|
| `fixed_transforms` | `[L,4,4]` | Homogeneous parent-to-link transforms |
| `parent_link` | `[L]` | Parent link index |
| `joint_index` | `[L]` | Active joint index, or `-1` for fixed |
| `joint_type` | `[L]` | Pinned cuRobo `JointType` integer (`-1..11`) |
| `joint_offset` | `[L,2]` | Joint multiplier and additive offset |
| `tool_link` | `[T]` | Link indices selected as tool frames |
| `link_spheres` | `[E,S,4]` or absent | Link-local xyz and radius per environment |
| `sphere_link` | `[S]` or absent | Owning link of each sphere |

Names, base link, degree count, and the upstream revision travel with the
arrays. Float data is normalized to float64 and indices to explicit integer
dtypes for deterministic portable handoff. This is configuration conversion,
not a production FK or collision implementation.

## Reproducible upstream workflow

The manager never places upstream source in this repository:

```bash
python3 tools/upstream/manage.py fetch --destination /tmp/curobo-v2
python3 tools/upstream/manage.py verify --source /tmp/curobo-v2
python3 tools/upstream/manage.py audit \
  --source /tmp/curobo-v2 \
  --output artifacts/compat/macos-import-audit.json
```

`fetch` requests the immutable commit directly and checks it out detached.
`verify` resolves `HEAD^{commit}`, reads the raw Git commit object, and
recomputes its Git SHA-1 (`SHA1("commit " + length + NUL + bytes)`). Thus a
matching branch name, tag, or checkout label is insufficient.

For an isolated upstream import experiment, install the verified source without
dependencies into a disposable target:

```bash
python3 tools/upstream/manage.py install \
  --source /tmp/curobo-v2 \
  --target /tmp/curobo-v2-install
```

Upstream's normal runtime dependencies must be supplied separately. They are
not dependencies of `curobo-metal` and the compatibility conversion does not
need that install.

## Import and device policy

Importing `curobo_metal.compat` may import NumPy and the Python standard
library. It must not import PyTorch, CUDA bindings, Warp, Isaac Sim, or
Omniverse. Tensor-like inputs are handled structurally through
`detach().cpu().numpy()`; NumPy mappings need none of those methods. The audit
checks both the static import graph and a fresh runtime import, while the tests
exercise mapping and tensor-like conversion with CUDA unavailable.

No device fallback occurs in this layer: it owns no execution device and
launches no operations. A future backend must make transfer and dispatch
explicit.

## Explicitly unsupported upstream features

The adapter rejects branching kinematic trees, malformed joint maps, unknown
joint types, and inconsistent sphere/tool maps. The following features remain
unsupported and are never silently discarded:

- URDF, USD, XRDF, or YAML parsing inside `curobo-metal`;
- mimic joints whose generated joint map is not a simple serial active-joint
  sequence, and arbitrary-axis joints outside pinned cuRobo `JointType`;
- multiple branches, multiple roots, closed chains, and whole-body models;
- `Kinematics.compute_kinematics` execution, Jacobians, center of mass,
  dynamics, gradients, and any production FK kernel;
- `RobotSceneCollisionCfg`, `RobotCollisionCheckerCfg`, and
  `RobotSceneCollision` construction;
- self-collision pair generation/reduction and all collision query execution;
- cuboid, mesh/BVH, voxel, ESDF, swept-volume, continuous collision, and sphere
  fitting;
- CUDA/PyBind and `cuda-core` backend launches, CUDA Graph capture, Warp
  initialization/launch, Isaac Sim, Omniverse, and USD integration;
- IK, trajectory optimization, graph planning, motion generation, and solver
  APIs;
- attached-object mutation after conversion, per-environment sphere mutation,
  mesh export, visualization, and asset resolution;
- joint limits, c-space weights, lock-state reconstruction, inertial data,
  grasp-contact metadata, and self-collision padding/ignore metadata.

Those omissions define the end of Wave 2C. Later work must add an explicit
backend contract and tests before consuming more upstream state.
