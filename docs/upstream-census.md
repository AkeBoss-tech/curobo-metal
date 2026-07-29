# cuRoboV2 upstream census: minimal kinematics and sphere collision

## Pin and scope

This census pins the `main` branch of
[NVlabs/curobo](https://github.com/NVlabs/curobo) as resolved on 2026-07-29:

```text
8e734f3ced1df898990bcd92de40abce475907db
2026-07-22T22:40:10-07:00
Fix mesh SDF gradient sign for query points outside the surface (#701)
```

The immutable source root is
[`https://github.com/NVlabs/curobo/tree/8e734f3ced1df898990bcd92de40abce475907db`](https://github.com/NVlabs/curobo/tree/8e734f3ced1df898990bcd92de40abce475907db).
The latest release tag visible at census time was `v0.8.0`
(`4ea77366ca48ee453e7df139e39fa6532af49f3b`), but this sprint deliberately
pins the newer current V2 source rather than a moving branch or older tag.
Code is Apache-2.0; bundled assets have separate terms in upstream
[`LICENSE_ASSETS`](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/LICENSE_ASSETS).

In scope:

- batched `Kinematics.compute_kinematics`, including link poses and transformed
  robot spheres, forward and first backward;
- discrete sphere self-collision;
- discrete robot-sphere to cuboid-world distance and first backward.

Deferred: Jacobian output, center of mass, swept collision, mesh, voxel/ESDF,
dynamics, IK/trajectory optimization, CUDA Graph acceleration, Isaac integration,
and sphere fitting.

## Reproduction

```bash
git clone --filter=blob:none --no-checkout https://github.com/NVlabs/curobo.git /tmp/curobo-v2
git -C /tmp/curobo-v2 checkout --detach 8e734f3ced1df898990bcd92de40abce475907db
git -C /tmp/curobo-v2 show -s --format='%H%n%aI%n%s' HEAD
python3 tools/inventory/scan_upstream.py \
  --source /tmp/curobo-v2 \
  --output artifacts/inventory/workload-dependencies.json
python3 -m json.tool artifacts/inventory/workload-dependencies.json >/dev/null
```

The scanner rejects any revision other than the pin by default, hashes each
traced source file, extracts Python imports with `ast`, and records line-level
CUDA, Warp, CUDA Graph, Isaac, and PyTorch evidence. It does not execute cuRobo
or require CUDA.

## Call graphs

### Forward kinematics and robot spheres

```text
curobo.kinematics.Kinematics
  -> Kinematics.compute_kinematics(JointState)
  -> Kinematics._forward(q[B,H,D])
  -> KinematicsFusedFunction.apply
  -> create_buffers: torch.zeros / zeros_like
  -> forward validation: dtype/device/shape checks
  -> backend proxy "kinematics"
  -> launch_kinematics_forward_spheres
  -> CUDA FK kernel:
       joint transform construction
       serial/tree transform composition
       selected link pose extraction
       link-local sphere transform
  -> KinematicsState(ToolPose, robot_spheres)

Backward:
autograd -> KinematicsFusedFunction.backward
  -> launch_kinematics_backward
  -> CUDA fused pose/quaternion/sphere-to-joint gradient kernel
```

Evidence:
[public/internal API](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/robot/kinematics/kinematics.py#L101-L199),
[autograd dispatch](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/curobolib/cuda_ops/kinematics.py#L37-L349),
and
[CUDA launch implementation](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/curobolib/backends/pybind/kinematics_forward_kernel_launch.cu).
V2 offers only `cuda_core` and `pybind` kernel backends; both are CUDA. Backend
loading is lazy enough for some imports on a non-CUDA host, but the first FK
launch cannot succeed without one of those backends.

### Sphere self-collision

```text
RobotCollisionChecker.get_scene_self_collision_distance_from_joints(q)
  -> FK graph above -> robot_spheres[B,H,N,4]
  -> get_self_collision_distance
  -> SelfCollisionCost.forward
  -> SelfCollisionDistance.apply
  -> backend proxy "geometry".self_collision_distance
  -> CUDA self-collision kernel:
       configured sphere-pair distances
       activation/padding
       max penetration reduction
       sparse sphere gradients
  -> distance[B,H,1]
```

Evidence:
[public composition](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/collision/collision_robot_scene.py#L173-L264),
[cost wrapper](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/cost/cost_self_collision.py#L91-L127),
and
[autograd/backend call](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/curobolib/cuda_ops/geometry.py#L12-L107).
The pair list, padding, block maxima, sparse gradient mask, and optional
per-pair output are part of the upstream contract even though the first sprint
only needs the reduced distance and input gradient.

### Discrete robot-sphere to cuboid world

```text
RobotCollisionChecker.get_collision_distance(KinematicsState)
  -> SceneCollisionCost._discrete_fn
  -> SceneCollision.get_sphere_distance
  -> CollisionChecker.get_sphere_distance
  -> SphereObstacleCollision.apply
  -> wp.from_torch for spheres, scene data, weights, outputs
  -> wp.launch(sphere_obstacle_collision_kernel)
  -> cuboid signed-distance/gradient
  -> atomic accumulation into distance and gradient buffers
  -> autograd returns precomputed sphere gradient
```

Evidence:
[collision cost](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/cost/cost_scene_collision.py#L58-L183),
[checker](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/geom/collision/checker_collision.py#L77-L123),
[Warp autograd bridge](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/geom/collision/wp_autograd.py#L37-L126),
and
[atomic accumulation](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/curobo/_src/geom/collision/wp_collision_common.py#L64-L96).

## Dependency classification

“MPS-ready” means the existing operation can remain ordinary PyTorch on an MPS
tensor. “PyTorch/MPS” means the CUDA/Warp operation should be re-expressed from
portable tensor primitives first. “Custom Metal” marks the likely optimized
implementation, not a requirement for the first correctness baseline.

| Dependency/operator | Upstream mechanism | Classification | Sprint treatment |
|---|---|---|---|
| Public `JointState`, `KinematicsState`, `ToolPose`, config/YAML validation | Python/dataclasses | MPS-ready | Reuse or mirror the narrow data contract |
| Buffer creation, reshape/unsqueeze, zeros, copies | PyTorch | MPS-ready | Keep device-local; audit int8/int16/bool MPS support |
| Joint-local revolute/prismatic transform construction | CUDA FK | PyTorch/MPS | Express with tensor trig, quaternion/matrix ops |
| Tree/chain transform propagation | fused CUDA FK | PyTorch/MPS; likely Custom Metal after profiling | Fixed topology permits a link-order loop with batched tensor math |
| Selected link pose extraction | CUDA FK | PyTorch/MPS | Index/gather after global transforms |
| Link-local sphere to world transform | fused CUDA FK | PyTorch/MPS; fusion candidate | Batched gather plus rotate/translate; preserve radius and env map |
| FK backward to joint positions | fused CUDA backward | PyTorch/MPS | Let autograd differentiate baseline; custom backward only if justified |
| Sphere-pair center distance, radii/padding, activation | CUDA self-collision | PyTorch/MPS | Gather configured pairs, norm, mask |
| Maximum penetration and winning-pair gradient | CUDA block reduction | PyTorch/MPS initially; likely Custom Metal for scale/determinism | Define tie behavior and reduction determinism before optimizing |
| Cuboid inverse-pose transform and box SDF | Warp | PyTorch/MPS | Closed-form tensor implementation is small and differentiable |
| Sphere/cuboid obstacle accumulation | Warp 2-D kernel + float atomics | PyTorch/MPS initially; Custom Metal if memory expansion is costly | Prefer deterministic reduction over atomics for reference |
| cuRobo CUDA backend selector and runtime compilation | `cuda-core` or PyBind CUDA | requires Custom Metal/backend dispatch | Add a third backend seam; neither existing backend can serve MPS |
| Warp initialization and Torch/Warp bridge | `warp-lang`, `wp.init`, `wp.from_torch` | deferred/bypass | No Warp dependency on the Metal minimal path |
| CUDA Graph capture | `torch.cuda.CUDAGraph` in solver/rollout layers | deferred | Not reached by these direct APIs; benchmark eager MPS first |
| Isaac/Omniverse/USD | examples/integration, not traced workload | deferred | No Isaac import is required by the three runtime paths |
| Mesh BVH/SDF, voxel/ESDF, swept collision | Warp/native specialized paths | deferred | Explicitly outside first collision milestone |
| `trimesh`, `yourdfpy`, YAML, NumPy/SciPy | host-side model/config parsing | MPS-ready host dependency | Allowed during setup; exclude from timed GPU query |

## Python/package and initialization findings

The upstream package requires Python `>=3.10`; its base dependencies include
PyTorch-adjacent utilities plus `warp-lang>=0.10.0`, but PyTorch itself is only
declared by the `cu12-torch`/`cu13-torch` extras. CUDA runtime and `cuda-core`
are optional CUDA extras. The exact dependency list is in
[upstream `pyproject.toml`](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/pyproject.toml#L9-L75).

`RobotSceneCollisionCfg.load_from_config` calls `init_warp` before it knows
whether a scene checker is needed. Thus the convenience constructor makes Warp
an eager runtime dependency even for self-collision-only use. Direct assembly
of narrower config objects can avoid that call, but a compatible Metal-facing
constructor should explicitly gate Warp by backend/collision type.

The direct FK API does not use CUDA Graph. CUDA Graph appears around iterative
solver and rollout execution and is therefore deferred, not an FK prerequisite.
Likewise, no `isaac` or `omni` import occurs in the traced runtime files.
Robot YAML/URDF parsing and optional USD/Isaac examples are setup/integration
concerns rather than low-level workload operators.

## Unresolved risks and contract questions

1. Upstream `main` moves. Every fixture and comparison must carry this SHA;
   updating it is a reviewed work item, not an automatic pull.
2. Upstream fused FK has topology-specialized launch choices, float32 checks,
   16-byte alignment assumptions, environment-indexed sphere configurations,
   quaternion conventions, and reusable output buffers. A simple tensor port
   must test each rather than assume mathematical equivalence.
3. The exact behavior at zero-length quaternion normalization, zero sphere
   separation, collision boundaries, and equal maximum penetrations needs
   golden CUDA outputs. These affect gradients and deterministic tie-breaking.
4. MPS support/performance for the upstream compact metadata dtypes
   (`int8`, `int16`, `int32`, `bool`, `uint8`) must be tested on the supported
   PyTorch/macOS matrix. Widening indices may be necessary.
5. Warp's primitive path accumulates floats atomically across obstacles.
   A composed implementation may differ in reduction order. Tolerances and a
   deterministic reduction policy must be defined before comparing gradients.
6. The convenience collision constructor also builds a Halton sample buffer and
   c-space cost, neither needed for a direct distance query. The compatibility
   layer should not accidentally include them in timing or make them mandatory.
7. V2's current world path is generic over cuboid, mesh, and voxel scene data.
   Supporting only cuboids requires an explicit error for active deferred types,
   never a silent omission.
8. No CUDA runner was available in this census, so low-level traces are
   source-derived rather than validated with runtime hooks. Golden values and
   gradients remain a separate reference/correctness task.

## Generated artifacts

- `artifacts/inventory/upstream-pin.json`: human-readable immutable pin and
  release context.
- `artifacts/inventory/workload-dependencies.json`: generated hashes, imports,
  source evidence, and scope.
- `tools/inventory/scan_upstream.py`: deterministic generator and pin check.

