# Coordination ledger

This ledger records durable child-thread handoffs and parent integration state.

| Scope | Child task | Status | Integrated commit / state |
|---|---|---|---|
| Upstream V2 census | `019fadf0-7655-7930-88d2-d1776c3efc4d` | complete | baseline `fbfdae3` |
| Metal toolchain spike | `019fadf0-a7a6-7ae3-88cd-9577a92c03fa` | complete | baseline `fbfdae3` |
| FK reference/contracts | `019fadf0-d56a-7232-bfb1-edd00b0de3ab` | complete | baseline `fbfdae3` |
| Production composed FK | `019fae4c-f42c-7d21-b1a8-bf7c1bedae36` | complete | `a482f10` |
| FK adversarial profile | `019fae52-791f-7102-8573-bb64157b9e7b` | complete | `766490f` |
| Fused Metal FK | `019fae58-035d-7b51-84d5-4de4d42ebca2` | complete | `245c839` |
| Collision contracts | `019fae4d-1c3e-7c02-abc5-36a2fe532de2` | complete | `27f32ad` |
| Production collision | `019fae54-9348-74b0-bf97-9f9c3c06de61` | complete | `5d1eea8` |
| Fused Metal collision | `019fae5a-647d-70b2-9e98-cb4d40dbdaee` | complete | `22d14bb` |
| cuRoboV2 adapter/audit | `019fae4d-703c-7de2-b35d-909c42fedaff` | complete | `bbc9413` + parent lazy-import fix `85d140b` |
| IK contracts/oracle | `019fae54-d65d-7963-9946-c64d889dd693` | complete | `6dde7ed` |
| Production IK | `019fae5b-b9a8-7821-93d5-398fbbea3bb4` | complete | `2eae74f` |
| Trajectory contracts/oracle | `019fae5d-393b-7bd1-8ca8-13c709664748` | complete | `678d412` + parent locked goldens `5d96b2d` |
| Production trajectory / motion generation | `019fae67-0559-7221-8ef6-37ce743c5832` | complete | `55ab1ab` |
| Graph-planning contracts/oracle | `019fae63-66e2-7c52-a40c-3593832946cb` | complete | `ad20ebf` |
| World-collision contracts/oracle | `019fae63-b19f-7ca1-b972-463314f9ac1b` | complete | `1d2da7a` |
| Production graph planning | `019fafb7-4f64-7b70-b109-9831075d94e7` | complete | `55eeebf` |
| Production mesh/voxel/ESDF collision | `019fafb7-84ad-7921-a398-cdfbbdeee073` | complete | `0cebc77` |
| Whole-body kinematics/dynamics contracts | `019fafb7-bf20-7600-a526-23b6dcc5d933` | complete | `8d46776` |
| Production whole-body dynamics | coordinated worktree | complete | `b4a1bc9` |
| Portable MotionGen facade | coordinated worktree | complete | `32f9548` |
| API parity/release audit | coordinated worktree | complete | `c703cce` |
| Config and cuRobo value types | coordinated worktree | complete | `6fcf2c2` |
| Jacobian and collision-checker APIs | coordinated worktree | complete | `94091f7` |
| MotionGen API compatibility | coordinated worktree | complete | `2c3c083` + parent fixture fix `cddfe21` |
| Depth-fused perception/ESDF | coordinated worktree | complete | `9738ec2` |
| Dynamics-aware B-spline trajectory | coordinated worktree | complete | `5f576e9` |
| Particle and L-BFGS optimizers | coordinated worktree | complete | `9e401b0` + parent seed fix `f8b69ea` |
| Final parity re-audit | `019fb01c-e076-7be3-9a49-a0677f096b9a` | complete | `aa1e349` |
| Closure: config/types/Jacobian/collision | `019fb024-24ff-7b80-9964-a1dad8658c37` | complete | `0323c18` |
| Closure: optimizers/solvers/planning | `019fb024-5e32-7db3-b2fb-110888d42d64` | complete | `464aab8` |
| Closure: whole-body/perception | `019fb024-9f04-74f0-bf84-99a0220a3e48` | complete | `06ea4b7` |
| Paired CUDA/Metal replay harness | `019fb06a-4696-7eb0-aa8e-eb3a9f4ff48d` | complete | `c38f392` + parent whitespace fix |

Parent integration suite after the upstream-example replay wave: 285 tests passing with
`PYTORCH_ENABLE_MPS_FALLBACK=0`.

The portable implementation and deterministic MPS replay corpus are complete.
On 2026-08-26 a fresh self-verifying handoff executed successfully on the iLab
NVIDIA RTX 3090 host against the exact pinned upstream revision. All 19
registered adapters passed the strict paired comparison with executed invalid,
edge, and expanded multi-case matrix evidence; the promotable aggregate report
is checked in under
`artifacts/parity/cuda-replay/paired-report-2026-08-26-matrix.json`.
The replay runner now has real, asset-independent pinned-upstream CUDA adapters
for `DeviceCfg`, `Pose`, `JointState`, and shared `BaseSolverResult`
construction/clone behavior. A fifth adapter uses a license-clean serialized
two-joint URDF to replay `RobotCfg`, cspace, and joint-limit loading without
external assets. Two compiled adapters reuse that robot for forward-kinematics
position/quaternion/gradient evidence and geometric-Jacobian value evidence;
an eighth asset-independent adapter exercises the upstream CUDA self-collision
kernel on an explicit two-sphere pair, including value, input-gradient, and
invalid-pair evidence. A ninth adapter reuses the serialized robot's inertial
model for native upstream CUDA RNEA torque and first-order VJP evidence,
including missing-acceleration rejection. A tenth adapter exercises the
position-tracking subspace of upstream `ToolPoseCost`, including first-order
position gradients and mismatched-tool rejection. High-level Franka IK and
trajectory-optimization outcome/layout adapters are also included. A compiled
cubic B-spline adapter compares position through jerk against the portable
boundary-constrained implementation. The PRM adapter now passes pinned CUDA
outcome replay against the declarative forbidden-box corpus, and the
EvolutionStrategies adapter passes multi-seed natural-gradient mean-update
outcomes without requiring identical RNG streams. The L-BFGS adapter passes
eager batched quadratic outcome checks while explicitly excluding unsupported
terminal freezing and hard projection. The final MotionGen stand-in was
replaced by a real `MotionPlanner.plan_cspace` adapter that passes packaged-
Franka trajectory layout, endpoint, path, status, and invalid-attempt outcome
checks. The ready-suite
command binds CUDA evidence to the exact committed
input and Metal output hashes, requires executed invalid-case evidence for every
ready adapter, records CUDA/GPU runtime provenance, and writes an aggregate
report. A deterministic self-verifying archive supports transfer to an NVIDIA
host that cannot clone this repository directly.

A resumed runner audit established a working password-authenticated IPv4 route
to iLab1. The saved two-hop `robo` helper remains unnecessary for the current
evidence run. All standalone examples and the isolated macOS wheel build/import
smoke pass on the current integration commit.

The pinned upstream Franka forward-kinematics tutorial workload now runs on
MPS from the original `franka.yml` and URDF. Loader fixes cover dependency-free
indentless YAML sequences, cuRobo `content/assets` resolution, explicit URDF
sub-roots, and single-tool path extraction from branched trees. The recorded
1,000-configuration replay matches the independent float64 oracle within
`3.273e-07` for transforms and `5.035e-08` for the loss gradient.
