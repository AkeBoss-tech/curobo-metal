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

Parent integration suite after the last completed wave: 268 tests passing with
`PYTORCH_ENABLE_MPS_FALLBACK=0`.
