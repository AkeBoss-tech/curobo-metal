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
| Production graph planning | `019fafb7-4f64-7b70-b109-9831075d94e7` | active | awaiting verified handoff |
| Production mesh/voxel/ESDF collision | `019fafb7-84ad-7921-a398-cdfbbdeee073` | active | awaiting verified handoff |
| Whole-body kinematics/dynamics contracts | `019fafb7-bf20-7600-a526-23b6dcc5d933` | active | awaiting verified handoff |

Parent integration suite after the last completed wave: 139 tests passing with
`PYTORCH_ENABLE_MPS_FALLBACK=0`.
