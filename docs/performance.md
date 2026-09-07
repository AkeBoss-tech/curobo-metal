# Apple Silicon performance gate

Stable releases run synchronized warm-latency workloads on the project's free,
self-hosted Apple M4 runner with `PYTORCH_ENABLE_MPS_FALLBACK=0`. The checked
artifact is `artifacts/performance/m4/gate.json`; CI regenerates the raw reports
and rejects a release regression above the declared ceilings.

| Workload | M4 median | Release ceiling |
| --- | ---: | ---: |
| FK, batch 1024 | 1.39 ms | 12 ms |
| Sphere/sphere collision, batch 512 | 0.99 ms | 25 ms |
| Sphere/cuboid collision, batch 512 | 0.27 ms | 40 ms |
| Reachable IK fixture | 977 ms | 1500 ms |
| Trajectory optimization fixture | 228 ms | 600 ms |
| PRM detour fixture | 91 ms | 250 ms |
| Dense perception update | 4.18 ms | 25 ms |
| ESDF query, 256 points | 3.14 ms | 20 ms |

These ceilings are regression limits for fixed repository fixtures, not general
latency guarantees. Workload size, seed count, robot, obstacles, map resolution,
and convergence tolerances can materially change runtime. Cold compilation and
initialization are recorded in the raw reports but excluded from warm medians.

The PRM edge validator batches ragged edge samples into one device operation.
This removed one implicit MPS synchronization per candidate edge and reduced the
fixed 2,763-edge benchmark from about 1.03 seconds to 91 milliseconds on the
same M4 while retaining its path and planning metrics.
