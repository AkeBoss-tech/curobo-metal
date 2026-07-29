# Fused Metal forward kinematics

## Result

The Wave 3C prototype passes the correctness and performance gates on the
tested Apple M4. The public synchronized steady-state median was **1.607 ms**
at batch 1,024 and **3.844 ms** at batch 8,192, below the respective 2.75 ms
and 16.5 ms targets. `PYTORCH_ENABLE_MPS_FALLBACK=0` was enforced.

The forward path is one runtime-compiled Metal dispatch per nonempty call. One
thread owns one configuration and performs the serial-chain recurrence, exact
transform derivatives, and geometric-Jacobian projection. This replaces the
composed implementation's large fixed operator graph. The composed CPU path
remains unchanged and is also the fallback for MPS chains above the prototype's
64-DOF thread-local storage bound.

## Correctness and differentiation

The shader preserves float32 shapes, row-major tensor storage with the
contract's column-vector transform convention, fixed/revolute/prismatic
joints, unbatched promotion, empty batches, and non-contiguous inputs.
Validation still happens in the public wrapper before dispatch.

`transforms` has a custom backward Metal kernel. It contracts an upstream
transform gradient with the exact transform Jacobian emitted by forward, so
the VJP does not replay FK. Independent JSON golden comparisons, the existing
public VJP suite, and a fused-specific central finite difference pass at the
contract tolerances. The two explicitly returned Jacobian tensors are marked
non-differentiable; higher derivatives of those diagnostic outputs are outside
the version-1 contract.

## Measurement

Reproduce from the repository root:

```sh
uv sync --extra test
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run pytest -q
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python \
  benchmarks/kinematics/fused_fk.py \
  --batch-sizes 1024 8192 --warmup 10 --iterations 50 \
  --output artifacts/profiling/fused-fk/benchmark-mps.json
```

Every sample synchronizes MPS immediately before and after the measured call.
Compilation/first use, ten warmups, and 50 steady-state iterations are
separate. Metadata upload happens in `KinematicChain` construction outside the
timed region.

| Batch | First public call | Validation median | Fused core median | Public median |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 41.137 ms | 0.314 ms | 2.416 ms | **1.607 ms** |
| 8,192 | 25.511 ms | 0.237 ms | 3.586 ms | **3.844 ms** |

The batch-1,024 core and public samples were collected in separate loops and
show allocator/thermal variance, so they must not be subtracted. The public
median is the performance gate. The 41.137 ms first batch includes runtime
shader compilation; the later first-call value is allocation/first-shape cost,
not another compile.

Raw samples and complete environment identifiers are in
`artifacts/profiling/fused-fk/benchmark-mps.json`.

## Prototype boundaries

- The fused path supports MPS float32 chains up to 64 movable joints.
- Non-contiguous input uses an MPS-resident contiguous staging operation.
- Transform VJP is first-order eager autograd. Higher-order gradients,
  forward-mode AD, `vmap`, export, and `torch.compile` are not claimed.
- Public finite-value validation retains its device-to-host synchronization.
