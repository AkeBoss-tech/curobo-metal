# cuRobo Metal: Multi-Agent Execution Plan

## Objective

Prove that a useful, cuRobo-compatible robotics workload can run correctly and
faster than CPU on Apple Silicon, then grow that proof into a maintainable Metal
backend.

The first success criterion is deliberately narrower than "port cuRobo":

> On a pinned cuRoboV2 revision, run batched forward kinematics and sphere
> collision queries on an Apple GPU, match trusted reference outputs within
> documented tolerances, and beat the optimized CPU baseline on representative
> batch sizes.

Full motion-generation parity is a later decision, not an initial promise.

## Core Technical Choice

Use PyTorch tensors and its MPS device at the public boundary. Implement simple
operations with ordinary PyTorch/MPS first, and use custom Metal kernels only
where profiling proves they are needed. Expose all implementations through a
small backend-neutral operator interface.

This preserves compatibility with cuRobo's Python-facing model while avoiding a
line-by-line CUDA translation. MLX can be evaluated later as a standalone
backend, but it is not on the critical path.

```text
cuRobo-compatible Python API
             |
      backend operator API
       /       |        \
CPU reference  MPS ops   custom Metal
       \       |        /
     shared fixtures and contract tests
```

## Multi-Agent System

One orchestrator owns the dependency graph, integration branch, acceptance
gates, and final decisions. Worker agents operate in isolated worktrees or
branches and submit small, reviewable changes. No two agents own the same files
at the same time.

### Agent roles

1. **Orchestrator / integrator**
   - Maintains `work-items.yaml`, assigns bounded tasks, and records upstream
     commit, environment, evidence, and status.
   - Integrates only work that passes its acceptance commands.
   - Resolves API decisions and prevents simultaneous edits to shared modules.

2. **Upstream cartographer**
   - Pins cuRoboV2 and inventories Python, PyTorch, CUDA, Warp, CUDA Graph, and
     Isaac-specific dependencies.
   - Produces a call graph from user-visible workloads to low-level operators.
   - Classifies each operator as already MPS-compatible, expressible with
     composed PyTorch ops, requiring Metal, or out of initial scope.

3. **Reference and correctness engineer**
   - Builds deterministic CPU reference implementations and compact fixtures.
   - Defines forward/backward contracts, shapes, dtypes, edge cases, and
     numerical tolerances.
   - Creates differential tests whose expected values are independent of the
     Metal implementation.

4. **Metal kernel engineer**
   - Implements one operator family at a time behind the backend interface.
   - Supplies forward and gradient support where required.
   - Profiles synchronization, memory layout, occupancy, and kernel fusion.

5. **Benchmark and platform engineer**
   - Builds reproducible CPU/MPS/CUDA benchmark runners and captures machine and
     software metadata.
   - Tests several Apple GPU generations when runners are available.
   - Detects regressions using distributions, not a single timing.

6. **Compatibility and API engineer**
   - Adapts the smallest possible cuRoboV2 surface to backend dispatch.
   - Keeps CUDA behavior unchanged and prevents Metal concerns from leaking
     through the public API.
   - Builds standalone examples that do not require Isaac Sim.

7. **Adversarial reviewer**
   - Does not implement the feature under review.
   - Searches for false speedups, CPU fallbacks, synchronization mistakes,
     nondeterminism, gradient errors, and fixtures derived from the code under
     test.
   - Attempts to falsify milestone claims before they are published.

With four concurrent agent slots, run the orchestrator plus at most three
workers. The reviewer is scheduled after implementation rather than kept active
continuously.

## Agent Work Contract

Each work item must contain:

```yaml
id: KIN-001
owner: metal-kernels
inputs:
  upstream_commit: "<sha>"
  contract: "contracts/forward_kinematics.md"
allowed_paths:
  - src/curobo_metal/ops/kinematics/
acceptance:
  - "pytest tests/ops/test_forward_kinematics.py"
  - "python benchmarks/kinematics.py --device mps"
artifacts:
  - correctness JSON
  - benchmark JSON
  - profiler trace
blocked_by: [REF-001]
```

Agent prompts should provide the exact pinned revision, allowed paths, required
commands, expected artifacts, and a stop condition. Agents may diagnose outside
their path but may not edit it. Every result includes commands run, failures,
assumptions, and the commit SHA.

## Execution Phases and Gates

### Phase 0: Pin, license, and feasibility (1–2 days)

- Pin a specific cuRoboV2 commit and record its license and asset constraints.
- Establish supported macOS, Xcode, Python, and PyTorch versions.
- Run a small MPS custom-operation spike, including autograd and packaging.
- Decide whether CUDA comparison occurs on CI, a remote runner, or recorded
  golden outputs.

**Gate 0:** A custom Metal operation can be built, loaded, tested, and packaged
on a clean Apple Silicon machine. If not, use composed PyTorch/MPS operations
for the first proof.

### Phase 1: Upstream census and contracts (3–5 days)

- Inventory kernels and trace the minimal forward-kinematics and collision paths.
- Identify CUDA-only imports and initialization that prevent importing on macOS.
- Define the operator interface and device-dispatch rules.
- Build small robot fixtures (2-link planar, Franka-class arm, one higher-DoF
  model) and deterministic scene fixtures.
- Specify value and gradient tolerances by dtype and operation.

**Gate 1:** A generated inventory accounts for every dependency on the selected
workload paths, and CPU contract tests pass without CUDA installed.

### Phase 2: Batched forward kinematics (1–2 weeks)

- Implement a clear CPU reference.
- Implement an MPS baseline with standard PyTorch operators.
- Profile it, then add fused Metal kernels only where justified.
- Validate poses, link transforms, Jacobians, gradients, non-contiguous inputs,
  batch sizes, and edge configurations.

**Gate 2:** Correctness passes; no silent CPU fallback occurs; MPS beats CPU at
predeclared representative batch sizes. Publish results even if the performance
gate fails.

### Phase 3: Sphere collision primitives (2–3 weeks)

- Add robot sphere transforms and sphere-to-sphere self-collision.
- Add sphere-to-primitive world distance and gradients.
- Defer mesh, voxel, ESDF, and swept-volume support until primitive contracts
  and performance are solid.
- Test contact boundaries, inactive spheres, padding, large scenes, and
  deterministic reductions.

**Gate 3:** Collision values and gradients pass differential and finite-
difference tests, with benchmark evidence for useful scene and batch sizes.

### Phase 4: IK proof (2–4 weeks)

- Compose kinematics, pose cost, joint-limit cost, collision cost, and a portable
  optimizer.
- Avoid porting specialized CUDA LBFGS initially; establish an optimizer baseline
  using backend-neutral PyTorch operations.
- Report success rate, residual, collision freedom, latency distribution, and
  warm-up/compile costs separately.

**Gate 4:** A standalone Mac example solves a fixed benchmark suite reliably,
and failures are reproducible and categorized.

### Phase 5: Integration decision

Based on evidence, choose one:

- **Upstream backend:** propose a small backend abstraction and incremental PRs.
- **Companion package:** maintain `curobo-metal` against pinned upstream
  versions.
- **Portable primitives library:** publish the validated kinematics/collision
  layer without claiming full cuRobo compatibility.
- **Stop:** publish the census, tests, and negative performance findings if the
  backend is not viable.

Only after this decision consider trajectory optimization, graph planning,
mesh/voxel collision, ESDF, or whole-body dynamics.

## Repository Shape

```text
PLAN.md
pyproject.toml
src/curobo_metal/
  backend.py
  ops/
    kinematics/
    collision/
    costs/
  metal/
tests/
  contracts/
  differential/
  fixtures/
benchmarks/
  results/
contracts/
docs/
  compatibility.md
  numerical_tolerances.md
  profiling.md
tools/
  inventory/
work-items.yaml
```

Keep the upstream source as a pinned submodule or test dependency rather than
copying it into this repository.

## Validation Strategy

- Compare Metal to an independent float64 CPU reference where possible.
- Compare against CUDA on identical serialized inputs, not separately generated
  random cases.
- Test forward values, backward gradients, and finite differences.
- Record seeds, synchronization points, warm-up, compile time, steady-state
  latency, memory use, and machine metadata.
- Benchmark batch-size curves and publish cases where CPU is faster.
- Run tests with CUDA unavailable to expose accidental imports and fallbacks.
- Require the reviewer to reproduce benchmark claims from a clean checkout.

## Coordination Cadence

The orchestrator runs three short loops:

1. **Dispatch:** select only unblocked work and assign disjoint paths.
2. **Evidence review:** ingest machine-readable test and benchmark artifacts;
   return failures to the same agent with a focused follow-up.
3. **Integration:** ask the adversarial reviewer to falsify the result, then
   merge only if the phase gate holds.

Maintain a decision log for architecture changes and a compatibility matrix for
upstream cuRobo, PyTorch, macOS, and Apple GPU generations.

## First Parallel Sprint

Run these three worker tasks concurrently:

- **A — Census:** pin cuRoboV2 and produce the operator/dependency inventory.
- **B — Toolchain spike:** package one forward/backward custom Metal operation.
- **C — Test foundation:** create robot fixtures, CPU FK reference, and
  differential-test format.

The orchestrator then reconciles their outputs into the backend contract. Do not
start production Metal kinematics before those three outputs agree.

## Definition of a Credible Public Result

A release is credible when a new user can install it on a supported Mac, run a
standalone example, reproduce correctness tests and benchmark results, inspect
known incompatibilities, and distinguish compile/warm-up latency from
steady-state performance. A partial, rigorously measured result is preferable to
an unverified claim of full cuRobo support.
