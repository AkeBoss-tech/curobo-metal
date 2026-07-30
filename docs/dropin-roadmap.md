# Drop-in cuRoboV2 compatibility program

## Contract

The distribution name is `curobo-metal`; the installed Python namespace is
`curobo`. Users switch the dependency and keep source code unchanged.

Compatibility is pinned to:

```text
8e734f3ced1df898990bcd92de40abce475907db
```

A public surface is compatible only when imports, signatures, defaults, value
types, tensor behavior, mutation, serialization, errors, and numerical results
have executable evidence. A local implementation without paired upstream
evidence is `implemented_unverified`, not compatible.

## Release gates

1. Generate a deterministic module and symbol manifest from the pinned source.
2. Resolve every applicable upstream `curobo.*` import on a clean macOS host.
3. Match signatures, defaults, enums, dataclass fields, and re-exports.
4. Load upstream robot and world configuration files without conversion.
5. Run applicable upstream tests without editing their imports or calls.
6. Run identical serialized cases on pinned CUDA and fallback-disabled MPS.
7. Run applicable upstream examples without source changes.
8. Publish every platform substitution or unavailable external integration.

## Compatibility states

- `compatible`: API and behavioral evidence, including paired replay where
  numerical.
- `implemented_unverified`: production implementation exists but independent
  upstream evidence is incomplete.
- `missing`: public behavior is not implemented.
- `platform_substituted`: observable behavior is implemented with a non-CUDA
  mechanism.
- `external_unavailable`: behavior requires an ecosystem unavailable on the
  target platform, such as Isaac Sim.

No stub may report success or accept an option that it does not execute.

## Work order

1. API census and import/signature gates.
2. `curobo` namespace plus foundational tensor/value types.
3. Content paths, packaged assets, and configuration loaders.
4. Kinematics, dynamics, and collision public wrappers.
5. Costs, rollouts, and optimizer wrappers.
6. IK, graph, and trajectory solver wrappers.
7. Complete `curobo.wrap.reacher.motion_gen` lifecycle.
8. Perception and mapper wrappers.
9. Optional external integration boundaries.
10. Upstream test/example replay and Metal performance closure.

## Non-negotiable device behavior

Requested MPS execution never silently falls back to CPU. MPS kernels target
float32 where required; unsupported dtype/device combinations fail at the same
public boundary documented by the compatibility manifest. CUDA Graphs and
Warp/CUDA kernels may be replaced by portable cache and Metal execution
mechanisms, but CUDA-specific objects are never impersonated when their
observable contract cannot be preserved.
