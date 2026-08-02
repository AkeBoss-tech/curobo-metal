# cuRoboV2 API inventory and compatibility gate

Phase 1 records the statically recoverable Python API of cuRoboV2 commit
`8e734f3ced1df898990bcd92de40abce475907db`. The checked-in inventory is
[`artifacts/api_compat/upstream-api.json`](../artifacts/api_compat/upstream-api.json).
It is a baseline for compatibility work; it is not a claim that every upstream
API is already implemented.

## What is inventoried

The generator walks every `curobo/**/*.py` file in lexical path order and parses
it with Python's `ast` module. It records:

- module names, repository-relative paths, and source SHA-256 values;
- public module assignments, imports/re-exports, functions, and classes;
- function and method parameter kinds, annotations, defaults, return annotations,
  async status, and decorators when their source form is statically recoverable;
- class bases, decorators, constructor signatures, public members, enum members,
  and dataclass status/options/fields;
- local module and top-level symbol resolution under `src/curobo` first and
  `src/curobo_metal` second.

“Public” means a bound name that does not begin with `_`. Dynamic exports,
conditional runtime mutations, generated methods, and values that require
evaluation are intentionally not inferred. Expressions are retained as
normalized source text rather than executed.

The local classifications are:

- `resolved`: the local module exists and binds every inventoried top-level name;
- `partial`: a local module exists but one or more names are absent;
- `missing_module`: neither local package path exists.

The inventory's module classification compares top-level symbol presence. It
captures class-member metadata but does not treat member differences as module
resolution failures.

`tools/api_compat/surface_gate.py` is the review companion. It consumes the
checked-in inventory and local source with AST only, reporting per-module
top-level export coverage and declared function/constructor/method shape matches.
It is intentionally an evidence report, not a drop-in verdict: dynamic exports,
runtime-generated methods, behavior, numerical results, autograd, device
residency, performance, and CUDA/Warp ABI compatibility remain outside its scope.

```shell
python tools/api_compat/surface_gate.py \
  --inventory artifacts/api_compat/upstream-api.json \
  --local-root src \
  --output /tmp/curobo-surface-report.json
```

`--require-exact-exports` makes missing static exports fail the command. It does
not promote callable-shape matches into behavioral or backend equivalence.

## Generate and check

Use a Git checkout at the exact pinned revision:

```shell
python tools/api_compat/generate.py \
  --source /path/to/pinned/curobo \
  --local-root src \
  --output artifacts/api_compat/upstream-api.json
```

CI or a local review can verify that the artifact matches both upstream and the
current local tree without rewriting it:

```shell
python tools/api_compat/generate.py \
  --source /path/to/pinned/curobo \
  --local-root src \
  --output artifacts/api_compat/upstream-api.json \
  --check
```

`--check` fails when the file is missing or stale. `--require-compatible` adds a
strict drop-in gate and fails while any module is partial or missing. It is
separate because Phase 1 intentionally establishes the gap baseline.

The generator resolves `HEAD^{commit}` and refuses every revision other than the
pin. There is no override. Neither upstream nor local modules are imported, so
inventory generation does not initialize CUDA, Torch, Warp, or Isaac
dependencies.

## Installed-wheel runtime import gate

`tools/api_compat/runtime_imports.py` imports each runtime module recorded in the
committed inventory. It needs no upstream checkout; the only input is the pinned
JSON artifact. The regular test validates source-tree imports, while CI builds a
wheel, installs it, changes to `/tmp`, and runs:

```shell
python tools/api_compat/runtime_imports.py \
  --inventory /path/to/upstream-api.json \
  --require-installed-wheel
```

The installed-wheel mode verifies that `curobo` resolves from the installed
`curobo-metal` distribution before it imports all recorded runtime namespaces.
This gate proves importability only. Optional CUDA, Warp, USD, Isaac, Blox, and
Viser integrations must stay lazy at import time; their unsupported operations
continue to raise explicit errors when invoked.

## Determinism and review

Paths, modules, symbols, members, and JSON object keys are sorted; JSON uses a
fixed indentation and trailing newline. The artifact contains no timestamps,
machine paths, Python version, or host metadata. Re-running the check with
identical upstream and local source must therefore produce byte-identical output.

Focused tests live in `tests/api_compat_inventory/` and cover wrong-revision
rejection, signature/default recovery, dataclass and enum metadata, deterministic
ordering, local classification, the no-import property, AST surface reporting,
and exhaustive runtime namespace importability.
