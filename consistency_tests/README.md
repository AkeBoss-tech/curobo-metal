# Full cuRobo consistency gates

This directory contains intentionally red, forward-looking tests for the full
pinned cuRoboV2 compatibility contract.  It is outside pytest's configured
`testpaths`, so the bounded alpha suite under `tests/` remains green.

Run the full consistency gates explicitly:

```sh
PYTHONPATH=.:src PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run pytest -q consistency_tests
```

The suite checks every pinned runtime module for static export and callable
shape parity, every upstream test/example census entry for completed review,
and every portable capability for a full semantic-equivalence classification.
Failures are the compatibility backlog; they must not be changed to skips or
xfails merely to make this suite green.

The authoritative portable gate is case-level, not module-level. Generate it
from a clean environment where the candidate wheel is installed, on an Apple
Silicon machine with MPS available:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/gauntlet/run_portable_upstream.py \
  --upstream /path/to/pinned/curobo \
  --wheel /path/to/curobo_metal.whl \
  --junit artifacts/gauntlet/portable-upstream-junit.xml \
  --output artifacts/gauntlet/portable-case-census.json
```

This command covers all 116 substituted upstream modules, rejects collection
errors and unreviewed skips, records each pinned source hash, verifies that
`curobo` came from `site-packages` and that every hashed installed file matches
the exact candidate wheel's `RECORD`, requires MPS, and disables PyTorch's MPS
CPU fallback. A failing assertion remains useful execution evidence while the
implementation is in progress; it still blocks final parity.

The raw failure count is not a quality score. Most runtime modules are private
`curobo._src` implementation surfaces, and census classification is weaker
than execution. Acceptance follows `gauntlet/curobo-consistency.json`: clean
installed-wheel upstream tests and paired numerical/runtime evidence outrank
static or documentary closure.

The checks are intentionally broader than the `0.1.0a1` claims.  Raw CUDA/Warp
ABIs and unavailable external integrations remain explicit exclusions rather
than false parity requirements.
