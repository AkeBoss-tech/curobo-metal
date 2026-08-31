# Vendored CUDA baseline tests

`src/curobo/tests` is an unmodified snapshot of `curobo/tests` from NVIDIA
cuRobo commit `8e734f3ced1df898990bcd92de40abce475907db` (Apache-2.0). It lets
the Metal port be measured against the exact upstream tests, rather than a
locally rewritten approximation.

These tests are expected to be red on Apple Silicon while CUDA/Warp behavior
is still being ported. Run the complete diagnostic suite explicitly:

```sh
PYTHONPATH=.:src PYTORCH_ENABLE_MPS_FALLBACK=0 uv run pytest -q src/curobo/tests
```

Do not edit individual vendored files. Refresh the snapshot only after updating
the compatibility pin:

```sh
uv run python tools/upstream/manage.py fetch --destination /tmp/curobo
uv run python tools/upstream/sync_tests.py --source /tmp/curobo
```

`src/curobo/tests/MANIFEST.json` holds a SHA-256 for every source file. The
snapshot gate in `tests/api_compat_inventory/test_upstream_test_snapshot.py`
requires all 177 upstream `test_*.py` modules (and their fixtures/helpers) to
remain present and byte-for-byte unchanged.
