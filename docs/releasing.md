# Release procedure

This procedure publishes `curobo-metal` artifacts built by GitHub Actions. Do
not upload a locally built wheel or sdist.

## One-time repository setup

1. Confirm ownership/availability of the `curobo-metal` project on TestPyPI and
   PyPI. A registry 404 is not a reservation; ownership is established only by
   the index account/project configuration.
2. On each index, configure a Trusted Publisher for this GitHub repository and
   `.github/workflows/release.yml`. Use the `testpypi` environment for TestPyPI
   and the `pypi` environment for production.
3. Create protected GitHub environments named `testpypi` and `pypi`. Require a
   human reviewer for `pypi`; do not store a long-lived API token.

## Alpha dry run

1. Start the `publish` workflow manually. It builds one wheel and one sdist,
   runs `twine check`, retains those exact artifacts, and publishes them to
   TestPyPI through OIDC.
   The build also runs `tools/packaging/check_distribution_contents.py` to
   require all notices and reject generated/source-tree leakage.
2. Install the TestPyPI artifact in an empty virtual environment while taking
   dependencies from PyPI:

   ```sh
   python3 -m venv /tmp/curobo-metal-testpypi
   /tmp/curobo-metal-testpypi/bin/pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     curobo-metal==0.1.0a1
   /tmp/curobo-metal-testpypi/bin/pip check
   ```

3. Run `tools/packaging/wheel_smoke.py` from outside the checkout. Confirm that
   namespace ownership is exactly `curobo-metal` and that `curobo.__version__`,
   `curobo_metal.__version__`, and distribution metadata agree.

## Production publish

1. Require a clean commit with all branch-protection checks passing. Confirm
   `CHANGELOG.md` has no unresolved release blockers hidden by the advertised
   scope.
2. Create an annotated tag matching the source version exactly, prefixed with
   `v` (for example `v0.1.0a1`).
3. Create and publish a GitHub release from that tag. The workflow refuses a
   tag/version mismatch, rebuilds the artifacts once from the tag, validates
   them, then pauses at the protected `pypi` environment for approval.
4. After publication, install from PyPI in a new environment and repeat the
   smoke test. Attach the workflow artifact and relevant parity report hashes
   to the GitHub release notes.

Published files are immutable. If any post-publish check fails, fix the issue,
increment the PEP 440 version, and publish a new release; never replace an
existing artifact.
