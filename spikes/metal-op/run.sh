#!/bin/sh
set -eu

cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=0

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 --python-preference managed .venv
fi
uv sync --extra test
.venv/bin/pytest
.venv/bin/python scripts/collect_evidence.py
