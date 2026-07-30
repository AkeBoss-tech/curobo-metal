#!/usr/bin/env python3
"""Build a deterministic, self-verifying archive for CUDA replay handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .cuda_adapters import ADAPTERS
from .replay_registry import PIN


ROOT = Path(__file__).resolve().parents[2]


def _payload_files() -> list[Path]:
    files = list((ROOT / "tools/parity").glob("*.py"))
    for capability in ADAPTERS:
        files.extend((ROOT / "artifacts/parity/replay" / capability).glob("*"))
    return sorted(path for path in files if path.is_file())


VERIFY_SCRIPT = """#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "handoff-manifest.json").read_text())
for name, expected in manifest["files"].items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing handoff file: {name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"handoff hash mismatch: {name}")
print(f"verified {len(manifest['files'])} handoff files")
"""


RUN_SCRIPT = """#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then
  echo "usage: ./run-cuda.sh /path/to/pinned/curobo" >&2
  exit 2
fi
python verify_handoff.py
PYTHONPATH=. python -m tools.parity.cuda_preflight --upstream "$1"
PYTHONPATH=. python -m tools.parity.run_cuda_ready \\
  --upstream "$1" \\
  --metal-root artifacts/parity/replay \\
  --output cuda-replay
"""


README = f"""# cuRobo Metal pinned CUDA evidence handoff

This archive contains the exact committed inputs and fallback-disabled Metal
outputs for every currently implemented CUDA adapter, plus the strict runner
and comparator. It requires Python, NumPy, CUDA-enabled PyTorch, `nvidia-smi`,
and a usable checkout of cuRobo at:

`{PIN}`

After extracting:

```sh
chmod +x run-cuda.sh
./run-cuda.sh /path/to/curobo
```

The command verifies package hashes, checks CUDA and pinned-upstream provenance,
runs every ready adapter, and writes `cuda-replay/paired-report.json`. A nonzero
exit means the evidence is incomplete or failed comparison.
"""


def build(output: Path) -> None:
    payload: dict[str, bytes] = {}
    for path in _payload_files():
        payload[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    payload["verify_handoff.py"] = VERIFY_SCRIPT.encode()
    payload["run-cuda.sh"] = RUN_SCRIPT.encode()
    payload["README.md"] = README.encode()
    manifest = {
        "format": "curobo-metal-cuda-handoff",
        "version": 1,
        "upstream_revision": PIN,
        "capabilities": sorted(ADAPTERS),
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(payload.items())
        },
    }
    payload["handoff-manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith((".sh", ".py")) else 0o644) << 16
            archive.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
