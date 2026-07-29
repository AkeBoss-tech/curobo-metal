# Collision profiling evidence

These JSON files were collected on the Apple Silicon environment described by
their embedded platform and PyTorch metadata. `reference-cpu.json` and
`fused-mps.json` are end-to-end public operator measurements.
`attribution-{cpu,mps}.json` separates composed clearance/reduction arithmetic
from the full result, including explicit gradient materialization and
validation. All reported steady-state samples are device-synchronized and
exclude first use plus warmup.
