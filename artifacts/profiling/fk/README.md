# FK profiling artifacts

`baseline-{cpu,mps}.json` contain synchronized correctness and latency curves.
`profile-{cpu,mps}-b{1,1024,8192}.json` contain synchronized cost decomposition,
CPU dispatcher aggregates, and output sizes. Matching `trace-*.json` files are
Chrome traces from one synchronized inference forward.

The exact reproduction commands and interpretation are in
`docs/profiling/fk-mps.md`. MPS evidence is valid only with
`PYTORCH_ENABLE_MPS_FALLBACK=0`.
