# Pinned cuRobo forward-kinematics example on Apple MPS

The pinned upstream tutorial
`curobo/examples/getting_started/forward_kinematics.py` loads `franka.yml`,
prints one end-effector pose, evaluates 1,000 configurations in a batch, and
back-propagates a position loss. Its source hard-codes `device="cuda"` and CUDA
timing events, so the unmodified file cannot execute on an Apple-only machine.

`examples/upstream_forward_kinematics_mps.py` preserves that workload while
changing only the backend-facing imports, device, and synchronization clock. It
loads the original pinned `franka.yml` and URDF directly; no translated robot
fixture is used. It also compares all batched transforms and the loss gradient
against the independent NumPy float64 FK implementation.

Run it with:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python \
  examples/upstream_forward_kinematics_mps.py \
  /path/to/pinned/curobo/content/configs/robot/franka.yml \
  --device mps --batch-size 1000 --seed 0 \
  --json-output artifacts/correctness/upstream_fk_example.json
```

On the development Apple M4, the replay produced the same tutorial-level
observables:

- 7 robot degrees of freedom;
- tool frame `panda_hand`;
- batched end-effector position shape `[1000, 1, 3]`;
- finite first-order joint gradients.

Numerically, Metal differed from the independent float64 reference by at most
`3.273e-07` over every batched transform and `5.035e-08` over the loss
gradient. The recorded synchronized 1,000-configuration time was 3.59 ms; this
single example timing is evidence of execution, not a replacement for the
checked-in benchmark distributions.

This does not assert that the original CUDA executable produced byte-identical
values on this Mac: CUDA is unavailable here. Paired CUDA equivalence remains
governed by the strict replay harness in `tools/parity/`.

The literal unmodified pinned file was also attempted with the pinned checkout
first on `PYTHONPATH`. It stops during upstream import because the portable
environment intentionally does not install upstream's `trimesh` dependency;
after that dependency boundary, the source still explicitly allocates
`device="cuda"` tensors and CUDA timing events. Thus “unchanged upstream
script” is not a supported macOS claim—the verified result is the same tutorial
workload through the Metal compatibility implementation.
