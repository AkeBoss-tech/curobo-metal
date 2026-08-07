# NVIDIA replay runners

This project has an external evidence gate: compare the portable Metal replay
corpus against pinned cuRobo V2 CUDA outputs. The accepted upstream revision is
`8e734f3ced1df898990bcd92de40abce475907db`.

## Approved Rutgers routes

The following local helper commands are approved for this repository. They
must use the operator's existing SSH authentication; this repository contains
no credentials, passwords, private keys, or internal-host secrets.

```sh
# Stable iLab host, useful for persistent sessions.
~/.local/bin/ssh-ilab

# Saved two-hop route: Mac -> ilab1.cs.rutgers.edu -> omen@172.16.90.195.
~/.local/bin/ssh-robo
```

For a manually selected iLab server, use `ssh ad2046@ilab.cs.rutgers.edu`.
Use `ilab1` when a stable host is needed. Research machines may require Slurm
or additional authorization; do not assume access merely from their hostname.

## Preflight on the CUDA host

```sh
nvidia-smi
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), (torch.__version__, torch.version.cuda)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
git -C /path/to/curobo rev-parse HEAD
```

The final command must print exactly the pinned revision above. Do not run the
comparison against an unpinned upstream checkout.

## Build and transfer a self-verifying handoff

On the Mac, from the repository root:

```sh
PYTHONPATH=.:src .venv/bin/python -m tools.parity.build_cuda_handoff \
  --output /tmp/curobo-metal-cuda-handoff.zip
scp /tmp/curobo-metal-cuda-handoff.zip omen@172.16.90.195:/tmp/
```

If the saved hop cannot forward `scp`, first copy to `ilab1` and then copy to
Omen from that session. This is intentionally an operator action because the
right upstream checkout path and available scratch space belong to the CUDA
host.

## Run the paired comparison

On the CUDA host:

```sh
mkdir -p /tmp/curobo-metal-cuda && cd /tmp/curobo-metal-cuda
unzip /tmp/curobo-metal-cuda-handoff.zip
chmod +x run-cuda.sh
./run-cuda.sh /path/to/pinned/curobo
```

The runner verifies the archive hashes, CUDA availability, upstream revision,
input/output schemas, and numerical tolerances. Its required result is
`cuda-replay/paired-report.json` with `"passed": true`. Copy that directory
back to this repository before changing any parity classification.

## Current local boundary

Metal-side replay is already generated and validates all 19 cases with
`PYTORCH_ENABLE_MPS_FALLBACK=0`. A failed non-interactive `ssh-robo` attempt
that presents password prompts is an authentication/session issue, not a CUDA
test result. Use an authenticated interactive session or provision a runner
with key-based automation before executing the commands above.
