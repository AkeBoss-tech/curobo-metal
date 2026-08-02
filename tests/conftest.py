"""Pytest capability probes shared by CPU and Apple Metal test runs."""

from __future__ import annotations

import torch


def _mps_can_allocate() -> bool:
    """Return whether this host can execute, not merely advertise, MPS.

    Some hosted macOS runners expose ``torch.backends.mps.is_available()`` but
    reject the first tiny shared-pool allocation.  Treating that state as an
    available accelerator produces dozens of cascading false failures.  A
    real Apple device still runs all MPS parametrizations normally.
    """

    if not torch.backends.mps.is_available():
        return False
    try:
        torch.empty(1, device="mps")
        torch.mps.synchronize()
    except RuntimeError:
        return False
    return True


if torch.backends.mps.is_available() and not _mps_can_allocate():
    # Test modules use this official availability predicate to decide whether
    # to parameterize MPS cases.  Override it only for this unusable runner.
    torch.backends.mps.is_available = lambda: False
