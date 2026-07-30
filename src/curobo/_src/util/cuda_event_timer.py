from __future__ import annotations

import time
import torch


class CudaEventTimer:
    """Synchronization-correct timer retaining the upstream class name."""

    def __init__(self):
        self.start_event = None
        self.end_event = None
        self._started = None

    def start(self):
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        self._started = time.perf_counter()
        return self

    def stop(self):
        if self._started is None:
            raise RuntimeError("timer has not been started")
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - self._started) * 1000.0
        self._started = None
        return elapsed
