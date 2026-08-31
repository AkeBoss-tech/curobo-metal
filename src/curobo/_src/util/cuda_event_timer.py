from __future__ import annotations

import time

import torch

from curobo._src import runtime


def _synchronize_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


class CudaEventTimer:
    """Synchronization-correct timer retaining the upstream class name."""

    def __init__(self):
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None
        self._started_at: float | None = None
        self._backend: str | None = None
        self.elapsed_seconds: float | None = None

    def start(self) -> CudaEventTimer:
        self.elapsed_seconds = None
        self._start = None
        self._end = None
        self._started_at = None
        self._backend = None
        if not runtime.cuda_event_timers:
            return self
        if torch.cuda.is_available():
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
            self._backend = "cuda"
        else:
            _synchronize_mps()
            self._started_at = time.perf_counter()
            self._backend = "mps" if torch.backends.mps.is_available() else "cpu"
        return self

    def stop(self) -> float:
        if not runtime.cuda_event_timers or self._backend is None:
            self.elapsed_seconds = 0.0
            return 0.0
        if self._backend == "cuda":
            assert self._start is not None and self._end is not None
            self._end.record()
            self._end.synchronize()
            result = self._start.elapsed_time(self._end) / 1000.0
        else:
            assert self._started_at is not None
            _synchronize_mps()
            result = time.perf_counter() - self._started_at
        self._backend = None
        self._started_at = None
        self.elapsed_seconds = float(result)
        return self.elapsed_seconds
