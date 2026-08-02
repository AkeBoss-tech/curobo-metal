# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Portable profiling helpers exposed at cuRobo's public import path.

``CudaEventTimer`` keeps the upstream public spelling while selecting the
best timing primitive available on the active backend.  Its return value is
always *seconds*, just as the CUDA implementation does.  On Metal this means
synchronizing the MPS queue at the two measurement boundaries; this is
deliberately an end-to-end timing result rather than an unsynchronized Python
dispatch measurement.
"""

from __future__ import annotations

import time
from typing import Self

import torch

from curobo._src import runtime as _runtime


class CudaEventTimer:
    """A synchronization-correct CUDA/MPS/CPU timer.

    The name and ``start().stop()`` protocol match pinned cuRoboV2.  CUDA
    uses device events when available; MPS and CPU use a monotonic wall-clock
    measurement after device synchronization.  Set
    ``curobo._src.runtime.cuda_event_timers = False`` to make the timer a
    no-op, matching cuRobo's documented runtime flag.
    """

    def __init__(self) -> None:
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None
        self._started_at: float | None = None
        self._backend: str | None = None
        self.elapsed_seconds: float | None = None

    @staticmethod
    def _synchronize_mps() -> None:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    def start(self) -> Self:
        """Begin a measurement and return this timer for method chaining."""
        self.elapsed_seconds = None
        self._start = None
        self._end = None
        self._started_at = None
        self._backend = None
        if not _runtime.cuda_event_timers:
            return self
        if torch.cuda.is_available():
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
            self._backend = "cuda"
        else:
            self._synchronize_mps()
            self._started_at = time.perf_counter()
            self._backend = "mps" if torch.backends.mps.is_available() else "cpu"
        return self

    def stop(self) -> float:
        """End a measurement and return elapsed wall/device time in seconds."""
        if not _runtime.cuda_event_timers or self._backend is None:
            self.elapsed_seconds = 0.0
            return 0.0
        if self._backend == "cuda":
            assert self._start is not None and self._end is not None
            self._end.record()
            self._end.synchronize()
            result = self._start.elapsed_time(self._end) / 1000.0
        else:
            assert self._started_at is not None
            self._synchronize_mps()
            result = time.perf_counter() - self._started_at
        self._backend = None
        self._started_at = None
        self.elapsed_seconds = float(result)
        return self.elapsed_seconds

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


__all__ = ["CudaEventTimer"]
