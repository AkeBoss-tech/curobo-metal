"""Seeded NumPy random sequence."""

import numpy as np
from .sequencer_base import BaseSequencer


class RandomSequencer(BaseSequencer):
    def __init__(self, ndims: int, seed: int = 123):
        super().__init__(ndims, seed)
        self.reset()

    def random(self, n_samples: int):
        return self._rng.random((n_samples, self.ndims))

    def reset(self):
        self._rng = np.random.default_rng(self.seed)

    def fast_forward(self, steps: int):
        self._rng.random((steps, self.ndims))
