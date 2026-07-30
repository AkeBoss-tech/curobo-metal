"""Roberts additive low-discrepancy sequence."""

import numpy as np
from .sequencer_base import BaseSequencer


class RobertsSequencer(BaseSequencer):
    def __init__(self, ndims: int, seed: int = 123):
        super().__init__(ndims, seed)
        self.reset()

    def random(self, n_samples: int):
        indices = np.arange(self._index + 1, self._index + n_samples + 1)[:, None]
        self._index += n_samples
        return (self._offset + indices * self._alpha) % 1.0

    def reset(self):
        self._index = 0
        rng = np.random.default_rng(self.seed)
        self._offset = rng.random((1, self.ndims))
        phi = (1 + np.sqrt(5)) / 2
        self._alpha = np.mod(phi ** -(np.arange(self.ndims) + 1), 1.0)[None]

    def fast_forward(self, steps: int):
        self._index += steps
