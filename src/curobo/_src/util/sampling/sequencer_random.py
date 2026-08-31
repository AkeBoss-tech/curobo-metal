"""Seeded NumPy random sequence."""

import numpy as np
from .sequencer_base import BaseSequencer


class RandomSequencer(BaseSequencer):
    def __init__(self, ndims: int, seed: int = 123):
        super().__init__(ndims, seed)
        self.rng = np.random.RandomState(seed)
        self._initial_state = self.rng.get_state()

    def random(self, n_samples: int) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=(n_samples, self.ndims))

    def reset(self) -> None:
        self.rng.set_state(self._initial_state)

    def fast_forward(self, steps: int) -> None:
        if steps > 0:
            self.rng.uniform(0.0, 1.0, size=(steps, self.ndims))
