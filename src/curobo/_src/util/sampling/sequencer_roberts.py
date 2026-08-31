"""Roberts additive low-discrepancy sequence."""

import numpy as np
from .sequencer_base import BaseSequencer


def _roberts_root(dim: int) -> float:
    x = 1.5
    for _ in range(100):
        function = x ** (dim + 1) - x - 1.0
        derivative = (dim + 1) * x**dim - 1.0
        next_x = x - function / derivative
        if abs(next_x - x) < 1.0e-12:
            x = next_x
            break
        x = next_x
    return x


class RobertsSequencer(BaseSequencer):
    def __init__(self, ndims: int, seed: int = 123):
        super().__init__(ndims, seed)
        self.root = _roberts_root(ndims)
        self.basis = 1.0 - 1.0 / self.root ** (1 + np.arange(ndims))
        self.current_index = 0

    def random(self, n_samples: int) -> np.ndarray:
        indices = np.arange(self.current_index, self.current_index + n_samples)
        samples = (indices[:, None] * self.basis[None, :]) % 1.0
        self.current_index += n_samples
        return samples.astype(np.float32)

    def reset(self) -> None:
        self.current_index = 0

    def fast_forward(self, steps: int) -> None:
        if steps > 0:
            self.current_index += steps
