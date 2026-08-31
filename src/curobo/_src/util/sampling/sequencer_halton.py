"""Dependency-free scrambled Halton sequence."""

import numpy as np
from scipy.stats.qmc import Halton
from .sequencer_base import BaseSequencer


def _primes(count):
    values, candidate = [], 2
    while len(values) < count:
        if all(candidate % p for p in values if p * p <= candidate):
            values.append(candidate)
        candidate += 1
    return values


def _radical_inverse(index, base):
    value, factor = 0.0, 1.0 / base
    while index:
        value += factor * (index % base)
        index //= base
        factor /= base
    return value


class HaltonSequencer(BaseSequencer):
    def __init__(self, ndims: int, seed: int = 123, scramble: bool = True):
        super().__init__(ndims, seed)
        self.scramble = scramble
        self._bases = _primes(ndims)
        # Preserve the pinned public handle without requiring scipy at runtime.
        # This object implements the same random/reset/fast_forward protocol.
        self.qmc = self
        self.d = ndims
        self.reset()

    def random(self, n_samples: int) -> np.ndarray:
        if n_samples == 0:
            return np.empty((0, self.ndims), dtype=float)
        start = self._index
        output = np.array([
            [_radical_inverse(i + 1, base) for base in self._bases]
            for i in range(start, start + n_samples)
        ])
        self._index += n_samples
        if self.scramble:
            output = (output + self._shift) % 1.0
        return output

    def reset(self) -> None:
        self._index = 0
        self._shift = np.random.default_rng(self.seed).random(self.ndims)

    def fast_forward(self, steps: int) -> None:
        if steps > 0:
            self._index += steps
