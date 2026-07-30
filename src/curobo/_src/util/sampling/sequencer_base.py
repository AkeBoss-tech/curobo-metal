"""Base interface for deterministic sample sequences."""

from abc import ABC, abstractmethod


class BaseSequencer(ABC):
    def __init__(self, ndims: int, seed: int = 123):
        if ndims < 1:
            raise ValueError("ndims must be positive")
        self.ndims, self.seed = ndims, seed

    @abstractmethod
    def random(self, n_samples: int): ...

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def fast_forward(self, steps: int): ...
