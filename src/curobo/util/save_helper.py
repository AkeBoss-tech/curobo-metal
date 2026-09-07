"""Small compatibility writer used by established example applications."""

from pathlib import Path
from typing import Any

import numpy as np


class SaveHelper:
    def __init__(self, output_path="output"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save_npy(self, name: str, value: Any):
        path = self.output_path / name
        if path.suffix != ".npy":
            path = path.with_suffix(".npy")
        np.save(path, value)
        return str(path)


__all__ = ["SaveHelper"]
