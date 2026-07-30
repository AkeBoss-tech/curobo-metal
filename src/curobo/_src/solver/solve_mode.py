from enum import Enum
from typing import Literal, Union


class SolveMode(Enum):
    SINGLE = "single"
    BATCH = "batch"
    MULTI_ENV = "multi_env"


SolveModeInput = Union[SolveMode, Literal["single", "batch", "multi_env"]]


def parse_solve_mode(mode: SolveModeInput) -> SolveMode:
    return mode if isinstance(mode, SolveMode) else SolveMode(mode)


__all__ = ["SolveMode", "SolveModeInput", "parse_solve_mode"]
