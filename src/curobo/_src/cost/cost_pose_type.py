from enum import Enum


class PoseErrorType(Enum):
    SINGLE_GOAL = 0
    BATCH_GOAL = 1
    GOALSET = 2
    BATCH_GOALSET = 3


__all__ = ["PoseErrorType"]
