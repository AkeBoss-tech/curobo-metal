"""Pure-Python arithmetic helpers."""


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)
