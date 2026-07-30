"""Pure-Python arithmetic helpers."""


def ceildiv(a: int, b: int):
    return -(a // -b)
