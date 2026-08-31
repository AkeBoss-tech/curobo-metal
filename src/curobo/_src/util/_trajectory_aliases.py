"""Lazy wrappers for trajectory helpers owned by the portable implementation."""


def get_cuda_linear_interpolation(*args, **kwargs):
    from .trajectory import _get_cuda_linear_interpolation_portable

    return _get_cuda_linear_interpolation_portable(*args, **kwargs)


def get_bspline_interpolation(*args, **kwargs):
    from .trajectory import _get_bspline_interpolation_portable

    return _get_bspline_interpolation_portable(*args, **kwargs)

