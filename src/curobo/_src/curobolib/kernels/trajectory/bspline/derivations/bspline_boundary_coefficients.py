"""Pure-Python B-spline basis helpers exposed by pinned cuRoboV2.

The upstream file is a developer derivation script rather than a runtime CUDA
kernel.  These functions retain its useful numerical surface without importing
or compiling CUDA.
"""

import numpy as np
from typing import Dict, Tuple


def compute_cubic_bspline_basis(t: float) -> np.ndarray:
    t = float(t)
    return np.asarray((
        (1.0 - t) ** 3 / 6.0,
        (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
        (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
        t**3 / 6.0,
    ))


def compute_cubic_bspline_derivatives(t: float, dt: float) -> Dict[str, np.ndarray]:
    t = float(t)
    position = compute_cubic_bspline_basis(t)
    velocity = np.asarray((
        -0.5 * (1.0 - t) ** 2,
        1.5 * t**2 - 2.0 * t,
        -1.5 * t**2 + t + 0.5,
        0.5 * t**2,
    )) / dt
    acceleration = np.asarray((1.0 - t, 3.0 * t - 2.0, -3.0 * t + 1.0, t)) / dt**2
    jerk = np.asarray((-1.0, 3.0, -3.0, 1.0)) / dt**3
    return {
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "jerk": jerk,
    }


def derive_fixed_knot_coefficients_degree3() -> Tuple[np.ndarray, np.ndarray]:
    derivatives = compute_cubic_bspline_derivatives(1.0, 1.0)
    matrix = np.stack(tuple(derivatives.values()))
    return np.linalg.inv(matrix), matrix


def derive_fixed_knot_coefficients_degree4():
    raise NotImplementedError("quartic derivation is a developer-only symbolic utility")


def derive_fixed_knot_coefficients_degree5():
    raise NotImplementedError("quintic derivation is a developer-only symbolic utility")
