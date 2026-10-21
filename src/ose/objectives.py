
"""Benchmark objective functions for OSE optimization research."""

import numpy as np


def sphere(x):
    """Sphere function: global minimum is 0 at the origin."""
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D vector")

    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")

    return float(np.sum(x ** 2))


def rastrigin(x):
    """Rastrigin function: global minimum is 0 at the origin."""
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1D vector")

    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")

    return float(
        10 * x.size
        + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    )
