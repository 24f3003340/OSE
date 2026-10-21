
"""Simple, reproducible optimization algorithms for OSE."""

import numpy as np


def random_search(
    objective,
    bounds,
    max_evaluations=1000,
    seed=42,
):
    """Minimize an objective using reproducible random search."""

    bounds = np.asarray(bounds, dtype=float)

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (dimensions, 2)")

    if bounds.shape[0] == 0:
        raise ValueError("at least one dimension is required")

    if not np.isfinite(bounds).all():
        raise ValueError("bounds must contain finite values")

    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be below its upper bound")

    if not isinstance(max_evaluations, int) or isinstance(
        max_evaluations, bool
    ) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")

    rng = np.random.default_rng(seed)

    best_x = None
    best_value = float("inf")

    for _ in range(max_evaluations):
        candidate = rng.uniform(
            bounds[:, 0], bounds[:, 1]
        )
        value = float(objective(candidate))

        if not np.isfinite(value):
            raise ValueError("objective returned a non-finite value")

        if value < best_value:
            best_x = candidate.copy()
            best_value = value

    return {
        "best_x": best_x.tolist(),
        "best_value": best_value,
        "evaluations": max_evaluations,
        "seed": seed,
    }
