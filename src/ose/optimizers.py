
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


def particle_swarm_optimization(
    objective,
    bounds,
    max_evaluations=1000,
    seed=42,
    swarm_size=30,
    w=0.7,
    c1=1.5,
    c2=1.5,
):
    """Minimize an objective using reproducible Particle Swarm Optimization."""

    bounds = np.asarray(bounds, dtype=float)

    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (dimensions, 2)")

    if bounds.shape[0] == 0:
        raise ValueError("at least one dimension is required")

    if not np.isfinite(bounds).all():
        raise ValueError("bounds must contain finite values")

    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("each lower bound must be below its upper bound")

    if (
        not isinstance(max_evaluations, int)
        or isinstance(max_evaluations, bool)
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer")

    if (
        not isinstance(swarm_size, int)
        or isinstance(swarm_size, bool)
        or swarm_size < 1
    ):
        raise ValueError("swarm_size must be a positive integer")

    if not np.isfinite([w, c1, c2]).all():
        raise ValueError("w, c1 and c2 must be finite")

    if w < 0 or c1 < 0 or c2 < 0:
        raise ValueError("w, c1 and c2 must be non-negative")

    rng = np.random.default_rng(seed)

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    span = upper - lower
    dimensions = len(bounds)

    # Do not initialize more particles than the evaluation budget.
    n_particles = min(swarm_size, max_evaluations)

    positions = rng.uniform(
        lower, upper, size=(n_particles, dimensions)
    )
    velocities = rng.uniform(
        -0.1 * span, 0.1 * span,
        size=(n_particles, dimensions)
    )

    personal_best_positions = positions.copy()
    personal_best_values = np.full(n_particles, np.inf)

    global_best_position = None
    global_best_value = float("inf")
    evaluations = 0

    # Evaluate the initial swarm.
    for i in range(n_particles):
        value = float(objective(positions[i]))
        evaluations += 1

        if not np.isfinite(value):
            raise ValueError("objective returned a non-finite value")

        personal_best_values[i] = value

        if value < global_best_value:
            global_best_value = value
            global_best_position = positions[i].copy()

    # Improve the swarm until the exact evaluation budget is used.
    while evaluations < max_evaluations:
        for i in range(n_particles):
            if evaluations >= max_evaluations:
                break

            r1 = rng.random(dimensions)
            r2 = rng.random(dimensions)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (
                    personal_best_positions[i] - positions[i]
                )
                + c2 * r2 * (
                    global_best_position - positions[i]
                )
            )

            positions[i] = np.clip(
                positions[i] + velocities[i],
                lower,
                upper,
            )

            value = float(objective(positions[i]))
            evaluations += 1

            if not np.isfinite(value):
                raise ValueError(
                    "objective returned a non-finite value"
                )

            if value < personal_best_values[i]:
                personal_best_values[i] = value
                personal_best_positions[i] = positions[i].copy()

                if value < global_best_value:
                    global_best_value = value
                    global_best_position = positions[i].copy()

    return {
        "best_x": global_best_position.tolist(),
        "best_value": float(global_best_value),
        "evaluations": evaluations,
        "seed": seed,
    }

