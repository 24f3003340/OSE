
"""Run reproducible OSE benchmark experiments."""

import csv
from pathlib import Path

from src.ose.objectives import sphere
from src.ose.optimizers import random_search


def run_sphere_experiment(
    seeds=range(10),
    dimensions=2,
    max_evaluations=1000,
    output_path="results/experiment_001_sphere.csv",
):
    """Run Random Search on Sphere using multiple seeds."""

    if dimensions < 1:
        raise ValueError("dimensions must be positive")

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")

    bounds = [(-5.0, 5.0)] * dimensions
    rows = []

    for seed in seeds:
        result = random_search(
            objective=sphere,
            bounds=bounds,
            max_evaluations=max_evaluations,
            seed=int(seed),
        )

        rows.append({
            "experiment_id": "001",
            "benchmark": "Sphere",
            "algorithm": "Random Search",
            "dimensions": dimensions,
            "seed": int(seed),
            "evaluations": result["evaluations"],
            "best_value": result["best_value"],
            "best_x": result["best_x"],
        })

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id", "benchmark", "algorithm",
        "dimensions", "seed", "evaluations",
        "best_value", "best_x",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows
