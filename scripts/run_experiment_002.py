
"""Compare Random Search and PSO on the Sphere benchmark."""

import csv
import json
from pathlib import Path
from statistics import mean, median

from src.ose.objectives import sphere
from src.ose.optimizers import (
    random_search,
    particle_swarm_optimization,
)


def run_experiment():
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    seeds = range(30)
    max_evaluations = 1000

    rows = []

    for seed in seeds:
        # Both algorithms use the same seed and evaluation budget.
        algorithms = [
            (
                "Random Search",
                random_search,
                {},
            ),
            (
                "PSO",
                particle_swarm_optimization,
                {"swarm_size": 30},
            ),
        ]

        for algorithm_name, algorithm, extra_args in algorithms:
            result = algorithm(
                objective=sphere,
                bounds=bounds,
                max_evaluations=max_evaluations,
                seed=seed,
                **extra_args,
            )

            rows.append({
                "experiment_id": "002",
                "benchmark": "Sphere",
                "algorithm": algorithm_name,
                "dimensions": 2,
                "seed": seed,
                "evaluations": result["evaluations"],
                "best_value": result["best_value"],
                "best_x": json.dumps(result["best_x"]),
            })

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "experiment_002_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {}

    for algorithm_name in ("Random Search", "PSO"):
        values = [
            row["best_value"]
            for row in rows
            if row["algorithm"] == algorithm_name
        ]

        summary[algorithm_name] = {
            "runs": len(values),
            "mean_best_value": mean(values),
            "median_best_value": median(values),
            "best_value": min(values),
            "successes_below_0.1": sum(
                value < 0.1 for value in values
            ),
        }

    summary_path = output_dir / "experiment_002_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Experiment 002 completed.")
    print(f"Total runs: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_experiment()
