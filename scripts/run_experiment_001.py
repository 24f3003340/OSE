
import json
import statistics
from pathlib import Path

from src.ose.experiment_runner import run_sphere_experiment


output = Path("results/experiment_001_sphere.csv")

rows = run_sphere_experiment(
    seeds=range(10),
    dimensions=2,
    max_evaluations=1000,
    output_path=str(output),
)

scores = [row["best_value"] for row in rows]

summary = {
    "experiment_id": "001",
    "benchmark": "Sphere",
    "algorithm": "Random Search",
    "dimensions": 2,
    "seeds": len(rows),
    "evaluations_per_seed": 1000,
    "total_evaluations": sum(row["evaluations"] for row in rows),
    "mean_best_value": statistics.mean(scores),
    "median_best_value": statistics.median(scores),
    "best_observed_value": min(scores),
    "worst_observed_value": max(scores),
}

summary_path = Path("results/experiment_001_summary.json")
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, indent=2))
print(f"\nRaw results saved to: {output}")
print(f"Summary saved to: {summary_path}")
