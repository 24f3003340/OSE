# OSE Experiment 059
# Extracted from the original notebook.
# Review cell order and dependencies before running as a script.


# ===== Original notebook cell 174 (markdown) =====
Experiment 059


# ===== Original notebook cell 175 (code) =====

# ============================================================
# OSE EXPERIMENT 059
# Independent Validation Across Dimensions
# 10D + 30D | Sphere + Rastrigin
# RS vs Standard PSO vs Low-Inertia PSO vs Adaptive PSO
# 100 paired seeds | 1000 evaluations/run | 1600 runs
# ============================================================

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

ROOT = Path("/content/OSE")
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

DIMENSIONS = [10, 30]
BENCHMARKS = ["Sphere", "Rastrigin"]
BUDGET = 1000
SWARM_SIZE = 20
SEEDS = list(range(59101, 59201))
LOW, HIGH = -5.12, 5.12

ALGORITHMS = [
    "Random_Search",
    "Standard_PSO",
    "Low_Inertia_PSO",
    "Scheduled_Adaptive_PSO"
]

TARGETS = {
    "Sphere": 1e-4,
    "Rastrigin": 0.1
}

def objective(x, benchmark):
    x = np.asarray(x)

    if benchmark == "Sphere":
        return float(np.sum(x ** 2))

    return float(
        10 * len(x)
        + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    )

# ------------------------------------------------------------
# Random Search
# ------------------------------------------------------------
def run_random_search(benchmark, seed, dim, budget):
    rng = np.random.default_rng(seed)
    best_score = float("inf")

    for _ in range(budget):
        x = rng.uniform(LOW, HIGH, dim)
        best_score = min(best_score, objective(x, benchmark))

    return best_score

# ------------------------------------------------------------
# PSO strategies
# ------------------------------------------------------------
def run_pso(benchmark, seed, dim, budget, algorithm):
    rng = np.random.default_rng(seed)
    n = min(SWARM_SIZE, budget)

    positions = rng.uniform(LOW, HIGH, (n, dim))
    velocities = rng.uniform(-1, 1, (n, dim))

    pbest = positions.copy()
    pbest_scores = np.array([
        objective(x, benchmark) for x in positions
    ])

    evaluations = n
    idx = int(np.argmin(pbest_scores))
    gbest = pbest[idx].copy()
    gbest_score = float(pbest_scores[idx])

    c1 = 1.49445
    c2 = 1.49445

    while evaluations < budget:
        if algorithm == "Standard_PSO":
            w = 0.729

        elif algorithm == "Low_Inertia_PSO":
            w = 0.40

        elif algorithm == "Scheduled_Adaptive_PSO":
            progress = min(evaluations / budget, 1.0)
            w = 0.90 - 0.50 * progress

        else:
            raise ValueError(f"Unknown PSO algorithm: {algorithm}")

        r1 = rng.random((n, dim))
        r2 = rng.random((n, dim))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest - positions)
            + c2 * r2 * (gbest - positions)
        )

        positions = np.clip(
            positions + velocities, LOW, HIGH
        )

        count = min(n, budget - evaluations)

        for i in range(count):
            score = objective(positions[i], benchmark)
            evaluations += 1

            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = positions[i].copy()

            if score < gbest_score:
                gbest_score = float(score)
                gbest = positions[i].copy()

    return gbest_score

# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
rows = []
expected_runs = (
    len(DIMENSIONS)
    * len(BENCHMARKS)
    * len(SEEDS)
    * len(ALGORITHMS)
)
completed = 0
start_all = time.perf_counter()

for dim in DIMENSIONS:
    for benchmark in BENCHMARKS:
        for seed_number, seed in enumerate(SEEDS, start=1):
            for algorithm in ALGORITHMS:
                start = time.perf_counter()

                if algorithm == "Random_Search":
                    score = run_random_search(
                        benchmark, seed, dim, BUDGET
                    )
                else:
                    score = run_pso(
                        benchmark, seed, dim, BUDGET, algorithm
                    )

                runtime = time.perf_counter() - start

                rows.append({
                    "experiment": "OSE-059",
                    "dimension": dim,
                    "benchmark": benchmark,
                    "algorithm": algorithm,
                    "seed": seed,
                    "budget": BUDGET,
                    "evaluations": BUDGET,
                    "best_score": float(score),
                    "target": TARGETS[benchmark],
                    "success": bool(score <= TARGETS[benchmark]),
                    "runtime_seconds": float(runtime)
                })

                completed += 1

            if seed_number % 10 == 0:
                print(
                    f"{dim}D {benchmark}: "
                    f"{seed_number}/100 seeds; "
                    f"{completed}/{expected_runs} runs complete"
                )

total_runtime = time.perf_counter() - start_all
df = pd.DataFrame(rows)

df.to_csv(OUT / "experiment_059_results.csv", index=False)
df.to_csv(OUT / "experiment_059_history.csv", index=False)

# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------
summary = (
    df.groupby(
        ["dimension", "benchmark", "algorithm"],
        as_index=False
    )
    .agg(
        runs=("best_score", "count"),
        mean_best_score=("best_score", "mean"),
        median_best_score=("best_score", "median"),
        std_best_score=("best_score", "std"),
        min_best_score=("best_score", "min"),
        max_best_score=("best_score", "max"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        median_runtime_seconds=("runtime_seconds", "median"),
        successes=("success", "sum"),
        mean_evaluations=("evaluations", "mean")
    )
)

summary["success_rate"] = summary["successes"] / summary["runs"]
summary.to_csv(OUT / "experiment_059_summary.csv", index=False)

print("\n=== EXPERIMENT 059 SUMMARY ===")
print(summary.to_string(index=False))

# ------------------------------------------------------------
# Paired comparisons
# Positive improvement means candidate has lower score.
# Compare every pair within each dimension/benchmark group.
# ------------------------------------------------------------
comparison_pairs = [
    ("Random_Search", "Standard_PSO"),
    ("Random_Search", "Low_Inertia_PSO"),
    ("Random_Search", "Scheduled_Adaptive_PSO"),
    ("Standard_PSO", "Low_Inertia_PSO"),
    ("Standard_PSO", "Scheduled_Adaptive_PSO"),
    ("Low_Inertia_PSO", "Scheduled_Adaptive_PSO")
]

comparisons = []

for dim in DIMENSIONS:
    for benchmark in BENCHMARKS:
        part = df[
            (df["dimension"] == dim)
            & (df["benchmark"] == benchmark)
        ]

        pivot = part.pivot(
            index="seed",
            columns="algorithm",
            values="best_score"
        )

        for baseline, candidate in comparison_pairs:
            base = pivot[baseline].to_numpy()
            cand = pivot[candidate].to_numpy()
            improvement = base - cand

            try:
                p_raw = float(wilcoxon(
                    base, cand, alternative="greater"
                ).pvalue)
            except ValueError:
                p_raw = 1.0

            rng = np.random.default_rng(
                59000 + len(comparisons)
            )

            boot_means = np.empty(10000)
            for b in range(10000):
                sample = rng.choice(
                    improvement,
                    size=len(improvement),
                    replace=True
                )
                boot_means[b] = sample.mean()

            comparisons.append({
                "dimension": dim,
                "benchmark": benchmark,
                "baseline": baseline,
                "candidate": candidate,
                "paired_seeds": len(improvement),
                "mean_improvement": float(improvement.mean()),
                "median_improvement": float(np.median(improvement)),
                "CI_2.5%": float(np.quantile(boot_means, 0.025)),
                "CI_97.5%": float(np.quantile(boot_means, 0.975)),
                "p_raw": p_raw,
                "candidate_better_runs": int(
                    (improvement > 0).sum()
                ),
                "baseline_better_runs": int(
                    (improvement < 0).sum()
                ),
                "ties": int((improvement == 0).sum())
            })

comp = pd.DataFrame(comparisons)

# Holm correction across all 24 planned comparisons
pvals = comp["p_raw"].to_numpy()
order = np.argsort(pvals)
adjusted = np.empty(len(pvals))
running_max = 0.0
m = len(pvals)

for rank, idx in enumerate(order):
    corrected = min(1.0, (m - rank) * pvals[idx])
    running_max = max(running_max, corrected)
    adjusted[idx] = running_max

comp["p_holm"] = adjusted
comp["significant_holm_0.05"] = adjusted < 0.05

comp.to_csv(
    OUT / "experiment_059_paired_comparisons.csv",
    index=False
)

print("\n=== PAIRED COMPARISONS ===")
print(comp.to_string(index=False))

# ------------------------------------------------------------
# Adaptive PSO vs both fixed PSO variants
# ------------------------------------------------------------
adaptive = comp[
    comp["candidate"] == "Scheduled_Adaptive_PSO"
].copy()

adaptive.to_csv(
    OUT / "experiment_059_adaptive_comparisons.csv",
    index=False
)

print("\n=== ADAPTIVE PSO COMPARISONS ===")
print(adaptive.to_string(index=False))

# ------------------------------------------------------------
# Validation checks
# ------------------------------------------------------------
checks = {
    "expected_runs": expected_runs,
    "actual_runs": int(len(df)),
    "all_budgets_correct": bool(
        (df["evaluations"] == BUDGET).all()
    ),
    "all_scores_finite": bool(
        np.isfinite(df["best_score"]).all()
    ),
    "all_runtimes_nonnegative": bool(
        (df["runtime_seconds"] >= 0).all()
    ),
    "paired_seeds_per_group": int(
        df.groupby(
            ["dimension", "benchmark", "algorithm"]
        )["seed"].nunique().min()
    ),
    "unique_run_keys": bool(
        not df.duplicated(
            ["dimension", "benchmark", "algorithm", "seed"]
        ).any()
    )
}

print("\n=== VALIDATION CHECKS ===")
print(json.dumps(checks, indent=2))

# ------------------------------------------------------------
# Research record
# ------------------------------------------------------------
record = {
    "experiment_id": "OSE-059",
    "title": "Independent Validation Across Dimensions",
    "research_question": (
        "Do the relative performances of fixed and scheduled "
        "adaptive PSO replicate across 10D and 30D benchmarks?"
    ),
    "design": {
        "dimensions": DIMENSIONS,
        "benchmarks": BENCHMARKS,
        "algorithms": ALGORITHMS,
        "evaluation_budget_per_run": BUDGET,
        "seeds": [min(SEEDS), max(SEEDS)],
        "seeds_per_group": len(SEEDS),
        "total_runs": len(df),
        "standard_inertia": 0.729,
        "low_inertia": 0.40,
        "adaptive_inertia_schedule": "linear 0.90 to 0.40",
        "swarm_size": SWARM_SIZE,
        "c1": 1.49445,
        "c2": 1.49445
    },
    "total_wall_clock_seconds": total_runtime,
    "summary": summary.to_dict(orient="records"),
    "paired_comparisons": comp.to_dict(orient="records"),
    "validation_checks": checks,
    "limitations": [
        "Only Sphere and Rastrigin were tested.",
        "The adaptive strategy uses a predetermined schedule.",
        "Conclusions apply only to the tested dimensions, budgets and settings.",
        "Equal evaluation budgets do not imply equal computational cost."
    ]
}

with open(OUT / "experiment_059_research_record.json", "w") as f:
    json.dump(record, f, indent=2)

md = f"""# OSE Experiment 059 — Independent Validation Across Dimensions

## Research question
Do relative performances replicate across 10D and 30D benchmarks?

## Design
- Dimensions: {DIMENSIONS}
- Benchmarks: {", ".join(BENCHMARKS)}
- Algorithms: {", ".join(ALGORITHMS)}
- Evaluations per run: {BUDGET}
- Seeds per group: {len(SEEDS)}
- Total runs: {len(df)}
- Total runtime: {total_runtime:.2f} seconds

## Summary
{summary.to_string(index=False)}

## Paired comparisons
Positive mean improvement means the candidate has a lower score.
{comp.to_string(index=False)}

## Validation checks
```json id="059check"
{json.dumps(checks, indent=2)}
```

## Limitations
The adaptive PSO uses a predetermined inertia schedule.
Results apply only to the tested functions, dimensions, budgets and settings.
"""

with open(OUT / "experiment_059_research_record.md", "w") as f:
    f.write(md)

print("\n=== SAVED FILES ===")
for name in [
    "experiment_059_results.csv",
    "experiment_059_history.csv",
    "experiment_059_summary.csv",
    "experiment_059_paired_comparisons.csv",
    "experiment_059_adaptive_comparisons.csv",
    "experiment_059_research_record.json",
    "experiment_059_research_record.md"
]:
    print(OUT / name)

print("\nExperiment 059 complete.")

