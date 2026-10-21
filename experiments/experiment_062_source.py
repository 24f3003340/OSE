# OSE Experiment 062
# Extracted from the original notebook.
# Review cell order and dependencies before running as a script.


# ===== Original notebook cell 180 (markdown) =====
Experiment 062


# ===== Original notebook cell 181 (code) =====

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests

ROOT = Path("/content/OSE")
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = ["059", "060"]
N_BOOTSTRAP = 5000
ALPHA = 0.05
start_time = time.perf_counter()

all_comparisons = []
audit_checks = []
missing_files = []

# Each experiment has its own independent audit.
for exp_id in EXPERIMENTS:
    path = OUT / f"experiment_{exp_id}_results.csv"

    if not path.exists():
        missing_files.append(str(path))
        print("MISSING:", path)
        continue

    df = pd.read_csv(path)

    required = {
        "dimension", "benchmark", "algorithm",
        "seed", "best_score"
    }
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        audit_checks.append({
            "experiment": exp_id,
            "check": "required_columns",
            "status": "FAIL",
            "details": str(missing)
        })
        continue

    audit_checks.append({
        "experiment": exp_id,
        "check": "required_columns",
        "status": "PASS",
        "details": "All required columns exist"
    })

    # Compare every algorithm pair within each benchmark group.
    for (dim, benchmark), group in df.groupby(
        ["dimension", "benchmark"]
    ):
        algorithms = sorted(group["algorithm"].unique())

        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                alg_a = algorithms[i]
                alg_b = algorithms[j]

                a = group[group["algorithm"] == alg_a][
                    ["seed", "best_score"]
                ].rename(columns={"best_score": "score_a"})

                b = group[group["algorithm"] == alg_b][
                    ["seed", "best_score"]
                ].rename(columns={"best_score": "score_b"})

                paired = a.merge(
                    b, on="seed", how="inner", validate="one_to_one"
                ).sort_values("seed")

                if len(paired) < 2:
                    all_comparisons.append({
                        "experiment": exp_id,
                        "dimension": dim,
                        "benchmark": benchmark,
                        "algorithm_a": alg_a,
                        "algorithm_b": alg_b,
                        "n_pairs": len(paired),
                        "status": "INSUFFICIENT_PAIRS"
                    })
                    continue

                x = paired["score_a"].to_numpy(dtype=float)
                y = paired["score_b"].to_numpy(dtype=float)

                # Improvement > 0 means algorithm B has lower scores.
                improvement = x - y

                try:
                    test = wilcoxon(
                        improvement,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto"
                    )
                    p_two_sided = float(test.pvalue)
                    statistic = float(test.statistic)
                except Exception:
                    p_two_sided = np.nan
                    statistic = np.nan

                try:
                    test_b_lower = wilcoxon(
                        x, y,
                        alternative="greater",
                        zero_method="wilcox",
                        method="auto"
                    )
                    p_b_better = float(test_b_lower.pvalue)
                except Exception:
                    p_b_better = np.nan

                # Paired bootstrap CI for mean(score_a - score_b).
                try:
                    boot = bootstrap(
                        (improvement,),
                        statistic=np.mean,
                        paired=False,
                        vectorized=False,
                        n_resamples=N_BOOTSTRAP,
                        confidence_level=0.95,
                        method="percentile",
                        rng=np.random.default_rng(20261014)
                    )
                    ci_low = float(boot.confidence_interval.low)
                    ci_high = float(boot.confidence_interval.high)
                except Exception:
                    ci_low = np.nan
                    ci_high = np.nan

                all_comparisons.append({
                    "experiment": exp_id,
                    "dimension": dim,
                    "benchmark": benchmark,
                    "algorithm_a": alg_a,
                    "algorithm_b": alg_b,
                    "n_pairs": len(paired),
                    "mean_score_a": float(np.mean(x)),
                    "mean_score_b": float(np.mean(y)),
                    "mean_improvement_a_minus_b": float(
                        np.mean(improvement)
                    ),
                    "median_improvement_a_minus_b": float(
                        np.median(improvement)
                    ),
                    "wilcoxon_statistic_two_sided": statistic,
                    "p_two_sided": p_two_sided,
                    "p_b_better_one_sided": p_b_better,
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "status": "CALCULATED"
                })

comparisons = pd.DataFrame(all_comparisons)

# Holm correction is applied separately to all valid two-sided
# comparisons within each experiment.
if not comparisons.empty:
    comparisons["p_holm_two_sided"] = np.nan

    for exp_id in EXPERIMENTS:
        mask = (
            (comparisons["experiment"] == exp_id)
            & comparisons["p_two_sided"].notna()
        )
        pvals = comparisons.loc[mask, "p_two_sided"].to_numpy()

        if len(pvals):
            adjusted = multipletests(
                pvals, alpha=ALPHA, method="holm"
            )[1]
            comparisons.loc[mask, "p_holm_two_sided"] = adjusted

    comparisons["significant_holm_005"] = (
        comparisons["p_holm_two_sided"] < ALPHA
    )

# Save unique Experiment 062 files.
comparisons_path = OUT / "experiment_062_statistical_comparisons.csv"
checks_path = OUT / "experiment_062_audit_checks.csv"
json_path = OUT / "experiment_062_research_record.json"
md_path = OUT / "experiment_062_research_record.md"

comparisons.to_csv(comparisons_path, index=False)
pd.DataFrame(audit_checks).to_csv(checks_path, index=False)

record = {
    "experiment_id": "OSE-062",
    "title": "Statistical Audit and Reproducibility Verification",
    "input_experiments": EXPERIMENTS,
    "missing_files": missing_files,
    "comparison_rows": len(comparisons),
    "bootstrap_resamples": N_BOOTSTRAP,
    "alpha": ALPHA,
    "holm_scope": "All valid two-sided comparisons within each experiment",
    "notes": [
        "Positive mean(score_a - score_b) favours algorithm B.",
        "Two-sided p-values test for a difference, not a specific winner.",
        "One-sided p-values test whether algorithm B has lower scores.",
        "Confidence intervals describe mean paired score differences.",
        "This script recalculates statistics; it does not validate algorithm code.",
        "Reproducibility of exact historical p-values requires matching the original test settings."
    ],
    "checks": audit_checks,
    "runtime_seconds": time.perf_counter() - start_time
}

with open(json_path, "w") as f:
    json.dump(record, f, indent=2, default=str)

with open(md_path, "w") as f:
    f.write("# OSE Experiment 062 — Statistical Audit\n\n")
    f.write(f"- Comparison rows: {len(comparisons)}\n")
    f.write(f"- Bootstrap resamples: {N_BOOTSTRAP}\n")
    f.write("- Holm correction: within each experiment\n\n")
    f.write("## Audit checks\n\n")
    f.write(pd.DataFrame(audit_checks).to_string(index=False))
    f.write("\n\n## Recalculated comparisons\n\n")
    f.write(comparisons.to_string(index=False))
    f.write("\n\n## Interpretation notes\n\n")
    for note in record["notes"]:
        f.write(f"- {note}\n")

print("\n=== EXPERIMENT 062: AUDIT CHECKS ===")
print(pd.DataFrame(audit_checks).to_string(index=False))

print("\n=== COMPARISON COUNT ===")
print("Total comparisons:", len(comparisons))
if not comparisons.empty:
    print(comparisons.groupby("experiment").size())

print("\n=== SIGNIFICANT AFTER HOLM CORRECTION ===")
if not comparisons.empty:
    cols = [
        "experiment", "dimension", "benchmark",
        "algorithm_a", "algorithm_b",
        "mean_improvement_a_minus_b",
        "p_two_sided", "p_holm_two_sided",
        "significant_holm_005"
    ]
    print(
        comparisons.loc[
            comparisons["significant_holm_005"], cols
        ].to_string(index=False)
    )

print("\n=== FILES SAVED ===")
for p in [
    comparisons_path, checks_path, json_path, md_path
]:
    print(p)

print("\nExperiment 062 complete.")

