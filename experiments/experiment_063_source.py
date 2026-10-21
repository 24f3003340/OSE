# OSE Experiment 063
# Extracted from the original notebook.
# Review cell order and dependencies before running as a script.


# ===== Original notebook cell 182 (markdown) =====
Experiment 063


# ===== Original notebook cell 183 (code) =====

from pathlib import Path
import json
import time
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path("/content/OSE")
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = ["059", "060"]
N_BOOTSTRAP = 10000
ALPHA = 0.05
BOOTSTRAP_SEED = 63014

start = time.perf_counter()
comparison_rows = []
audit_rows = []
missing_files = []

# Fixed-seed paired bootstrap for reproducible intervals.
def paired_bootstrap_ci(differences, n_resamples, rng):
    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    if n < 2:
        return np.nan, np.nan

    means = np.empty(n_resamples)

    for i in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        means[i] = differences[sample_idx].mean()

    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


for exp_id in EXPERIMENTS:
    path = OUT / f"experiment_{exp_id}_results.csv"

    if not path.exists():
        missing_files.append(str(path))
        audit_rows.append({
            "experiment": exp_id,
            "check": "input_file",
            "status": "FAIL",
            "details": str(path)
        })
        continue

    df = pd.read_csv(path)

    required = {
        "dimension", "benchmark", "algorithm",
        "seed", "best_score"
    }
    missing_cols = required - set(df.columns)

    if missing_cols:
        audit_rows.append({
            "experiment": exp_id,
            "check": "required_columns",
            "status": "FAIL",
            "details": str(sorted(missing_cols))
        })
        continue

    audit_rows.append({
        "experiment": exp_id,
        "check": "required_columns",
        "status": "PASS",
        "details": "Required columns available"
    })

    for (dim, benchmark), group in df.groupby(
        ["dimension", "benchmark"]
    ):
        algorithms = sorted(group["algorithm"].unique())

        for alg_a, alg_b in itertools.combinations(algorithms, 2):
            a = group.loc[
                group["algorithm"] == alg_a, ["seed", "best_score"]
            ].rename(columns={"best_score": "score_a"})

            b = group.loc[
                group["algorithm"] == alg_b, ["seed", "best_score"]
            ].rename(columns={"best_score": "score_b"})

            paired = a.merge(
                b, on="seed", how="inner", validate="one_to_one"
            ).sort_values("seed")

            x = paired["score_a"].to_numpy(dtype=float)
            y = paired["score_b"].to_numpy(dtype=float)

            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            n = len(x)

            if n < 2:
                comparison_rows.append({
                    "experiment": exp_id,
                    "dimension": dim,
                    "benchmark": benchmark,
                    "algorithm_a": alg_a,
                    "algorithm_b": alg_b,
                    "n_pairs": n,
                    "status": "INSUFFICIENT_PAIRS"
                })
                continue

            # Positive difference means A has a higher score,
            # therefore B is better for minimization.
            d = x - y
            mean_d = float(np.mean(d))
            median_d = float(np.median(d))

            rng = np.random.default_rng(
                BOOTSTRAP_SEED
                + int(exp_id)
                + int(dim)
                + sum(ord(c) for c in benchmark + alg_a + alg_b)
            )
            ci_low, ci_high = paired_bootstrap_ci(
                d, N_BOOTSTRAP, rng
            )

            # Handle identical scores explicitly.
            if np.allclose(d, 0.0, rtol=0, atol=0):
                stat_two, p_two = 0.0, 1.0
                p_b_better = 1.0
                p_a_better = 1.0
                test_status = "ALL_PAIRED_DIFFERENCES_ZERO"
            else:
                try:
                    res_two = wilcoxon(
                        x, y,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto"
                    )
                    stat_two = float(res_two.statistic)
                    p_two = float(res_two.pvalue)

                    res_b = wilcoxon(
                        x, y,
                        alternative="greater",
                        zero_method="wilcox",
                        method="auto"
                    )
                    p_b_better = float(res_b.pvalue)

                    res_a = wilcoxon(
                        x, y,
                        alternative="less",
                        zero_method="wilcox",
                        method="auto"
                    )
                    p_a_better = float(res_a.pvalue)
                    test_status = "CALCULATED"
                except Exception as e:
                    stat_two = np.nan
                    p_two = np.nan
                    p_b_better = np.nan
                    p_a_better = np.nan
                    test_status = f"TEST_ERROR: {type(e).__name__}"

            comparison_rows.append({
                "experiment": exp_id,
                "dimension": dim,
                "benchmark": benchmark,
                "algorithm_a": alg_a,
                "algorithm_b": alg_b,
                "n_pairs": n,
                "mean_score_a": float(np.mean(x)),
                "mean_score_b": float(np.mean(y)),
                "mean_difference_a_minus_b": mean_d,
                "median_difference_a_minus_b": median_d,
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "wilcoxon_statistic": stat_two,
                "p_two_sided": p_two,
                "p_b_better_one_sided": p_b_better,
                "p_a_better_one_sided": p_a_better,
                "test_status": test_status,
                "status": "CALCULATED"
            })

comparisons = pd.DataFrame(comparison_rows)

# Holm correction for two-sided tests, separately by experiment.
# The family is all valid pairwise comparisons in that experiment.
if not comparisons.empty:
    comparisons["p_holm_two_sided"] = np.nan
    comparisons["significant_holm_005"] = False

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
            comparisons.loc[mask, "significant_holm_005"] = (
                adjusted < ALPHA
            )

# An interval wholly above zero favours B; wholly below zero favours A.
def interval_interpretation(row):
    lo = row["paired_bootstrap_ci_low"]
    hi = row["paired_bootstrap_ci_high"]

    if not np.isfinite(lo) or not np.isfinite(hi):
        return "UNAVAILABLE"
    if lo > 0:
        return "B_LOWER_SCORE"
    if hi < 0:
        return "A_LOWER_SCORE"
    return "INTERVAL_INCLUDES_ZERO"

if not comparisons.empty:
    comparisons["bootstrap_direction"] = comparisons.apply(
        interval_interpretation, axis=1
    )

checks_df = pd.DataFrame(audit_rows)

comparisons_path = OUT / "experiment_063_corrected_comparisons.csv"
checks_path = OUT / "experiment_063_audit_checks.csv"
json_path = OUT / "experiment_063_research_record.json"
md_path = OUT / "experiment_063_research_record.md"

comparisons.to_csv(comparisons_path, index=False)
checks_df.to_csv(checks_path, index=False)

record = {
    "experiment_id": "OSE-063",
    "title": "Corrected Statistical Verification",
    "input_experiments": EXPERIMENTS,
    "missing_files": missing_files,
    "comparison_count": len(comparisons),
    "bootstrap_resamples": N_BOOTSTRAP,
    "bootstrap_method": "Paired resampling of within-seed score differences",
    "confidence_level": 0.95,
    "holm_scope": "All valid two-sided pairwise comparisons within each experiment",
    "alpha": ALPHA,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "limitations": [
        "Statistical significance does not establish practical importance.",
        "Pairwise comparisons are not the same as a pre-registered primary hypothesis.",
        "The audit recalculates statistics from recorded scores; it does not rerun optimizers.",
        "Bootstrap confidence intervals and Wilcoxon tests use different assumptions."
    ],
    "audit_checks": audit_rows,
    "runtime_seconds": time.perf_counter() - start
}

with open(json_path, "w") as f:
    json.dump(record, f, indent=2, default=str)

with open(md_path, "w") as f:
    f.write("# OSE Experiment 063 — Corrected Statistical Verification\n\n")
    f.write(f"- Pairwise comparisons: {len(comparisons)}\n")
    f.write(f"- Bootstrap resamples per comparison: {N_BOOTSTRAP}\n")
    f.write("- Confidence intervals: paired bootstrap, 95%\n")
    f.write("- Holm correction: within each experiment\n\n")
    f.write("## Audit checks\n\n")
    f.write(checks_df.to_string(index=False))
    f.write("\n\n## Comparisons\n\n")
    f.write(comparisons.to_string(index=False))
    f.write("\n\n## Limitations\n\n")
    for note in record["limitations"]:
        f.write(f"- {note}\n")

print("\n=== EXPERIMENT 063: AUDIT CHECKS ===")
print(checks_df.to_string(index=False))

print("\n=== COMPARISON COUNT ===")
print("Total:", len(comparisons))
if not comparisons.empty:
    print(comparisons.groupby("experiment").size())

print("\n=== BOOTSTRAP DIRECTION COUNTS ===")
if not comparisons.empty:
    print(comparisons["bootstrap_direction"].value_counts(dropna=False))

print("\n=== SIGNIFICANT TWO-SIDED TESTS AFTER HOLM ===")
if not comparisons.empty:
    cols = [
        "experiment", "dimension", "benchmark",
        "algorithm_a", "algorithm_b",
        "mean_difference_a_minus_b",
        "paired_bootstrap_ci_low",
        "paired_bootstrap_ci_high",
        "p_two_sided", "p_holm_two_sided",
        "bootstrap_direction"
    ]
    print(
        comparisons.loc[
            comparisons["significant_holm_005"], cols
        ].to_string(index=False)
    )

print("\n=== SAVED FILES ===")
for p in [
    comparisons_path, checks_path, json_path, md_path
]:
    print(p)

print("\nExperiment 063 complete.")

