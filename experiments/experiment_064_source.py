# OSE Experiment 064
# Extracted from the original notebook.
# Review cell order and dependencies before running as a script.


# ===== Original notebook cell 184 (markdown) =====
Experiment 064


# ===== Original notebook cell 185 (code) =====

# ============================================================
# OPEN SCIENCE ENGINE (OSE)
# EXPERIMENT 064
# Statistical Consistency & Reproducibility Audit
# ============================================================

from pathlib import Path
import json
import hashlib
import time
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# -----------------------------
# 1. Configuration
# -----------------------------

ROOT = Path("/content/OSE")
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "059": OUT / "experiment_059_results.csv",
    "060": OUT / "experiment_060_results.csv",
}

PRIOR_FILE = OUT / "experiment_063_corrected_comparisons.csv"

N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 63014
ALPHA = 0.05
TOL = 1e-8

start_time = time.perf_counter()

audit_rows = []
comparison_rows = []
input_hashes = {}
missing_files = []
raw_data = {}

# -----------------------------
# 2. Helper functions
# -----------------------------

def sha256_file(path):
    """Return SHA-256 hash of a file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def stable_rng_seed(exp_id, dim, benchmark, alg_a, alg_b):
    """
    Deterministic bootstrap seed.
    Sorted algorithm pairs ensure stable pair ordering.
    """
    text = (
        f"{exp_id}|{dim}|{benchmark}|"
        f"{alg_a}|{alg_b}|{BOOTSTRAP_SEED}"
    )

    digest = hashlib.sha256(text.encode()).hexdigest()
    return int(digest[:8], 16)


def paired_bootstrap(differences, n_resamples, seed):
    """
    Bootstrap the paired within-seed differences.
    Returns the percentile 95% confidence interval.
    """
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)

    n = len(differences)
    boot_means = np.empty(n_resamples)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(differences[idx])

    low, high = np.quantile(
        boot_means, [0.025, 0.975]
    )

    return float(low), float(high)


def record_check(experiment, check, status, details):
    audit_rows.append({
        "experiment": str(experiment),
        "check": str(check),
        "status": str(status),
        "details": str(details),
    })


# -----------------------------
# 3. Load raw data and hashes
# -----------------------------

for exp_id, path in RAW_FILES.items():

    if not path.exists():
        missing_files.append(str(path))

        record_check(
            exp_id,
            "raw_file_exists",
            "FAIL",
            f"Missing file: {path}"
        )
        continue

    input_hashes[str(path)] = sha256_file(path)
    df = pd.read_csv(path)

    required = {
        "dimension",
        "benchmark",
        "algorithm",
        "seed",
        "best_score",
    }

    missing_columns = required - set(df.columns)

    if missing_columns:
        record_check(
            exp_id,
            "raw_required_columns",
            "FAIL",
            f"Missing columns: {sorted(missing_columns)}"
        )
        continue

    raw_data[exp_id] = df

    record_check(
        exp_id,
        "raw_file_and_columns",
        "PASS",
        f"{len(df)} rows; SHA-256={input_hashes[str(path)]}"
    )


# -----------------------------
# 4. Load Experiment 063 record
# -----------------------------

if PRIOR_FILE.exists():

    prior = pd.read_csv(PRIOR_FILE)
    input_hashes[str(PRIOR_FILE)] = sha256_file(PRIOR_FILE)

    record_check(
        "063",
        "prior_comparison_file",
        "PASS",
        f"Loaded {len(prior)} comparison rows."
    )

else:

    prior = pd.DataFrame()

    missing_files.append(str(PRIOR_FILE))

    record_check(
        "063",
        "prior_comparison_file",
        "FAIL",
        f"Missing file: {PRIOR_FILE}"
    )


# -----------------------------
# 5. Validate raw data
# -----------------------------

for exp_id, df in raw_data.items():

    duplicate_keys = df.duplicated(
        ["dimension", "benchmark", "algorithm", "seed"]
    ).sum()

    finite_scores = np.isfinite(
        pd.to_numeric(df["best_score"], errors="coerce")
    ).all()

    record_check(
        exp_id,
        "unique_run_keys",
        "PASS" if duplicate_keys == 0 else "FAIL",
        f"Duplicate run keys: {duplicate_keys}"
    )

    record_check(
        exp_id,
        "finite_scores",
        "PASS" if finite_scores else "FAIL",
        f"All scores finite: {finite_scores}"
    )


# -----------------------------
# 6. Recalculate paired comparisons
# -----------------------------

for exp_id, df in raw_data.items():

    for (dim, benchmark), group in df.groupby(
        ["dimension", "benchmark"]
    ):

        algorithms = sorted(group["algorithm"].unique())

        # Check whether every algorithm has the same seed set.
        seed_sets = {
            alg: set(
                group.loc[group["algorithm"] == alg, "seed"]
            )
            for alg in algorithms
        }

        same_seed_sets = (
            len(seed_sets) > 0
            and all(
                seeds == next(iter(seed_sets.values()))
                for seeds in seed_sets.values()
            )
        )

        record_check(
            exp_id,
            f"paired_seed_sets_{dim}_{benchmark}",
            "PASS" if same_seed_sets else "FAIL",
            f"Same seed sets across algorithms: {same_seed_sets}"
        )

        for alg_a, alg_b in itertools.combinations(algorithms, 2):

            a = group.loc[
                group["algorithm"] == alg_a,
                ["seed", "best_score"]
            ].rename(columns={"best_score": "score_a"})

            b = group.loc[
                group["algorithm"] == alg_b,
                ["seed", "best_score"]
            ].rename(columns={"best_score": "score_b"})

            paired = a.merge(
                b,
                on="seed",
                how="inner",
                validate="one_to_one"
            ).sort_values("seed")

            x = pd.to_numeric(
                paired["score_a"], errors="coerce"
            ).to_numpy(float)

            y = pd.to_numeric(
                paired["score_b"], errors="coerce"
            ).to_numpy(float)

            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]

            if len(x) < 2:
                record_check(
                    exp_id,
                    f"comparison_{dim}_{benchmark}_{alg_a}_{alg_b}",
                    "FAIL",
                    "Insufficient valid paired observations."
                )
                continue

            # Positive difference means algorithm B has lower scores.
            differences = x - y

            try:
                # Same two-sided Wilcoxon test convention.
                if np.allclose(differences, 0):
                    p_two = 1.0
                    p_b_lower = 0.5
                else:
                    p_two = float(
                        wilcoxon(
                            x, y,
                            alternative="two-sided",
                            method="auto"
                        ).pvalue
                    )

                    p_b_lower = float(
                        wilcoxon(
                            x, y,
                            alternative="greater",
                            method="auto"
                        ).pvalue
                    )

            except ValueError as e:
                record_check(
                    exp_id,
                    f"wilcoxon_{dim}_{benchmark}_{alg_a}_{alg_b}",
                    "REVIEW",
                    str(e)
                )
                p_two = np.nan
                p_b_lower = np.nan

            seed = stable_rng_seed(
                exp_id, dim, benchmark, alg_a, alg_b
            )

            ci_low, ci_high = paired_bootstrap(
                differences,
                N_BOOTSTRAP,
                seed
            )

            comparison_rows.append({
                "experiment": str(exp_id),
                "dimension": dim,
                "benchmark": benchmark,
                "algorithm_a": alg_a,
                "algorithm_b": alg_b,
                "n_pairs": len(differences),
                "mean_score_a": float(np.mean(x)),
                "mean_score_b": float(np.mean(y)),
                "mean_difference_a_minus_b": float(
                    np.mean(differences)
                ),
                "median_difference_a_minus_b": float(
                    np.median(differences)
                ),
                "p_two_sided": p_two,
                "p_b_lower": p_b_lower,
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "bootstrap_seed": seed,
                "bootstrap_resamples": N_BOOTSTRAP,
            })


recomputed = pd.DataFrame(comparison_rows)


# -----------------------------
# 7. Holm correction
# -----------------------------

if not recomputed.empty:

    recomputed["p_holm_two_sided"] = np.nan

    for exp_id in recomputed["experiment"].unique():

        mask = (
            (recomputed["experiment"] == exp_id)
            & recomputed["p_two_sided"].notna()
        )

        pvals = recomputed.loc[
            mask, "p_two_sided"
        ].to_numpy(float)

        if len(pvals):
            adjusted = multipletests(
                pvals,
                alpha=ALPHA,
                method="holm"
            )[1]

            recomputed.loc[
                mask, "p_holm_two_sided"
            ] = adjusted


# -----------------------------
# 8. Compare against Experiment 063
# -----------------------------

if not prior.empty and not recomputed.empty:

    keys = [
        "experiment",
        "dimension",
        "benchmark",
        "algorithm_a",
        "algorithm_b",
    ]

    # Normalize key data types before merging.
    # This fixes int64 versus object merge errors.
    for col in keys:

        if col in recomputed.columns:
            recomputed[col] = (
                recomputed[col].astype(str).str.strip()
            )

        if col in prior.columns:
            prior[col] = (
                prior[col].astype(str).str.strip()
            )

    missing_new = [
        col for col in keys
        if col not in recomputed.columns
    ]

    missing_old = [
        col for col in keys
        if col not in prior.columns
    ]

    if missing_new or missing_old:

        record_check(
            "059+060",
            "comparison_merge_columns",
            "FAIL",
            f"New missing={missing_new}; prior missing={missing_old}"
        )

        joined = pd.DataFrame()

    else:

        joined = recomputed.merge(
            prior,
            on=keys,
            how="outer",
            suffixes=("_new", "_old"),
            indicator=True
        )

        # Save merge diagnostic without overwriting old records.
        joined.to_csv(
            OUT / "experiment_064_merge_diagnostic.csv",
            index=False
        )

        print("\n=== MERGE DIAGNOSTICS ===")
        print("Total rows:", len(joined))
        print(joined["_merge"].value_counts().to_string())

        unmatched = joined.loc[
            joined["_merge"] != "both",
            keys + ["_merge"]
        ]

        if not unmatched.empty:
            print("\nUnmatched comparisons:")
            print(unmatched.to_string(index=False))

        # Check numerical consistency.
        for _, row in joined.iterrows():

            matched = row["_merge"] == "both"

            mean_new = row.get(
                "mean_difference_a_minus_b_new", np.nan
            )
            mean_old = row.get(
                "mean_difference_a_minus_b_old", np.nan
            )

            mean_match = (
                matched
                and pd.notna(mean_new)
                and pd.notna(mean_old)
                and np.isclose(
                    float(mean_new),
                    float(mean_old),
                    atol=TOL,
                    rtol=TOL
                )
            )

            # The prior Experiment 063 code used these CI names.
            old_low = pd.to_numeric(
                pd.Series([row.get("paired_bootstrap_ci_low_old")]),
                errors="coerce"
            ).iloc[0]

            old_high = pd.to_numeric(
                pd.Series([row.get("paired_bootstrap_ci_high_old")]),
                errors="coerce"
            ).iloc[0]

            new_low = pd.to_numeric(
                pd.Series([row.get("paired_bootstrap_ci_low_new")]),
                errors="coerce"
            ).iloc[0]

            new_high = pd.to_numeric(
                pd.Series([row.get("paired_bootstrap_ci_high_new")]),
                errors="coerce"
            ).iloc[0]

            ci_match = (
                matched
                and all(pd.notna(v) for v in [
                    old_low, old_high, new_low, new_high
                ])
                and np.isclose(
                    old_low, new_low, atol=TOL, rtol=TOL
                )
                and np.isclose(
                    old_high, new_high, atol=TOL, rtol=TOL
                )
            )

            if not matched:
                status = "FAIL"
            elif mean_match and ci_match:
                status = "PASS"
            else:
                status = "REVIEW"

            record_check(
                str(row.get("experiment", "")),
                "raw_metrics_vs_exp063",
                status,
                (
                    f"{row.get('benchmark')} "
                    f"{row.get('dimension')}D; "
                    f"mean_match={mean_match}; "
                    f"ci_match={ci_match}; "
                    f"merge={row['_merge']}"
                )
            )

        record_check(
            "059+060",
            "comparison_merge",
            "PASS" if (joined["_merge"] == "both").all() else "REVIEW",
            (
                f"Matched={(joined['_merge'] == 'both').sum()}; "
                f"unmatched={(joined['_merge'] != 'both').sum()}"
            )
        )

else:

    joined = pd.DataFrame()

    record_check(
        "059+060",
        "comparison_merge",
        "REVIEW",
        "Prior comparison file or recomputed comparisons are empty."
    )


# -----------------------------
# 9. Final audit status
# -----------------------------

audit_df = pd.DataFrame(audit_rows)

overall_status = "PASS"

if missing_files or (
    not audit_df.empty
    and (audit_df["status"] == "FAIL").any()
):
    overall_status = "FAIL"

elif (
    not audit_df.empty
    and (audit_df["status"] == "REVIEW").any()
):
    overall_status = "REVIEW"


# -----------------------------
# 10. Save Experiment 064 files
# -----------------------------

recomputed_path = OUT / "experiment_064_recomputed_metrics.csv"
audit_path = OUT / "experiment_064_audit_checks.csv"
json_path = OUT / "experiment_064_research_record.json"
md_path = OUT / "experiment_064_research_record.md"

recomputed.to_csv(recomputed_path, index=False)
audit_df.to_csv(audit_path, index=False)

record = {
    "experiment_id": "OSE-064",
    "title": "Statistical Consistency and Reproducibility Audit",
    "overall_status": overall_status,
    "input_hashes_sha256": input_hashes,
    "missing_files": missing_files,
    "recomputed_comparisons": len(recomputed),
    "bootstrap_resamples_per_comparison": N_BOOTSTRAP,
    "bootstrap_confidence_level": 0.95,
    "holm_correction": "Within each experiment",
    "audit_checks": audit_rows,
    "limitations": [
        "This is a reanalysis of historical results, not new optimizer runs.",
        "A passing audit verifies selected consistency checks, not the scientific hypothesis.",
        "Wilcoxon p-values and percentile bootstrap confidence intervals use different methods.",
        "A REVIEW status requires investigation and does not automatically mean the data are wrong.",
    ],
    "runtime_seconds": round(
        time.perf_counter() - start_time, 3
    ),
}

with open(json_path, "w") as f:
    json.dump(record, f, indent=2, default=str)

with open(md_path, "w") as f:

    f.write("# OSE Experiment 064\n\n")
    f.write("## Statistical Consistency & Reproducibility Audit\n\n")
    f.write(f"**Overall status:** {overall_status}\n\n")
    f.write(f"**Recomputed comparisons:** {len(recomputed)}\n\n")
    f.write("## Audit checks\n\n")

    if not audit_df.empty:
        f.write(audit_df.to_string(index=False))

    f.write("\n\n## Limitations\n\n")

    for note in record["limitations"]:
        f.write(f"- {note}\n")


# -----------------------------
# 11. Print final report
# -----------------------------

print("\n" + "=" * 60)
print("OSE EXPERIMENT 064 — FINAL REPORT")
print("=" * 60)

print("\nOVERALL STATUS:", overall_status)
print("Recomputed comparisons:", len(recomputed))
print("Missing files:", len(missing_files))

if not audit_df.empty:
    print("\n=== AUDIT STATUS COUNTS ===")
    print(audit_df["status"].value_counts().to_string())

    print("\n=== AUDIT CHECKS ===")
    print(audit_df.to_string(index=False))

print("\n=== RECOMPUTED COMPARISONS PREVIEW ===")
if not recomputed.empty:
    print(recomputed.head(10).to_string(index=False))

print("\n=== SAVED FILES ===")
for path in [
    recomputed_path,
    audit_path,
    json_path,
    md_path,
    OUT / "experiment_064_merge_diagnostic.csv",
]:
    if path.exists():
        print(path)

print("\nRuntime:", round(time.perf_counter() - start_time, 2), "seconds")
print("\nExperiment 064 completed.")



# ===== Original notebook cell 186 (code) =====

# OSE Experiment 064 — Normalize experiment IDs and merge again

keys = [
    "experiment",
    "dimension",
    "benchmark",
    "algorithm_a",
    "algorithm_b",
]

# Normalize experiment IDs: 59 -> 059, 60 -> 060
for df in [recomputed, prior]:
    df["experiment"] = (
        df["experiment"]
        .astype(str)
        .str.strip()
        .str.replace(r"^0*(\d+)$", r"\1", regex=True)
        .str.zfill(3)
    )

    # Normalize the remaining comparison keys.
    for col in keys:
        if col != "experiment":
            df[col] = df[col].astype(str).str.strip()

# Verify merge-key uniqueness before merging.
print("Duplicate keys in recomputed:",
      recomputed.duplicated(keys).sum())

print("Duplicate keys in prior:",
      prior.duplicated(keys).sum())

# Compare records
joined_fixed = recomputed.merge(
    prior,
    on=keys,
    how="outer",
    suffixes=("_new", "_old"),
    indicator=True,
    validate="one_to_one",
)

print("\n=== NORMALIZED MERGE RESULT ===")
print("Total rows:", len(joined_fixed))
print(joined_fixed["_merge"].value_counts())

print("\nUnmatched records:")
unmatched = joined_fixed.loc[
    joined_fixed["_merge"] != "both",
    keys + ["_merge"]
]
print(unmatched.to_string(index=False))

# Save as a new diagnostic file
diagnostic_path = (
    OUT / "experiment_064_merge_diagnostic_fixed.csv"
)
joined_fixed.to_csv(diagnostic_path, index=False)

print("\nSaved:", diagnostic_path)



# ===== Original notebook cell 187 (code) =====

# OSE Experiment 064 — Compare matched statistical metrics

import numpy as np
import pandas as pd

checks = []

def compare_numeric(row, old_col, new_col, tolerance=1e-8):
    old = pd.to_numeric(
        pd.Series([row.get(old_col)]), errors="coerce"
    ).iloc[0]
    new = pd.to_numeric(
        pd.Series([row.get(new_col)]), errors="coerce"
    ).iloc[0]

    if pd.isna(old) or pd.isna(new):
        return False

    return bool(
        np.isclose(old, new, atol=tolerance, rtol=tolerance)
    )

for _, row in joined_fixed.iterrows():

    if row["_merge"] != "both":
        continue

    # Experiment 063 may use either of these historical names.
    mean_old = (
        "mean_difference_a_minus_b_old"
        if "mean_difference_a_minus_b_old" in joined_fixed.columns
        else None
    )

    mean_new = (
        "mean_difference_a_minus_b_new"
        if "mean_difference_a_minus_b_new" in joined_fixed.columns
        else None
    )

    # Discover the actual paired CI column names.
    old_low_col = next(
        (c for c in [
            "paired_bootstrap_ci_low_old",
            "bootstrap_ci_low_old"
        ] if c in joined_fixed.columns),
        None
    )

    old_high_col = next(
        (c for c in [
            "paired_bootstrap_ci_high_old",
            "bootstrap_ci_high_old"
        ] if c in joined_fixed.columns),
        None
    )

    new_low_col = next(
        (c for c in [
            "paired_bootstrap_ci_low_new",
            "bootstrap_ci_low_new"
        ] if c in joined_fixed.columns),
        None
    )

    new_high_col = next(
        (c for c in [
            "paired_bootstrap_ci_high_new",
            "bootstrap_ci_high_new"
        ] if c in joined_fixed.columns),
        None
    )

    mean_match = (
        compare_numeric(row, mean_old, mean_new)
        if mean_old and mean_new else False
    )

    ci_match = (
        compare_numeric(row, old_low_col, new_low_col)
        and compare_numeric(row, old_high_col, new_high_col)
        if all([
            old_low_col, old_high_col,
            new_low_col, new_high_col
        ]) else False
    )

    checks.append({
        "experiment": row["experiment"],
        "dimension": row["dimension"],
        "benchmark": row["benchmark"],
        "algorithm_a": row["algorithm_a"],
        "algorithm_b": row["algorithm_b"],
        "mean_match": mean_match,
        "ci_match": ci_match,
        "status": (
            "PASS" if mean_match and ci_match
            else "REVIEW"
        )
    })

checks_df = pd.DataFrame(checks)

check_path = OUT / "experiment_064_metric_consistency.csv"
checks_df.to_csv(check_path, index=False)

print("=== METRIC CONSISTENCY SUMMARY ===")
print("Total matched comparisons:", len(checks_df))
print("\nStatus counts:")
print(checks_df["status"].value_counts())

print("\nMean matches:", int(checks_df["mean_match"].sum()))
print("CI matches:", int(checks_df["ci_match"].sum()))

print("\nFirst 15 records:")
print(checks_df.head(15).to_string(index=False))

print("\nSaved:", check_path)



# ===== Original notebook cell 188 (code) =====

# Diagnose the actual confidence-interval column names

print("=== ALL MERGED COLUMN NAMES ===")
for col in joined_fixed.columns:
    print(col)

print("\n=== CI / BOOTSTRAP RELATED COLUMNS ===")
ci_cols = [
    c for c in joined_fixed.columns
    if any(word in c.lower() for word in
           ["ci", "confidence", "bootstrap", "lower", "upper"])
]
print(ci_cols)

print("\n=== FIRST 5 MATCHED RECORDS: CI VALUES ===")
if ci_cols:
    display(joined_fixed[
        ["experiment", "dimension", "benchmark",
         "algorithm_a", "algorithm_b"] + ci_cols
    ].head(5))
else:
    print("No CI-related columns found.")

print("\n=== EXPERIMENT 063 FILE SCHEMA ===")
prior_path = OUT / "experiment_063_corrected_comparisons.csv"
prior_check = pd.read_csv(prior_path)

print("Columns:")
print(prior_check.columns.tolist())

print("\nFirst 3 rows:")
display(prior_check.head(3))



# ===== Original notebook cell 189 (code) =====

# Experiment 064 — Quantify differences from Experiment 063

import numpy as np
import pandas as pd

df = joined_fixed.copy()

metric_pairs = {
    "mean_difference": (
        "mean_difference_a_minus_b_new",
        "mean_difference_a_minus_b_old"
    ),
    "ci_low": (
        "paired_bootstrap_ci_low_new",
        "paired_bootstrap_ci_low_old"
    ),
    "ci_high": (
        "paired_bootstrap_ci_high_new",
        "paired_bootstrap_ci_high_old"
    ),
    "p_two_sided": (
        "p_two_sided_new",
        "p_two_sided_old"
    ),
    "p_holm": (
        "p_holm_two_sided_new",
        "p_holm_two_sided_old"
    ),
}

rows = []

for metric, (new_col, old_col) in metric_pairs.items():
    new_values = pd.to_numeric(df[new_col], errors="coerce")
    old_values = pd.to_numeric(df[old_col], errors="coerce")

    valid = new_values.notna() & old_values.notna()
    diffs = (new_values[valid] - old_values[valid]).abs()

    rows.append({
        "metric": metric,
        "valid_comparisons": int(valid.sum()),
        "exact_matches": int(
            np.isclose(
                new_values[valid],
                old_values[valid],
                atol=1e-8,
                rtol=1e-8
            ).sum()
        ),
        "max_absolute_difference": (
            float(diffs.max()) if len(diffs) else np.nan
        ),
        "mean_absolute_difference": (
            float(diffs.mean()) if len(diffs) else np.nan
        )
    })

report = pd.DataFrame(rows)
display(report)

report.to_csv(
    OUT / "experiment_064_metric_difference_report.csv",
    index=False
)

print("Saved: experiment_064_metric_difference_report.csv")



# ===== Original notebook cell 190 (code) =====

# ================================================================
# OPEN SCIENCE ENGINE (OSE) — EXPERIMENT 064
# Statistical Consistency & Reproducibility Audit
#
# Inputs:
#   results/experiment_059_results.csv
#   results/experiment_060_results.csv
#   results/experiment_063_corrected_comparisons.csv (if available)
#
# Outputs: new Experiment 064 files only.
# Does NOT rerun Experiments 059/060 or overwrite their raw data.
# ================================================================

import os
import json
import hashlib
import itertools
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------
# 1. Setup
# ---------------------------
PROJECT_DIR = "/content/OSE"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FILE_059 = os.path.join(RESULTS_DIR, "experiment_059_results.csv")
FILE_060 = os.path.join(RESULTS_DIR, "experiment_060_results.csv")
FILE_063 = os.path.join(
    RESULTS_DIR, "experiment_063_corrected_comparisons.csv"
)

BOOTSTRAP_RESAMPLES = 10_000
MASTER_SEED = 64064
TOLERANCE = 1e-10

OUT_COMPARISONS = os.path.join(
    RESULTS_DIR, "experiment_064_recomputed_metrics.csv"
)
OUT_CHECKS = os.path.join(
    RESULTS_DIR, "experiment_064_audit_checks.csv"
)
OUT_RECORD_JSON = os.path.join(
    RESULTS_DIR, "experiment_064_research_record.json"
)
OUT_RECORD_MD = os.path.join(
    RESULTS_DIR, "experiment_064_research_record.md"
)
OUT_MERGE = os.path.join(
    RESULTS_DIR, "experiment_064_merge_diagnostic.csv"
)
OUT_DIFF_REPORT = os.path.join(
    RESULTS_DIR, "experiment_064_metric_difference_report.csv"
)

checks = []

def record_check(name, passed, details=""):
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "details": str(details)
    })

def normalize_experiment_id(value):
    """Normalize IDs such as 59, '059', and 'experiment_059'."""
    s = str(value).strip().lower()
    s = s.replace("experiment_", "").replace("experiment", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.zfill(3) if digits else s

def find_column(df, candidates, required=True):
    """Find a column using a list of supported aliases."""
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise ValueError(
            f"Required column not found. Tried {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None

def normalize_raw_file(path, experiment_id):
    """Load a raw experiment file and standardize its key columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Check that you are using the correct /content/OSE/results path."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")

    algorithm_col = find_column(
        df, ["algorithm", "algorithm_name", "method", "optimizer"]
    )
    score_col = find_column(
        df, ["best_value", "best_score", "best_objective",
             "best_objective_value", "score", "objective_value",
             "final_best_value"]
    )
    seed_col = find_column(df, ["seed", "random_seed"])
    dimension_col = find_column(
        df, ["dimension", "dimensions", "dim", "n_dimensions"]
    )
    benchmark_col = find_column(
        df, ["benchmark", "function", "objective", "function_name",
             "benchmark_name"]
    )

    runtime_col = find_column(
        df, ["runtime_seconds", "runtime", "wall_clock_seconds",
             "elapsed_seconds", "time_seconds"],
        required=False
    )
    eval_col = find_column(
        df, ["evaluations", "n_evaluations", "function_evaluations",
             "evals", "evaluation_count"],
        required=False
    )
    success_col = find_column(
        df, ["success", "target_reached", "reached_target"],
        required=False
    )

    out = pd.DataFrame({
        "experiment": str(experiment_id).zfill(3),
        "dimension": pd.to_numeric(df[dimension_col], errors="coerce"),
        "benchmark": df[benchmark_col].astype(str).str.strip(),
        "algorithm": df[algorithm_col].astype(str).str.strip(),
        "seed": df[seed_col].astype(str).str.strip(),
        "score": pd.to_numeric(df[score_col], errors="coerce")
    })

    if runtime_col:
        out["runtime_seconds"] = pd.to_numeric(
            df[runtime_col], errors="coerce"
        )
    else:
        out["runtime_seconds"] = np.nan

    if eval_col:
        out["evaluations"] = pd.to_numeric(
            df[eval_col], errors="coerce"
        )
    else:
        out["evaluations"] = np.nan

    if success_col:
        out["success"] = df[success_col].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    else:
        out["success"] = np.nan

    return out

# ---------------------------
# 2. Load Experiments 059/060
# ---------------------------
df059 = normalize_raw_file(FILE_059, 59)
df060 = normalize_raw_file(FILE_060, 60)
raw = pd.concat([df059, df060], ignore_index=True)

record_check(
    "Raw input files loaded",
    len(df059) > 0 and len(df060) > 0,
    f"Experiment 059 rows={len(df059)}; Experiment 060 rows={len(df060)}"
)

required_raw = [
    "experiment", "dimension", "benchmark", "algorithm", "seed", "score"
]
record_check(
    "Required standardized columns present",
    all(c in raw.columns for c in required_raw),
    ", ".join(required_raw)
)

record_check(
    "All objective scores finite",
    bool(np.isfinite(raw["score"].to_numpy(dtype=float)).all()),
    f"Non-finite scores={int((~np.isfinite(raw['score'])).sum())}"
)

record_check(
    "No duplicate algorithm-seed run keys",
    not raw.duplicated(
        ["experiment", "dimension", "benchmark", "algorithm", "seed"]
    ).any(),
    "Each experiment/dimension/benchmark/algorithm/seed key should be unique."
)

# ---------------------------
# 3. Deterministic paired bootstrap
# ---------------------------
def deterministic_seed(key):
    """
    Deterministic seed for this version of Experiment 064.
    This is not guaranteed to reproduce Experiment 063's bootstrap
    intervals unless Experiment 063 used the same seed-generation rule.
    """
    payload = f"{MASTER_SEED}|{key}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % (2**32 - 1)

def paired_bootstrap_ci(differences, seed, n_resamples=BOOTSTRAP_RESAMPLES):
    """
    Percentile CI for the mean of paired differences.
    Positive difference means algorithm A has a higher (worse) score
    than algorithm B. Negative means A has a lower (better) score.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    if len(differences) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(differences)
    boot_means = np.empty(n_resamples, dtype=float)

    # Process in batches to limit memory use.
    batch_size = 500
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, n, size=(stop - start, n))
        boot_means[start:stop] = differences[indices].mean(axis=1)

    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)

def holm_adjust(p_values):
    """Holm-Bonferroni adjustment; preserves NaN entries."""
    p = np.asarray(p_values, dtype=float)
    result = np.full(len(p), np.nan)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return result

    order = valid_idx[np.argsort(p[valid_idx])]
    m = len(order)
    adjusted_sorted = np.maximum.accumulate(
        [(m - rank) * p[idx] for rank, idx in enumerate(order)]
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    for idx, adjusted in zip(order, adjusted_sorted):
        result[idx] = adjusted
    return result

# ---------------------------
# 4. Recompute all pairwise comparisons
# ---------------------------
comparison_rows = []
group_cols = ["experiment", "dimension", "benchmark"]

for (exp_id, dimension, benchmark), group in raw.groupby(
    group_cols, dropna=False
):
    algorithms = sorted(group["algorithm"].unique())

    for alg_a, alg_b in itertools.combinations(algorithms, 2):
        a = group.loc[group["algorithm"] == alg_a, ["seed", "score"]]
        b = group.loc[group["algorithm"] == alg_b, ["seed", "score"]]

        a = a.rename(columns={"score": "score_a"})
        b = b.rename(columns={"score": "score_b"})

        paired = a.merge(
            b, on="seed", how="inner", validate="one_to_one"
        ).sort_values("seed")

        x = paired["score_a"].to_numpy(dtype=float)
        y = paired["score_b"].to_numpy(dtype=float)
        differences = x - y
        n = len(differences)

        mean_a = float(np.mean(x)) if n else np.nan
        mean_b = float(np.mean(y)) if n else np.nan
        mean_diff = float(np.mean(differences)) if n else np.nan
        median_diff = float(np.median(differences)) if n else np.nan

        if n >= 2 and not np.allclose(differences, 0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    w_stat, p_two = wilcoxon(
                        differences,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto"
                    )
                    _, p_b_lower = wilcoxon(
                        differences,
                        alternative="less",
                        zero_method="wilcox",
                        method="auto"
                    )
                    _, p_a_lower = wilcoxon(
                        differences,
                        alternative="greater",
                        zero_method="wilcox",
                        method="auto"
                    )
                    test_status = "OK"
                except Exception as exc:
                    w_stat, p_two, p_b_lower, p_a_lower = (
                        np.nan, np.nan, np.nan, np.nan
                    )
                    test_status = f"TEST_ERROR: {exc}"
        elif n >= 2:
            w_stat, p_two, p_b_lower, p_a_lower = 0.0, 1.0, 0.5, 0.5
            test_status = "ALL_PAIRED_DIFFERENCES_ZERO"
        else:
            w_stat, p_two, p_b_lower, p_a_lower = (
                np.nan, np.nan, np.nan, np.nan
            )
            test_status = "INSUFFICIENT_PAIRED_SEEDS"

        key = (
            f"{exp_id}|{dimension}|{benchmark}|{alg_a}|{alg_b}"
        )
        boot_seed = deterministic_seed(key)
        ci_low, ci_high = paired_bootstrap_ci(
            differences, boot_seed
        ) if n >= 2 else (np.nan, np.nan)

        if np.isfinite(ci_low) and ci_low > 0:
            direction = "A_HIGHER_SCORE"
        elif np.isfinite(ci_high) and ci_high < 0:
            direction = "A_LOWER_SCORE"
        else:
            direction = "INTERVAL_INCLUDES_ZERO"

        comparison_rows.append({
            "experiment": str(exp_id).zfill(3),
            "dimension": dimension,
            "benchmark": benchmark,
            "algorithm_a": alg_a,
            "algorithm_b": alg_b,
            "n_pairs": n,
            "mean_score_a": mean_a,
            "mean_score_b": mean_b,
            "mean_difference_a_minus_b": mean_diff,
            "median_difference_a_minus_b": median_diff,
            "wilcoxon_statistic": w_stat,
            "p_two_sided": p_two,
            "p_b_better_one_sided": p_b_lower,
            "p_a_better_one_sided": p_a_lower,
            "paired_bootstrap_ci_low": ci_low,
            "paired_bootstrap_ci_high": ci_high,
            "bootstrap_seed": boot_seed,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_method": "paired_percentile_mean_difference_95pct",
            "test_status": test_status,
            "bootstrap_direction": direction
        })

recomputed = pd.DataFrame(comparison_rows)

# Holm correction is applied within each experiment/dimension/benchmark
# family, matching the comparison family design used here.

# Holm correction applied separately within each experiment
recomputed["p_holm_two_sided"] = np.nan

for _, idx in recomputed.groupby("experiment").groups.items():
    recomputed.loc[idx, "p_holm_two_sided"] = holm_adjust(
        recomputed.loc[idx, "p_two_sided"].to_numpy(dtype=float)
    )

recomputed["significant_holm_005"] = (
    recomputed["p_holm_two_sided"] < 0.05
)


recomputed.to_csv(OUT_COMPARISONS, index=False)

record_check(
    "Pairwise comparisons generated",
    len(recomputed) > 0,
    f"Total comparisons={len(recomputed)}"
)

record_check(
    "All comparisons have at least two paired seeds",
    bool((recomputed["n_pairs"] >= 2).all()),
    f"Minimum paired seeds={recomputed['n_pairs'].min() if len(recomputed) else 0}"
)

record_check(
    "All finite p-values are within [0, 1]",
    bool(
        recomputed["p_two_sided"].dropna().between(0, 1).all()
        and recomputed["p_holm_two_sided"].dropna().between(0, 1).all()
    ),
    "Checks raw and Holm-adjusted two-sided p-values."
)

record_check(
    "Confidence intervals are ordered",
    bool((
        recomputed["paired_bootstrap_ci_low"]
        <= recomputed["paired_bootstrap_ci_high"]
    ).fillna(True).all()),
    "Lower bound must not exceed upper bound."
)

# ---------------------------
# 5. Compare against Experiment 063, if available
# ---------------------------
merge_diagnostic = pd.DataFrame()
difference_report = pd.DataFrame()

if os.path.exists(FILE_063):
    prior = pd.read_csv(FILE_063)

    prior_exp_col = find_column(prior, ["experiment"])
    prior_dim_col = find_column(prior, ["dimension"])
    prior_bench_col = find_column(prior, ["benchmark"])
    prior_a_col = find_column(prior, ["algorithm_a"])
    prior_b_col = find_column(prior, ["algorithm_b"])

    prior = prior.rename(columns={
        prior_exp_col: "experiment",
        prior_dim_col: "dimension",
        prior_bench_col: "benchmark",
        prior_a_col: "algorithm_a",
        prior_b_col: "algorithm_b"
    })

    prior["experiment"] = prior["experiment"].apply(
        normalize_experiment_id
    )
    recomputed["experiment"] = recomputed["experiment"].apply(
        normalize_experiment_id
    )

    merge_keys = [
        "experiment", "dimension", "benchmark",
        "algorithm_a", "algorithm_b"
    ]

    for col in ["dimension"]:
        prior[col] = pd.to_numeric(prior[col], errors="coerce")
        recomputed[col] = pd.to_numeric(recomputed[col], errors="coerce")

    for col in ["benchmark", "algorithm_a", "algorithm_b"]:
        prior[col] = prior[col].astype(str).str.strip()
        recomputed[col] = recomputed[col].astype(str).str.strip()

    # Avoid silent many-to-many joins.
    duplicate_new = int(recomputed.duplicated(merge_keys).sum())
    duplicate_old = int(prior.duplicated(merge_keys).sum())

    record_check(
        "Experiment 063 comparison keys unique",
        duplicate_old == 0,
        f"Duplicate keys in Experiment 063={duplicate_old}"
    )
    record_check(
        "Experiment 064 comparison keys unique",
        duplicate_new == 0,
        f"Duplicate keys in recomputed results={duplicate_new}"
    )

    old_rename = {}
    for metric in [
        "n_pairs", "mean_difference_a_minus_b",
        "paired_bootstrap_ci_low", "paired_bootstrap_ci_high",
        "p_two_sided", "p_holm_two_sided"
    ]:
        if metric in prior.columns:
            old_rename[metric] = metric + "_old"

    prior_compare = prior.rename(columns=old_rename)

    new_compare = recomputed.rename(columns={
        "n_pairs": "n_pairs_new",
        "mean_difference_a_minus_b": "mean_difference_a_minus_b_new",
        "paired_bootstrap_ci_low": "paired_bootstrap_ci_low_new",
        "paired_bootstrap_ci_high": "paired_bootstrap_ci_high_new",
        "p_two_sided": "p_two_sided_new",
        "p_holm_two_sided": "p_holm_two_sided_new"
    })

    joined = new_compare.merge(
        prior_compare,
        on=merge_keys,
        how="outer",
        indicator=True,
        validate="one_to_one"
    )

    joined.to_csv(OUT_MERGE, index=False)
    merge_diagnostic = joined

    record_check(
        "Experiment 063 and 064 comparison keys align",
        bool((joined["_merge"] == "both").all()),
        joined["_merge"].value_counts().to_dict()
    )

    metric_pairs = [
        ("mean_difference",
         "mean_difference_a_minus_b_new",
         "mean_difference_a_minus_b_old"),
        ("ci_low",
         "paired_bootstrap_ci_low_new",
         "paired_bootstrap_ci_low_old"),
        ("ci_high",
         "paired_bootstrap_ci_high_new",
         "paired_bootstrap_ci_high_old"),
        ("p_two_sided",
         "p_two_sided_new",
         "p_two_sided_old"),
        ("p_holm",
         "p_holm_two_sided_new",
         "p_holm_two_sided_old")
    ]

    report_rows = []
    for metric, new_col, old_col in metric_pairs:
        if new_col not in joined.columns or old_col not in joined.columns:
            report_rows.append({
                "metric": metric,
                "valid_comparisons": 0,
                "exact_matches": 0,
                "max_absolute_difference": np.nan,
                "mean_absolute_difference": np.nan,
                "status": "COLUMN_MISSING"
            })
            continue

        new_values = pd.to_numeric(joined[new_col], errors="coerce")
        old_values = pd.to_numeric(joined[old_col], errors="coerce")
        valid = new_values.notna() & old_values.notna()
        differences = (new_values[valid] - old_values[valid]).abs()

        exact = int(np.isclose(
            new_values[valid].to_numpy(),
            old_values[valid].to_numpy(),
            atol=TOLERANCE,
            rtol=TOLERANCE
        ).sum())

        report_rows.append({
            "metric": metric,
            "valid_comparisons": int(valid.sum()),
            "exact_matches": exact,
            "max_absolute_difference": (
                float(differences.max()) if len(differences) else np.nan
            ),
            "mean_absolute_difference": (
                float(differences.mean()) if len(differences) else np.nan
            ),
            "status": (
                "PASS" if exact == int(valid.sum()) else "REVIEW"
            )
        })

    difference_report = pd.DataFrame(report_rows)
    difference_report.to_csv(OUT_DIFF_REPORT, index=False)

else:
    record_check(
        "Experiment 063 comparison",
        False,
        "Prior file not found; recomputation completed without comparison."
    )

# ---------------------------
# 6. Summary and research record
# ---------------------------
direction_counts = (
    recomputed["bootstrap_direction"].value_counts().to_dict()
    if len(recomputed) else {}
)

summary_rows = []
if len(recomputed):
    for (exp_id, dimension, benchmark), group in raw.groupby(
        ["experiment", "dimension", "benchmark"]
    ):
        algorithm_summary = group.groupby("algorithm")["score"].agg(
            ["mean", "median", "std", "min", "max", "count"]
        ).reset_index()
        winner_idx = algorithm_summary["mean"].idxmin()
        winner = algorithm_summary.loc[winner_idx, "algorithm"]
        best_mean = float(algorithm_summary.loc[winner_idx, "mean"])

        summary_rows.append({
            "experiment": exp_id,
            "dimension": dimension,
            "benchmark": benchmark,
            "lowest_mean_score_algorithm": winner,
            "lowest_mean_score": best_mean,
            "algorithm_summary": algorithm_summary.to_dict(orient="records")
        })

check_df = pd.DataFrame(checks)
check_df.to_csv(OUT_CHECKS, index=False)

record = {
    "project": "Open Science Engine (OSE)",
    "experiment_id": "064",
    "title": "Statistical Consistency and Reproducibility Audit",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "input_files": [
        "experiment_059_results.csv",
        "experiment_060_results.csv",
        "experiment_063_corrected_comparisons.csv (optional)"
    ],
    "method": {
        "pairing": "Matched by experiment, dimension, benchmark, and seed",
        "primary_test": "Wilcoxon signed-rank, two-sided",
        "multiple_testing": (
            "Holm-Bonferroni within each experiment/dimension/benchmark "
            "comparison family"
        ),
        "confidence_interval": (
            "95% paired percentile bootstrap CI for mean score difference "
            "(algorithm A minus algorithm B)"
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "master_seed": MASTER_SEED,
        "note": (
            "Bootstrap seed generation in this version may differ from "
            "Experiment 063. Confidence intervals need not match exactly "
            "unless the original resampling seed and implementation match."
        )
    },
    "total_comparisons": int(len(recomputed)),
    "bootstrap_direction_counts": direction_counts,
    "group_summaries": summary_rows,
    "audit_checks": checks,
    "overall_structural_audit": (
        "PASS" if all(c["status"] == "PASS" for c in checks) else "REVIEW"
    ),
    "outputs": [
        os.path.basename(OUT_COMPARISONS),
        os.path.basename(OUT_CHECKS),
        os.path.basename(OUT_RECORD_JSON),
        os.path.basename(OUT_RECORD_MD),
        os.path.basename(OUT_MERGE),
        os.path.basename(OUT_DIFF_REPORT)
    ]
}

with open(OUT_RECORD_JSON, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, ensure_ascii=False, default=str)

# ---------------------------
# 7. Markdown report
# ---------------------------
lines = [
    "# OSE Experiment 064 — Statistical Consistency & Reproducibility Audit",
    "",
    f"- Created (UTC): {record['created_utc']}",
    f"- Comparisons recomputed: {len(recomputed)}",
    f"- Structural audit status: {record['overall_structural_audit']}",
    "",
    "## Method",
    "- Paired comparisons matched by seed.",
    "- Wilcoxon signed-rank test, two-sided.",
    "- Holm correction within each experiment/dimension/benchmark family.",
    "- Paired percentile bootstrap, 10,000 resamples.",
    "- Difference is mean(score A - score B); negative favours A.",
    "",
    "## Bootstrap direction counts",
]
for direction, count in direction_counts.items():
    lines.append(f"- {direction}: {count}")

lines += ["", "## Audit checks"]
for c in checks:
    lines.append(f"- **{c['status']}** — {c['check']}: {c['details']}")

if len(difference_report):
    lines += ["", "## Comparison against Experiment 063", ""]
    lines.append(difference_report.to_markdown(index=False))
    lines += [
        "",
        "Important: confidence interval differences may reflect different "
        "bootstrap random seeds or implementation settings. Do not label "
        "the intervals incorrect solely because they are not identical."
    ]

lines += [
    "",
    "## Interpretation limitations",
    "- Audit checks validate data structure and statistical recomputation, "
    "not the truth of a research hypothesis.",
    "- Benchmark findings apply only to the tested functions, dimensions, "
    "budgets, seeds, and parameter settings.",
    "- Exact bootstrap reproduction requires the same paired data, random "
    "seed, resampling implementation, and percentile settings."
]

with open(OUT_RECORD_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ---------------------------
# 8. Print final results
# ---------------------------
print("=" * 72)
print("OSE EXPERIMENT 064 — COMPLETED")
print("=" * 72)

print("\nRaw input rows:")
print(f"  Experiment 059: {len(df059)}")
print(f"  Experiment 060: {len(df060)}")

print(f"\nPairwise comparisons recomputed: {len(recomputed)}")
print("\nBootstrap direction counts:")
print(pd.Series(direction_counts).to_string())

print("\nAudit checks:")
print(check_df.to_string(index=False))

if len(difference_report):
    print("\nComparison against Experiment 063:")
    print(difference_report.to_string(index=False))

print("\nFiles saved:")
for path in [
    OUT_COMPARISONS,
    OUT_CHECKS,
    OUT_RECORD_JSON,
    OUT_RECORD_MD,
    OUT_MERGE,
    OUT_DIFF_REPORT
]:
    if os.path.exists(path):
        print(" -", path)

print("\nImportant:")
print(
    "This version uses its own deterministic bootstrap seed scheme. "
    "CI bounds may differ slightly from Experiment 063 even when paired "
    "means and Wilcoxon p-values match."
)
print(
    "A PASS indicates the specified audit checks passed; it does not "
    "establish universal superiority of any optimizer."
)



# ===== Original notebook cell 191 (code) =====

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = "/content/OSE/results"

old = pd.read_csv(
    os.path.join(R, "experiment_063_corrected_comparisons.csv")
)
new = pd.read_csv(
    os.path.join(R, "experiment_064_recomputed_metrics.csv")
)

def norm_exp(x):
    s = str(x).strip().lower().replace("experiment_", "")
    return s.zfill(3) if s.isdigit() else s

for df in [old, new]:
    df["experiment"] = df["experiment"].map(norm_exp)
    df["dimension"] = pd.to_numeric(df["dimension"])
    for c in ["benchmark", "algorithm_a", "algorithm_b"]:
        df[c] = df[c].astype(str).str.strip()

keys = [
    "experiment", "dimension", "benchmark",
    "algorithm_a", "algorithm_b"
]

old = old.rename(columns={
    "p_two_sided": "p_two_old",
    "p_holm_two_sided": "holm_old"
})
new = new.rename(columns={
    "p_two_sided": "p_two_new",
    "p_holm_two_sided": "holm_new"
})

m = new.merge(
    old[keys + ["p_two_old", "holm_old"]],
    on=keys, how="inner", validate="one_to_one"
)

def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    order = valid[np.argsort(p[valid])]
    adjusted = np.maximum.accumulate([
        (len(order) - i) * p[j]
        for i, j in enumerate(order)
    ])
    for j, v in zip(order, np.minimum(adjusted, 1.0)):
        out[j] = v
    return out

scopes = {
    "experiment_only": ["experiment"],
    "experiment_dimension": ["experiment", "dimension"],
    "experiment_dimension_benchmark": [
        "experiment", "dimension", "benchmark"
    ],
    "all_comparisons": []
}

print("Matched comparisons:", len(m))
print("\nHolm correction scope diagnostic")

for name, cols in scopes.items():
    trial = m.copy()
    trial["holm_trial"] = np.nan

    if cols:
        groups = trial.groupby(cols, dropna=False).groups.values()
    else:
        groups = [trial.index]

    for idx in groups:
        trial.loc[idx, "holm_trial"] = holm(
            trial.loc[idx, "p_two_new"].to_numpy()
        )

    valid = trial["holm_old"].notna() & trial["holm_trial"].notna()
    dif = (
        trial.loc[valid, "holm_trial"] -
        trial.loc[valid, "holm_old"]
    ).abs()

    matches = np.isclose(
        trial.loc[valid, "holm_trial"],
        trial.loc[valid, "holm_old"],
        atol=1e-10, rtol=1e-10
    ).sum()

    print(
        f"{name}: {int(matches)}/{int(valid.sum())} matches; "
        f"max difference={dif.max() if len(dif) else np.nan:.6g}; "
        f"mean difference={dif.mean() if len(dif) else np.nan:.6g}"
    )



# ===== Original notebook cell 192 (code) =====

# ================================================================
# OSE EXPERIMENT 064 — BOOTSTRAP CONFIDENCE INTERVAL AUDIT
# Read-only analysis of Experiments 063 and 064
# Does not overwrite existing experiment results.
# ================================================================

import os
import numpy as np
import pandas as pd

R = "/content/OSE/results"

OLD_FILE = os.path.join(
    R, "experiment_063_corrected_comparisons.csv"
)
NEW_FILE = os.path.join(
    R, "experiment_064_recomputed_metrics.csv"
)
OUT_FILE = os.path.join(
    R, "experiment_064_bootstrap_diagnostic.csv"
)

if not os.path.exists(OLD_FILE):
    raise FileNotFoundError(f"Missing file: {OLD_FILE}")

if not os.path.exists(NEW_FILE):
    raise FileNotFoundError(f"Missing file: {NEW_FILE}")

old = pd.read_csv(OLD_FILE)
new = pd.read_csv(NEW_FILE)

def normalize_id(value):
    s = str(value).strip().lower().replace("experiment_", "")
    return s.zfill(3) if s.isdigit() else s

for df in [old, new]:
    df["experiment"] = df["experiment"].map(normalize_id)
    df["dimension"] = pd.to_numeric(df["dimension"], errors="coerce")
    for c in ["benchmark", "algorithm_a", "algorithm_b"]:
        df[c] = df[c].astype(str).str.strip()

keys = [
    "experiment", "dimension", "benchmark",
    "algorithm_a", "algorithm_b"
]

old = old.rename(columns={
    "paired_bootstrap_ci_low": "ci_low_old",
    "paired_bootstrap_ci_high": "ci_high_old"
})

new = new.rename(columns={
    "paired_bootstrap_ci_low": "ci_low_new",
    "paired_bootstrap_ci_high": "ci_high_new"
})

merged = new.merge(
    old[keys + ["ci_low_old", "ci_high_old"]],
    on=keys,
    how="inner",
    validate="one_to_one"
)

if len(merged) != 64:
    print(f"WARNING: Expected 64 matched comparisons; found {len(merged)}")

# Compare interval bounds and widths.
merged["low_abs_difference"] = (
    merged["ci_low_new"] - merged["ci_low_old"]
).abs()

merged["high_abs_difference"] = (
    merged["ci_high_new"] - merged["ci_high_old"]
).abs()

merged["width_old"] = (
    merged["ci_high_old"] - merged["ci_low_old"]
)

merged["width_new"] = (
    merged["ci_high_new"] - merged["ci_low_new"]
)

merged["width_abs_difference"] = (
    merged["width_new"] - merged["width_old"]
).abs()

# Check whether each interval includes zero.
merged["old_includes_zero"] = (
    (merged["ci_low_old"] <= 0) &
    (merged["ci_high_old"] >= 0)
)

merged["new_includes_zero"] = (
    (merged["ci_low_new"] <= 0) &
    (merged["ci_high_new"] >= 0)
)

merged["zero_decision_matches"] = (
    merged["old_includes_zero"] ==
    merged["new_includes_zero"]
)

# Check whether the two intervals overlap.
merged["intervals_overlap"] = (
    np.maximum(merged["ci_low_old"], merged["ci_low_new"])
    <=
    np.minimum(merged["ci_high_old"], merged["ci_high_new"])
)

# Save detailed diagnostic without changing old files.
merged.to_csv(OUT_FILE, index=False)

print("=" * 68)
print("OSE EXPERIMENT 064 — BOOTSTRAP AUDIT")
print("=" * 68)

print(f"\nMatched comparisons: {len(merged)}")

print("\nLower-bound absolute difference:")
print(f"  Mean: {merged['low_abs_difference'].mean():.6f}")
print(f"  Max:  {merged['low_abs_difference'].max():.6f}")

print("\nUpper-bound absolute difference:")
print(f"  Mean: {merged['high_abs_difference'].mean():.6f}")
print(f"  Max:  {merged['high_abs_difference'].max():.6f}")

print("\nInterval width difference:")
print(f"  Mean absolute difference: {merged['width_abs_difference'].mean():.6f}")
print(f"  Max absolute difference:  {merged['width_abs_difference'].max():.6f}")

print("\nInterval overlap:")
print(
    f"  {int(merged['intervals_overlap'].sum())}"
    f"/{len(merged)} pairs have overlapping intervals"
)

print("\nZero-inclusion decision:")
print(
    f"  {int(merged['zero_decision_matches'].sum())}"
    f"/{len(merged)} pairs agree on whether the interval includes zero"
)

print("\nExamples with largest lower-bound differences:")
cols = [
    "experiment", "dimension", "benchmark",
    "algorithm_a", "algorithm_b",
    "ci_low_old", "ci_low_new",
    "ci_high_old", "ci_high_new",
    "intervals_overlap", "zero_decision_matches"
]

print(
    merged.sort_values("low_abs_difference", ascending=False)
    [cols].head(10).to_string(index=False)
)

print("\nSaved diagnostic:")
print(OUT_FILE)

print("\nInterpretation:")
print(
    "Different intervals can arise from different bootstrap seeds or "
    "resampling settings. Overlap alone does not prove the methods "
    "are identical. Exact reproduction requires the original Experiment "
    "063 bootstrap implementation and seed-generation rule."
)

