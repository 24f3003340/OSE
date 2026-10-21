# OSE Experiment 065
# Extracted from the original notebook.
# Review cell order and dependencies before running as a script.


# ===== Original notebook cell 193 (markdown) =====
Experiment 065


# ===== Original notebook cell 194 (code) =====

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

os.chdir("/content/OSE")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Load original raw experimental results
# --------------------------------------------------
raw_files = {
    "059": os.path.join(RESULTS_DIR, "experiment_059_results.csv"),
    "060": os.path.join(RESULTS_DIR, "experiment_060_results.csv"),
}

frames = []

for exp_id, path in raw_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required raw data missing: {path}. "
            "Run this experiment in the OSE Colab workspace first."
        )

    df = pd.read_csv(path)
    df["source_experiment"] = exp_id
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)

print("Raw data loaded:", len(raw), "rows")
print("Columns:", raw.columns.tolist())

# --------------------------------------------------
# 2. Identify columns without assuming exact naming
# --------------------------------------------------
def find_col(df, candidates, required=True):
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(
            f"Could not find any of {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return None

algorithm_col = find_col(
    raw, ["algorithm", "Algorithm", "method", "optimizer"]
)
benchmark_col = find_col(
    raw, ["benchmark", "function", "objective", "problem"]
)
dimension_col = find_col(
    raw, ["dimension", "dimensions", "dim", "n_dimensions"]
)
score_col = find_col(
    raw, ["best_value", "best_score", "best_objective",
          "final_score", "score", "best_fitness"]
)
seed_col = find_col(raw, ["seed", "random_seed"])
success_col = find_col(
    raw, ["success", "target_reached", "reached_target"],
    required=False
)
runtime_col = find_col(
    raw, ["runtime_seconds", "runtime", "wall_clock_seconds",
          "elapsed_seconds"],
    required=False
)
eval_col = find_col(
    raw, ["evaluations", "n_evaluations", "function_evaluations",
          "evals"],
    required=False
)

# Normalize key columns for consistent grouping.
raw["_algorithm"] = raw[algorithm_col].astype(str)
raw["_benchmark"] = raw[benchmark_col].astype(str)
raw["_dimension"] = pd.to_numeric(raw[dimension_col], errors="coerce")
raw["_score"] = pd.to_numeric(raw[score_col], errors="coerce")
raw["_seed"] = raw[seed_col].astype(str)

if raw["_score"].isna().any() or raw["_dimension"].isna().any():
    raise ValueError("Missing or non-numeric score/dimension values found.")

# --------------------------------------------------
# 3. Build benchmark leaderboard
# Lower score is better for these minimization tasks.
# --------------------------------------------------
group_cols = [
    "source_experiment", "_benchmark", "_dimension", "_algorithm"
]

agg_spec = {
    "runs": ("_score", "size"),
    "mean_score": ("_score", "mean"),
    "median_score": ("_score", "median"),
    "std_score": ("_score", "std"),
    "best_observed_score": ("_score", "min"),
    "worst_observed_score": ("_score", "max"),
}

if success_col:
    raw["_success_numeric"] = (
        raw[success_col].astype(str).str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )
    if raw["_success_numeric"].notna().all():
        agg_spec["success_rate_pct"] = (
            "_success_numeric", lambda x: 100 * x.mean()
        )

if runtime_col:
    raw["_runtime"] = pd.to_numeric(raw[runtime_col], errors="coerce")
    agg_spec["median_runtime_seconds"] = ("_runtime", "median")

if eval_col:
    raw["_evaluations"] = pd.to_numeric(raw[eval_col], errors="coerce")
    agg_spec["median_evaluations"] = ("_evaluations", "median")

leaderboard = (
    raw.groupby(group_cols, dropna=False)
       .agg(**agg_spec)
       .reset_index()
       .rename(columns={
           "_benchmark": "benchmark",
           "_dimension": "dimension",
           "_algorithm": "algorithm",
       })
)

leaderboard["rank_by_mean"] = (
    leaderboard.groupby(
        ["source_experiment", "benchmark", "dimension"]
    )["mean_score"]
    .rank(method="min", ascending=True)
    .astype(int)
)

leaderboard = leaderboard.sort_values(
    ["source_experiment", "benchmark", "dimension", "rank_by_mean"]
)

leaderboard_path = os.path.join(
    RESULTS_DIR, "experiment_065_benchmark_leaderboard.csv"
)
leaderboard.to_csv(leaderboard_path, index=False)

# --------------------------------------------------
# 4. Load statistical evidence from Experiments 063/064
# --------------------------------------------------
stat_paths = [
    os.path.join(RESULTS_DIR, "experiment_064_recomputed_metrics.csv"),
    os.path.join(RESULTS_DIR, "experiment_063_corrected_comparisons.csv"),
    os.path.join(RESULTS_DIR, "experiment_062_statistical_comparisons.csv"),
]

stat_source = None
stat_df = None

for path in stat_paths:
    if os.path.exists(path):
        candidate = pd.read_csv(path)
        if len(candidate) > 0:
            stat_source = path
            stat_df = candidate
            break

if stat_df is not None:
    stat_path = os.path.join(
        RESULTS_DIR, "experiment_065_statistical_evidence.csv"
    )
    stat_df.to_csv(stat_path, index=False)
    stat_status = (
        f"Loaded {len(stat_df)} comparison rows from "
        f"{os.path.basename(stat_source)}"
    )
else:
    stat_path = None
    stat_status = (
        "No prior statistical comparison file found; "
        "leaderboard is descriptive only."
    )

# --------------------------------------------------
# 5. Data integrity checks
# --------------------------------------------------
checks = []

def record_check(name, passed, detail):
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "detail": str(detail),
    })

record_check(
    "Raw input files loaded",
    len(raw) > 0,
    f"{len(raw)} rows from Experiments 059 and 060"
)

record_check(
    "Scores finite",
    bool(np.isfinite(raw["_score"]).all()),
    f"Finite score values: {np.isfinite(raw['_score']).sum()}/{len(raw)}"
)

record_check(
    "Leaderboard counts reconcile",
    int(leaderboard["runs"].sum()) == len(raw),
    f"Leaderboard runs={int(leaderboard['runs'].sum())}; raw rows={len(raw)}"
)

record_check(
    "Leaderboard ranks valid",
    bool((leaderboard["rank_by_mean"] >= 1).all()),
    "Ranks are calculated within each experiment/benchmark/dimension group"
)

checks_df = pd.DataFrame(checks)
checks_path = os.path.join(
    RESULTS_DIR, "experiment_065_audit_checks.csv"
)
checks_df.to_csv(checks_path, index=False)

# --------------------------------------------------
# 6. Create readable research report
# --------------------------------------------------
lines = [
    "# Experiment 065 — Research Evidence Summary",
    "",
    f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
    "",
    "## Objective",
    "Consolidate existing benchmark results and create a descriptive "
    "leaderboard without modifying the original raw records.",
    "",
    "## Data sources",
    "- experiment_059_results.csv",
    "- experiment_060_results.csv",
    "",
    f"- Statistical evidence: {stat_status}",
    "",
    "## Interpretation rules",
    "- Lower objective score is better for these minimization benchmarks.",
    "- Rank 1 means lowest observed mean score in that group.",
    "- Ranking by mean alone does not establish statistical superiority.",
    "- Results apply only to the tested functions, dimensions, budgets, "
    "seeds and algorithm settings.",
    "",
    "## Leaderboard",
    "",
]

for (exp_id, benchmark, dimension), group in leaderboard.groupby(
    ["source_experiment", "benchmark", "dimension"]
):
    lines.append(
        f"### Experiment {exp_id} — {benchmark}, {int(dimension)}D"
    )
    lines.append("")
    lines.append(
        "| Rank | Algorithm | Runs | Mean | Median | Std. dev. |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|")

    for _, row in group.sort_values("rank_by_mean").iterrows():
        std_value = row["std_score"]
        std_text = "NA" if pd.isna(std_value) else f"{std_value:.6g}"
        lines.append(
            f"| {row['rank_by_mean']} | {row['algorithm']} | "
            f"{int(row['runs'])} | {row['mean_score']:.6g} | "
            f"{row['median_score']:.6g} | {std_text} |"
        )
    lines.append("")

lines.extend([
    "## Audit checks",
    "",
    "| Check | Status | Detail |",
    "|---|---|---|",
])

for _, row in checks_df.iterrows():
    lines.append(
        f"| {row['check']} | {row['status']} | {row['detail']} |"
    )

lines.extend([
    "",
    "## Limitations",
    "",
    "This report summarizes existing data; it does not run new optimizer "
    "trials. Descriptive ranks do not prove that one algorithm is "
    "universally superior. Statistical comparison files are carried forward "
    "as evidence, not independently recalculated here. Exact bootstrap "
    "confidence intervals in earlier audits were not perfectly reproduced; "
    "their zero-inclusion decisions matched, but that is not proof of "
    "identical resampling.",
    "",
])

report_path = os.path.join(
    RESULTS_DIR, "experiment_065_research_report.md"
)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# --------------------------------------------------
# 7. Machine-readable research record
# --------------------------------------------------
record = {
    "experiment_id": "065",
    "title": "Research Evidence Summary and Benchmark Leaderboard",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Consolidate prior benchmark results and audit basic integrity.",
    "raw_sources": list(raw_files.values()),
    "raw_rows": int(len(raw)),
    "leaderboard_rows": int(len(leaderboard)),
    "statistical_evidence_source": stat_source,
    "statistical_evidence_status": stat_status,
    "audit_passed": bool((checks_df["status"] == "PASS").all()),
    "audit_checks": checks,
    "limitations": [
        "No new optimizer trials were run.",
        "Mean-score ranks are descriptive, not universal claims.",
        "Statistical evidence was carried forward, not recalculated.",
        "Exact bootstrap CI reproduction remains under review."
    ],
    "outputs": [
        leaderboard_path,
        checks_path,
        report_path,
        os.path.join(RESULTS_DIR, "experiment_065_research_record.json"),
    ] + ([stat_path] if stat_path else []),
}

record_path = os.path.join(
    RESULTS_DIR, "experiment_065_research_record.json"
)
with open(record_path, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, ensure_ascii=False)

print("\n========== EXPERIMENT 065 COMPLETE ==========")
print("Raw rows:", len(raw))
print("Leaderboard rows:", len(leaderboard))
print("\nAudit checks:")
print(checks_df.to_string(index=False))
print("\nLeaderboard preview:")
print(leaderboard.head(20).to_string(index=False))
print("\nStatistical evidence:", stat_status)
print("\nFiles created:")
for path in record["outputs"]:
    print("-", path)

print("\nImportant: leaderboard is descriptive; it does not prove a universal winner.")

