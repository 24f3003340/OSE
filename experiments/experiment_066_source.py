# OSE Experiment 066
# Extracted from the original notebook.
# Review cell order and dependencies before running as a script.


# ===== Original notebook cell 195 (markdown) =====
Experiment 066


# ===== Original notebook cell 196 (code) =====

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

os.chdir("/content/OSE")
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# --------------------------------------------------
# 1. Load Experiment 065 outputs
# --------------------------------------------------
leaderboard_path = os.path.join(
    RESULTS, "experiment_065_benchmark_leaderboard.csv"
)
record_path = os.path.join(
    RESULTS, "experiment_065_research_record.json"
)

if not os.path.exists(leaderboard_path):
    raise FileNotFoundError(
        f"Missing {leaderboard_path}. Run Experiment 065 first."
    )

leaderboard = pd.read_csv(leaderboard_path)

required = {
    "source_experiment", "benchmark", "dimension", "algorithm",
    "runs", "mean_score", "median_score", "std_score",
    "rank_by_mean"
}
missing = required - set(leaderboard.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# --------------------------------------------------
# 2. Validate leaderboard values
# --------------------------------------------------
checks = []

def add_check(name, passed, detail):
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "detail": str(detail)
    })

numeric_cols = [
    "dimension", "runs", "mean_score",
    "median_score", "std_score", "rank_by_mean"
]

numeric_ok = True
for col in numeric_cols:
    leaderboard[col] = pd.to_numeric(
        leaderboard[col], errors="coerce"
    )
    if leaderboard[col].isna().any():
        numeric_ok = False

add_check(
    "Required columns and numeric fields",
    numeric_ok,
    "All required fields present and numeric"
)

finite_ok = bool(
    np.isfinite(
        leaderboard[
            ["mean_score", "median_score", "rank_by_mean"]
        ].to_numpy(dtype=float)
    ).all()
)
add_check(
    "Finite leaderboard metrics",
    finite_ok,
    "Mean, median and ranks contain finite values"
)

positive_runs = bool((leaderboard["runs"] > 0).all())
add_check(
    "Positive run counts",
    positive_runs,
    f"Minimum runs per algorithm/group: {leaderboard['runs'].min()}"
)

nonnegative_std = bool(
    (leaderboard["std_score"].dropna() >= 0).all()
)
add_check(
    "Standard deviation valid",
    nonnegative_std,
    "Standard deviations are non-negative where available"
)

# Confirm ranks exactly match the ranking by mean within each group.
rank_group = [
    "source_experiment", "benchmark", "dimension"
]
expected_rank = (
    leaderboard.groupby(rank_group)["mean_score"]
    .rank(method="min", ascending=True)
)
rank_ok = bool(
    np.allclose(
        leaderboard["rank_by_mean"].to_numpy(dtype=float),
        expected_rank.to_numpy(dtype=float)
    )
)
add_check(
    "Ranking consistency",
    rank_ok,
    "Saved ranks match ranks recalculated from mean scores"
)

# --------------------------------------------------
# 3. Create best-mean summary for dashboard
# --------------------------------------------------
best_rows = (
    leaderboard.sort_values("mean_score")
    .groupby(rank_group, as_index=False)
    .first()
    .rename(columns={
        "algorithm": "lowest_mean_algorithm",
        "mean_score": "lowest_mean_score",
        "median_score": "median_score_of_lowest_mean",
        "std_score": "std_score_of_lowest_mean",
        "rank_by_mean": "rank"
    })
)

# Optional metrics can be included if available.
for optional in [
    "success_rate_pct", "median_runtime_seconds",
    "median_evaluations", "best_observed_score",
    "worst_observed_score"
]:
    if optional in leaderboard.columns:
        # These remain in the full leaderboard CSV.
        pass

best_path = os.path.join(
    RESULTS, "experiment_066_best_mean_summary.csv"
)
best_rows.to_csv(best_path, index=False)

# --------------------------------------------------
# 4. Dashboard JSON: grouped results and checks
# --------------------------------------------------
dashboard_groups = []

for keys, group in leaderboard.groupby(rank_group, dropna=False):
    exp_id, benchmark, dimension = keys
    algorithms = []

    for _, row in group.sort_values("rank_by_mean").iterrows():
        item = {
            "algorithm": str(row["algorithm"]),
            "runs": int(row["runs"]),
            "mean_score": float(row["mean_score"]),
            "median_score": float(row["median_score"]),
            "std_score": (
                None if pd.isna(row["std_score"])
                else float(row["std_score"])
            ),
            "rank_by_mean": int(row["rank_by_mean"])
        }

        for col in [
            "success_rate_pct", "median_runtime_seconds",
            "median_evaluations"
        ]:
            if col in group.columns and pd.notna(row.get(col)):
                item[col] = float(row[col])

        algorithms.append(item)

    dashboard_groups.append({
        "source_experiment": str(exp_id),
        "benchmark": str(benchmark),
        "dimension": int(dimension),
        "ranking_rule": "Lowest mean objective score ranks first",
        "algorithms": algorithms
    })

checks_df = pd.DataFrame(checks)
checks_path = os.path.join(
    RESULTS, "experiment_066_validation_checks.csv"
)
checks_df.to_csv(checks_path, index=False)

dashboard = {
    "experiment_id": "066",
    "title": "OSE Research Evidence Dashboard Data",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source": "experiment_065_benchmark_leaderboard.csv",
    "groups_count": len(dashboard_groups),
    "leaderboard_rows": len(leaderboard),
    "validation_passed": bool(
        (checks_df["status"] == "PASS").all()
    ),
    "validation_checks": checks,
    "groups": dashboard_groups,
    "interpretation": [
        "Lower mean score is better for these minimization benchmarks.",
        "Ranks are descriptive and apply only to the reported settings.",
        "Lowest mean does not automatically imply statistical superiority.",
        "This dashboard data summarizes prior runs; no new optimizer trials ran."
    ]
}

dashboard_path = os.path.join(
    RESULTS, "experiment_066_dashboard_data.json"
)
with open(dashboard_path, "w", encoding="utf-8") as f:
    json.dump(dashboard, f, indent=2, ensure_ascii=False)

# --------------------------------------------------
# 5. Create research record and Markdown report
# --------------------------------------------------
record = {
    "experiment_id": "066",
    "title": "Research Report Validation and Evidence Dashboard",
    "generated_utc": dashboard["generated_utc"],
    "input": leaderboard_path,
    "rows_checked": int(len(leaderboard)),
    "groups_count": int(len(dashboard_groups)),
    "validation_passed": dashboard["validation_passed"],
    "checks": checks,
    "outputs": [
        best_path, checks_path, dashboard_path,
        os.path.join(RESULTS, "experiment_066_research_report.md"),
        os.path.join(RESULTS, "experiment_066_research_record.json")
    ]
}

record_out = os.path.join(
    RESULTS, "experiment_066_research_record.json"
)
with open(record_out, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, ensure_ascii=False)

report_lines = [
    "# Experiment 066 — Research Report Validation",
    "",
    f"Generated UTC: {dashboard['generated_utc']}",
    "",
    f"- Leaderboard rows checked: {len(leaderboard)}",
    f"- Experiment/benchmark/dimension groups: {len(dashboard_groups)}",
    f"- Overall validation: {'PASS' if dashboard['validation_passed'] else 'FAIL'}",
    "",
    "## Validation checks",
    "",
    "| Check | Status | Detail |",
    "|---|---|---|"
]

for _, row in checks_df.iterrows():
    report_lines.append(
        f"| {row['check']} | {row['status']} | {row['detail']} |"
    )

report_lines.extend([
    "",
    "## Lowest mean score per group",
    "",
    "| Experiment | Benchmark | Dimension | Algorithm | Mean score |",
    "|---|---|---:|---|---:|"
])

for _, row in best_rows.iterrows():
    report_lines.append(
        f"| {row['source_experiment']} | {row['benchmark']} | "
        f"{int(row['dimension'])} | {row['lowest_mean_algorithm']} | "
        f"{row['lowest_mean_score']:.6g} |"
    )

report_lines.extend([
    "",
    "## Limitations",
    "",
    "This validation checks the integrity and internal consistency of the "
    "existing leaderboard. It does not independently rerun all 3,600 raw "
    "trials, verify every statistical test, or establish a universal winner. "
    "A future production dashboard should link every summary to its raw data "
    "and experiment configuration.",
    ""
])

report_path = os.path.join(
    RESULTS, "experiment_066_research_report.md"
)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# --------------------------------------------------
# 6. Final output
# --------------------------------------------------
print("========== EXPERIMENT 066 COMPLETE ==========")
print("Leaderboard rows:", len(leaderboard))
print("Groups:", len(dashboard_groups))
print("\nValidation checks:")
print(checks_df.to_string(index=False))

print("\nLowest mean score per group:")
print(best_rows[
    ["source_experiment", "benchmark", "dimension",
     "lowest_mean_algorithm", "lowest_mean_score"]
].to_string(index=False))

print("\nFiles created:")
for path in record["outputs"]:
    print("-", path)

if not dashboard["validation_passed"]:
    print("\nWARNING: One or more validation checks failed.")
else:
    print("\nAll validation checks passed.")



# ===== Original notebook cell 197 (code) =====

import os
import zipfile

source = "/content/OSE/results"
output = "/content/ose_dashboard_results.zip"

files = [
    "experiment_065_benchmark_leaderboard.csv",
    "experiment_065_statistical_evidence.csv",
    "experiment_066_dashboard_data.json",
    "experiment_059_results.csv",
    "experiment_060_results.csv",
]

with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
    for name in files:
        path = os.path.join(source, name)
        if os.path.exists(path):
            z.write(path, arcname=f"results/{name}")
            print("Added:", name)
        else:
            print("Optional/missing:", name)

print("ZIP ready:", output)



# ===== Original notebook cell 198 (code) =====
import json
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

notebook_path = Path(
    "/content/drive/MyDrive/Colab Notebooks/"
    "OSE_Day01_Research_Foundation.ipynb"
)

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell_index in [194, 196, 197]:
    cell = notebook["cells"][cell_index]
    print(f"\n{'=' * 60}")
    print("CELL:", cell_index)
    print("OUTPUTS:")

    for output in cell.get("outputs", []):
        if "text" in output:
            print("".join(output["text"]))
        elif "data" in output:
            text_data = output["data"].get("text/plain")
            if text_data:
                print("".join(text_data))


# ===== Original notebook cell 199 (code) =====
from pathlib import Path
import json

drive_dir = Path("/content/drive/MyDrive/Colab Notebooks")
notebook_path = drive_dir / "OSE_Day01_Research_Foundation.ipynb"

if not notebook_path.exists():
    print("Notebook not found at:", notebook_path)
    print("Available OSE notebooks:")
    for p in drive_dir.glob("*OSE*.ipynb"):
        print(p)
else:
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    sections = [
        "OSE EXPERIMENTS 065–066 SAVED OUTPUTS BACKUP",
        "Original notebook: " + notebook_path.name,
        "Saved outputs only; missing raw data is not reconstructed."
    ]

    for idx in [194, 196, 197]:
        sections += ["\n" + "=" * 60, f"CELL {idx}"]
        for out in notebook["cells"][idx].get("outputs", []):
            if "text" in out:
                sections.append("".join(out["text"]))
            elif "data" in out and "text/plain" in out["data"]:
                sections.append("".join(out["data"]["text/plain"]))

    backup_path = drive_dir / "OSE_065_066_SAVED_OUTPUTS_BACKUP.txt"
    backup_path.write_text("\n".join(sections), encoding="utf-8")

    print("Backup saved:", backup_path)
    print("Size:", backup_path.stat().st_size, "bytes")


# ===== Original notebook cell 200 (code) =====
from pathlib import Path

root = Path("/content/drive/MyDrive")

targets = {
    "experiment_059_results.csv",
    "experiment_060_results.csv",
    "experiment_065_benchmark_leaderboard.csv",
    "experiment_065_statistical_evidence.csv",
    "experiment_066_dashboard_data.json",
    "ose_dashboard_results.zip",
}

found = []

for path in root.rglob("*"):
    if path.is_file() and path.name.lower() in targets:
        found.append(path)
        print("FOUND:", path, "|", path.stat().st_size, "bytes")

if not found:
    print("No matching result files found in Google Drive.")


# ===== Original notebook cell 201 (code) =====
import json
from pathlib import Path

notebook_path = Path(
    "/content/drive/MyDrive/Colab Notebooks/"
    "OSE_Day01_Research_Foundation.ipynb"
)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Total cells:", len(nb["cells"]))

for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    output_text = []

    for out in cell.get("outputs", []):
        if "text" in out:
            output_text.append("".join(out["text"]))
        elif "data" in out and "text/plain" in out["data"]:
            output_text.append("".join(out["data"]["text/plain"]))

    output = "\n".join(output_text)
    combined = (source + "\n" + output).lower()

    if any(term in combined for term in [
        "experiment_059_results.csv",
        "experiment_060_results.csv",
        "raw data loaded: 3600 rows",
        "experiment 059 complete",
        "experiment 060 complete"
    ]):
        print(
            f"Cell {i} | "
            f"source chars: {len(source)} | "
            f"output chars: {len(output)}"
        )
        print("Source preview:", source[:180].replace("\n", " "))
        print("Output preview:", output[:180].replace("\n", " "))
        print("-" * 50)


# ===== Original notebook cell 202 (code) =====
import json
from pathlib import Path

notebook_path = Path(
    "/content/drive/MyDrive/Colab Notebooks/"
    "OSE_Day01_Research_Foundation.ipynb"
)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

keywords = [
    "to_csv", "to_json", "to_pickle",
    "write_text", "write_bytes", "open(",
    "read_csv", "ROOT =", "OUTPUT", "RESULTS_DIR",
    "experiment_059_results", "experiment_060_results"
]

for idx in [175, 177]:
    cell = nb["cells"][idx]
    source = "".join(cell.get("source", []))

    print(f"\n{'=' * 55}")
    print(f"CELL {idx}: FILE OPERATIONS AND PATHS")

    for line_no, line in enumerate(source.splitlines(), start=1):
        if any(word.lower() in line.lower() for word in keywords):
            print(f"{line_no}: {line.strip()[:300]}")

    print("\nSAVED OUTPUT — LAST 25 LINES:")
    outputs = []
    for out in cell.get("outputs", []):
        if "text" in out:
            outputs.append("".join(out["text"]))
        elif "data" in out and "text/plain" in out["data"]:
            outputs.append("".join(out["data"]["text/plain"]))

    output = "\n".join(outputs)
    print("\n".join(output.splitlines()[-25:]))


# ===== Original notebook cell 203 (code) =====

from pathlib import Path
import shutil
import zipfile

root = Path("/content/OSE")
drive = Path("/content/drive/MyDrive/Colab Notebooks")
backup = drive / "OSE_RECOVERY_BACKUP"

targets = [
    "experiment_059_results.csv",
    "experiment_060_results.csv",
    "experiment_059_history.csv",
    "experiment_060_history.csv",
    "experiment_059_summary.csv",
    "experiment_060_summary.csv",
    "experiment_065_benchmark_leaderboard.csv",
    "experiment_065_statistical_evidence.csv",
    "experiment_066_dashboard_data.json",
]

found = []

print("OSE folder exists:", root.exists())

for name in targets:
    matches = list(root.rglob(name)) if root.exists() else []

    if matches:
        for src in matches:
            print("FOUND:", src, "|", src.stat().st_size, "bytes")
            found.append(src)
    else:
        print("MISSING:", name)

if found:
    backup.mkdir(parents=True, exist_ok=True)

    for src in found:
        shutil.copy2(src, backup / src.name)

    zip_path = drive / "OSE_RECOVERED_RESULTS.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for src in found:
            z.write(src, arcname=src.name)

    print("\nBackup folder:", backup)
    print("ZIP backup:", zip_path)
    print("Files backed up:", len(found))
else:
    print("\nNo result files found in the current runtime.")
    print("No experiments were rerun.")



# ===== Original notebook cell 204 (code) =====
from pathlib import Path
import shutil

drive = Path("/content/drive/MyDrive/Colab Notebooks")
stage = drive / "OSE_GITHUB_INITIAL_BACKUP"
stage.mkdir(parents=True, exist_ok=True)

files = [
    "OSE_Day01_Research_Foundation.ipynb",
    "OSE_Day01_Research_Foundation_BACKUP.ipynb",
    "OSE_065_066_SAVED_OUTPUTS_BACKUP.txt",
]

for name in files:
    src = drive / name
    if src.exists():
        dst = stage / name
        shutil.copy2(src, dst)
        print("STAGED:", dst.name, "|", dst.stat().st_size, "bytes")
    else:
        print("NOT FOUND:", name)

print("\nStaging folder:", stage)
