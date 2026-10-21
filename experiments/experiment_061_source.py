# OSE Experiment 061
# Extracted from the original notebook.
# Review cell order and dependencies before running as a script.


# ===== Original notebook cell 178 (markdown) =====
Experiment 061


# ===== Original notebook cell 179 (code) =====

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

ROOT = Path("/content/OSE")
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = ["059", "060"]
audit_rows = []
summary_rows = []
pairing_rows = []
start_time = time.perf_counter()

required_columns = {
    "dimension", "benchmark", "algorithm",
    "seed", "evaluations", "best_score",
    "target", "success", "runtime_seconds"
}

for exp_id in EXPERIMENTS:
    path = OUT / f"experiment_{exp_id}_results.csv"

    if not path.exists():
        audit_rows.append({
            "experiment": exp_id,
            "status": "MISSING_FILE",
            "rows": 0,
            "checks_passed": 0,
            "checks_failed": 1,
            "notes": f"File not found: {path}"
        })
        print(f"Skipping Experiment {exp_id}: file not found")
        continue

    df = pd.read_csv(path)
    missing_cols = required_columns - set(df.columns)

    if missing_cols:
        audit_rows.append({
            "experiment": exp_id,
            "status": "MISSING_COLUMNS",
            "rows": len(df),
            "checks_passed": 0,
            "checks_failed": len(missing_cols),
            "notes": ", ".join(sorted(missing_cols))
        })
        print(f"Experiment {exp_id}: missing columns {missing_cols}")
        continue

    checks = {}

    checks["nonempty"] = len(df) > 0
    checks["finite_scores"] = bool(
        np.isfinite(pd.to_numeric(
            df["best_score"], errors="coerce"
        )).all()
    )
    checks["finite_runtime"] = bool(
        np.isfinite(pd.to_numeric(
            df["runtime_seconds"], errors="coerce"
        )).all()
    )
    checks["nonnegative_runtime"] = bool(
        (df["runtime_seconds"] >= 0).all()
    )
    checks["positive_evaluations"] = bool(
        (df["evaluations"] > 0).all()
    )

    run_key = [
        "dimension", "benchmark", "algorithm", "seed"
    ]
    checks["unique_run_keys"] = not df.duplicated(
        run_key
    ).any()

    checks["success_matches_target"] = bool(
        (
            df["success"].astype(bool).to_numpy()
            ==
            (
                df["best_score"].to_numpy()
                <= df["target"].to_numpy()
            )
        ).all()
    )

    # Compare actual evaluation counts with the declared budget
    if "budget" in df.columns:
        checks["evaluations_match_budget"] = bool(
            (df["evaluations"] == df["budget"]).all()
        )

    passed = sum(checks.values())
    failed = len(checks) - passed

    audit_rows.append({
        "experiment": exp_id,
        "status": "PASS" if failed == 0 else "REVIEW",
        "rows": len(df),
        "checks_passed": passed,
        "checks_failed": failed,
        "notes": json.dumps(checks)
    })

    # Recalculate summary from raw results
    recalculated = (
        df.groupby(
            ["dimension", "benchmark", "algorithm"],
            as_index=False
        )
        .agg(
            runs=("best_score", "count"),
            recalculated_mean=("best_score", "mean"),
            recalculated_median=("best_score", "median"),
            recalculated_std=("best_score", "std"),
            recalculated_successes=("success", "sum"),
            recalculated_runtime=("runtime_seconds", "mean"),
            unique_seeds=("seed", "nunique")
        )
    )
    recalculated["experiment"] = exp_id
    summary_rows.append(recalculated)

    # Check whether algorithms have the same seed set
    for (dim, benchmark), group in df.groupby(
        ["dimension", "benchmark"]
    ):
        seed_sets = {
            alg: set(part["seed"].tolist())
            for alg, part in group.groupby("algorithm")
        }
        all_same = (
            len(seed_sets) > 0
            and len({tuple(sorted(s)) for s in seed_sets.values()}) == 1
        )

        pairing_rows.append({
            "experiment": exp_id,
            "dimension": dim,
            "benchmark": benchmark,
            "algorithms": len(seed_sets),
            "same_seed_sets": all_same,
            "seed_counts": json.dumps({
                k: len(v) for k, v in seed_sets.items()
            })
        })

audit = pd.DataFrame(audit_rows)
audit.to_csv(OUT / "experiment_061_audit.csv", index=False)

if summary_rows:
    recalculated_summary = pd.concat(
        summary_rows, ignore_index=True
    )
else:
    recalculated_summary = pd.DataFrame()

recalculated_summary.to_csv(
    OUT / "experiment_061_recalculated_summary.csv",
    index=False
)

pairing = pd.DataFrame(pairing_rows)
pairing.to_csv(
    OUT / "experiment_061_pairing_checks.csv",
    index=False
)

report = {
    "experiment_id": "OSE-061",
    "title": "Automated Research Audit",
    "input_experiments": EXPERIMENTS,
    "audit_results": audit.to_dict(orient="records"),
    "pairing_checks": pairing.to_dict(orient="records"),
    "summary_rows": len(recalculated_summary),
    "runtime_seconds": time.perf_counter() - start_time,
    "limitations": [
        "This checks recorded CSV data, not whether the original algorithm implementation was scientifically correct.",
        "Missing experiment files are reported, not reconstructed.",
        "A successful audit does not prove a scientific hypothesis."
    ]
}

with open(
    OUT / "experiment_061_research_record.json", "w"
) as f:
    json.dump(report, f, indent=2, default=str)

with open(
    OUT / "experiment_061_research_record.md", "w"
) as f:
    f.write("# OSE Experiment 061 — Automated Research Audit\n\n")
    f.write("## Audit results\n\n")
    f.write(audit.to_string(index=False))
    f.write("\n\n## Recalculated summaries\n\n")
    if len(recalculated_summary):
        f.write(recalculated_summary.to_string(index=False))
    f.write("\n\n## Paired-seed checks\n\n")
    if len(pairing):
        f.write(pairing.to_string(index=False))
    f.write("\n\n## Limitations\n\n")
    for item in report["limitations"]:
        f.write(f"- {item}\n")

print("\n=== EXPERIMENT 061 AUDIT ===")
print(audit.to_string(index=False))

print("\n=== PAIRED-SEED CHECKS ===")
if len(pairing):
    print(pairing.to_string(index=False))
else:
    print("No paired-seed groups available.")

print("\n=== RECALCULATED SUMMARY ===")
if len(recalculated_summary):
    print(recalculated_summary.to_string(index=False))
else:
    print("No input summaries could be recalculated.")

print("\n=== SAVED FILES ===")
for name in [
    "experiment_061_audit.csv",
    "experiment_061_recalculated_summary.csv",
    "experiment_061_pairing_checks.csv",
    "experiment_061_research_record.json",
    "experiment_061_research_record.md"
]:
    print(OUT / name)

print("\nExperiment 061 audit complete.")

