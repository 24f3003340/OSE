
"""Validation helpers for OSE research results."""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "source_experiment",
    "benchmark",
    "dimension",
    "algorithm",
    "runs",
    "mean_score",
    "median_score",
    "std_score",
    "rank_by_mean",
}


def validate_leaderboard(leaderboard: pd.DataFrame) -> list[dict]:
    """Validate leaderboard structure, values, and ranking."""

    checks = []

    def add_check(name, passed, detail):
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": str(detail),
        })

    missing = REQUIRED_COLUMNS - set(leaderboard.columns)
    add_check(
        "Required columns",
        not missing,
        f"Missing columns: {sorted(missing)}",
    )

    if missing:
        return checks

    df = leaderboard.copy()

    numeric_cols = [
        "dimension", "runs", "mean_score",
        "median_score", "std_score", "rank_by_mean",
    ]

    for col in numeric_cols:
        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")

        # Missing standard deviation is allowed.
        invalid = converted.isna() & original.notna()
        if col != "std_score":
            invalid |= converted.isna()

        add_check(
            f"Numeric field: {col}",
            not invalid.any(),
            f"Invalid values: {int(invalid.sum())}",
        )
        df[col] = converted

    finite_cols = ["dimension", "runs", "mean_score",
                   "median_score", "rank_by_mean"]

    finite = np.isfinite(df[finite_cols].to_numpy(dtype=float)).all()
    add_check("Finite required metrics", finite, f"Rows: {len(df)}")

    valid_std = df["std_score"].dropna().ge(0).all()
    add_check("Standard deviation non-negative", valid_std, "NaN is allowed")

    positive_runs = df["runs"].gt(0).all()
    add_check("Positive run counts", positive_runs, "Every group needs runs")

    rank_group = ["source_experiment", "benchmark", "dimension"]
    expected = df.groupby(rank_group)["mean_score"].rank(
        method="min", ascending=True
    )

    rank_ok = np.allclose(
        df["rank_by_mean"].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
    )
    add_check("Ranking consistency", rank_ok, "Ranks compared with mean scores")

    return checks
