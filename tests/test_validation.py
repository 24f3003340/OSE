
import pandas as pd

from src.ose.validation import validate_leaderboard


def sample_leaderboard():
    return pd.DataFrame([
        {
            "source_experiment": "059",
            "benchmark": "Sphere",
            "dimension": 2,
            "algorithm": "PSO",
            "runs": 10,
            "mean_score": 1.0,
            "median_score": 0.8,
            "std_score": 0.2,
            "rank_by_mean": 1,
        },
        {
            "source_experiment": "059",
            "benchmark": "Sphere",
            "dimension": 2,
            "algorithm": "Random Search",
            "runs": 10,
            "mean_score": 5.0,
            "median_score": 4.5,
            "std_score": 1.0,
            "rank_by_mean": 2,
        },
    ])


def test_valid_leaderboard_passes():
    checks = validate_leaderboard(sample_leaderboard())
    failed = [c for c in checks if c["status"] == "FAIL"]
    assert failed == []


def test_missing_column_fails():
    df = sample_leaderboard().drop(columns=["algorithm"])
    checks = validate_leaderboard(df)
    required_check = next(
        c for c in checks if c["check"] == "Required columns"
    )
    assert required_check["status"] == "FAIL"


def test_wrong_ranking_fails():
    df = sample_leaderboard()
    df.loc[0, "rank_by_mean"] = 2
    df.loc[1, "rank_by_mean"] = 1

    checks = validate_leaderboard(df)
    rank_check = next(
        c for c in checks if c["check"] == "Ranking consistency"
    )
    assert rank_check["status"] == "FAIL"
