
import csv

import pytest

from src.ose.experiment_runner import run_sphere_experiment


def test_experiment_creates_expected_rows(tmp_path):
    output = tmp_path / "experiment.csv"

    rows = run_sphere_experiment(
        seeds=range(3),
        dimensions=2,
        max_evaluations=20,
        output_path=str(output),
    )

    assert len(rows) == 3
    assert output.exists()


def test_experiment_records_evaluation_count(tmp_path):
    rows = run_sphere_experiment(
        seeds=range(2),
        dimensions=2,
        max_evaluations=25,
        output_path=str(tmp_path / "results.csv"),
    )

    assert all(row["evaluations"] == 25 for row in rows)


def test_csv_contains_experiment_results(tmp_path):
    output = tmp_path / "results.csv"

    run_sphere_experiment(
        seeds=range(2),
        max_evaluations=10,
        output_path=str(output),
    )

    with output.open(newline="", encoding="utf-8") as file:
        saved_rows = list(csv.DictReader(file))

    assert len(saved_rows) == 2
    assert saved_rows[0]["benchmark"] == "Sphere"
    assert saved_rows[0]["algorithm"] == "Random Search"


def test_same_seeds_produce_same_results(tmp_path):
    first = run_sphere_experiment(
        seeds=range(3),
        max_evaluations=30,
        output_path=str(tmp_path / "first.csv"),
    )
    second = run_sphere_experiment(
        seeds=range(3),
        max_evaluations=30,
        output_path=str(tmp_path / "second.csv"),
    )

    assert [r["best_value"] for r in first] == [
        r["best_value"] for r in second
    ]
    assert [r["best_x"] for r in first] == [
        r["best_x"] for r in second
    ]


def test_invalid_dimensions_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_sphere_experiment(
            dimensions=0,
            output_path=str(tmp_path / "invalid.csv"),
        )
