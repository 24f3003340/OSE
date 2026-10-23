
import pytest

from src.ose.objectives import sphere
from src.ose.optimizers import (
    random_search,
    particle_swarm_optimization,
)

def test_random_search_is_reproducible():
    first = random_search(
        sphere, [(-5, 5), (-5, 5)],
        max_evaluations=100, seed=42,
    )
    second = random_search(
        sphere, [(-5, 5), (-5, 5)],
        max_evaluations=100, seed=42,
    )

    assert first["best_value"] == second["best_value"]
    assert first["best_x"] == second["best_x"]


def test_evaluation_count_is_correct():
    result = random_search(
        sphere, [(-5, 5), (-5, 5)],
        max_evaluations=75, seed=1,
    )
    assert result["evaluations"] == 75


def test_best_value_is_valid():
    result = random_search(
        sphere, [(-5, 5), (-5, 5)],
        max_evaluations=500, seed=42,
    )
    assert 0 <= result["best_value"] <= 50


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError):
        random_search(sphere, [(5, -5)], max_evaluations=10)


def test_zero_evaluations_are_rejected():
    with pytest.raises(ValueError):
        random_search(sphere, [(-5, 5)], max_evaluations=0)


def test_non_finite_objective_is_rejected():
    def bad_objective(x):
        return float("nan")

    with pytest.raises(ValueError):
        random_search(
            bad_objective, [(-1, 1)],
            max_evaluations=5, seed=42,
        )


def test_pso_uses_exact_evaluation_budget():
    result = particle_swarm_optimization(
        objective=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_evaluations=103,
        seed=42,
        swarm_size=30,
    )

    assert result["evaluations"] == 103
    assert result["best_value"] >= 0


def test_pso_is_reproducible():
    kwargs = {
        "objective": sphere,
        "bounds": [(-5, 5), (-5, 5)],
        "max_evaluations": 100,
        "seed": 42,
    }

    first = particle_swarm_optimization(**kwargs)
    second = particle_swarm_optimization(**kwargs)

    assert first["best_value"] == second["best_value"]
    assert first["best_x"] == second["best_x"]


def test_pso_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        particle_swarm_optimization(
            objective=sphere,
            bounds=[(5, -5)],
            max_evaluations=10,
        )


def test_pso_rejects_zero_evaluations():
    with pytest.raises(ValueError):
        particle_swarm_optimization(
            objective=sphere,
            bounds=[(-5, 5)],
            max_evaluations=0,
        )

