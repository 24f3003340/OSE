
import numpy as np
import pytest

from src.ose.objectives import sphere, rastrigin


def test_sphere_global_minimum():
    assert sphere([0, 0, 0]) == 0.0


def test_sphere_known_value():
    assert sphere([3, 4]) == 25.0


def test_rastrigin_global_minimum():
    assert rastrigin([0, 0, 0]) == pytest.approx(0.0)


def test_rastrigin_known_value():
    assert rastrigin([1, 1]) == pytest.approx(2.0)


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        sphere([])


def test_non_finite_input_rejected():
    with pytest.raises(ValueError):
        rastrigin([np.nan])
