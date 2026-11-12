import pytest

from beauty_scorer.percentile import percentile_of

REFERENCE = [1.0, 2.0, 3.0, 4.0, 5.0]


def test_scalar_value():
    assert float(percentile_of(3.0, REFERENCE)) == 60.0


def test_multiple_values():
    values = percentile_of([1.0, 5.0], REFERENCE)
    assert [float(v) for v in values] == [20.0, 100.0]


def test_below_reference_minimum():
    assert float(percentile_of(0.5, REFERENCE)) == 0.0


def test_above_reference_maximum():
    assert float(percentile_of(9.0, REFERENCE)) == 100.0


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        percentile_of(3.0, [])
