"""Percentile conversion based on an empirical reference distribution."""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np

Number = Union[int, float]


def percentile_of(
    values: Union[Number, Iterable[Number]],
    reference_sorted: Iterable[Number],
) -> np.ndarray:
    """Map raw scores to percentiles of a reference distribution.

    Parameters
    ----------
    values:
        A single score or an iterable of scores.
    reference_sorted:
        Sorted reference scores, e.g. human ratings from a calibration set.

    Returns
    -------
    numpy.ndarray
        Percentile in ``[0, 100]``. A scalar input produces a 0-d array.
    """
    ref = np.asarray(list(reference_sorted), dtype=float)
    if ref.size == 0:
        raise ValueError("reference_sorted must not be empty")

    x = np.asarray(values, dtype=float)
    return np.searchsorted(ref, x, side="right") / ref.size * 100.0
