# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Moving averages: simple and exponentially weighted.

These are the smoothing primitives most other indicators and every trend rule
are built from. Both are causal: the value at time *t* uses only observations
up to and including *t*.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_positive_int, ensure_series

__all__ = ["ewma", "sma"]


def sma(price: FloatSeries, *, window: int, min_periods: int | None = None) -> FloatSeries:
    r"""Simple moving average over a trailing window.

    The value at each timestamp is the arithmetic mean of the ``window`` most
    recent observations, including the current one:

    $$\mathrm{SMA}_t = \frac{1}{n} \sum_{i=0}^{n-1} x_{t-i}$$

    Worked example: for ``[1, 2, 3, 4, 5]`` with ``window=3`` the first two
    positions are `NaN` (fewer than three observations), then the averages are
    ``(1 + 2 + 3) / 3 = 2``, ``(2 + 3 + 4) / 3 = 3`` and ``(3 + 4 + 5) / 3 = 4``.

    Args:
        price: Input series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.
        min_periods: Observations required before a value is emitted. Defaults
            to ``window``, so warmup is `NaN` until the window is full.

    Returns:
        The moving average, carrying ``price``'s index with `NaN` through warmup.
    """
    validated = ensure_series(price, "price")
    length = ensure_positive_int(window, "window")
    floor = length if min_periods is None else ensure_positive_int(min_periods, "min_periods")
    return validated.rolling(window=length, min_periods=floor).mean()


def ewma(price: FloatSeries, *, span: int, min_periods: int | None = None) -> FloatSeries:
    r"""Exponentially weighted moving average, recursive (Carver) convention.

    Uses the recursive form with smoothing factor :math:`\alpha = 2 / (s + 1)`
    for span *s*, seeded from the first observation (pandas ``adjust=False``):

    $$\mathrm{EWMA}_0 = x_0, \qquad
      \mathrm{EWMA}_t = \alpha\, x_t + (1 - \alpha)\, \mathrm{EWMA}_{t-1}$$

    This is Carver's convention. It differs from an EMA seeded with a simple
    moving average of the first *s* points; the two disagree during and after
    warmup, so match the convention when comparing against another library.

    Worked example: for ``[1, 2, 3]`` with ``span=2`` the factor is
    :math:`\alpha = 2/3`, giving ``1``, ``(2/3)(2) + (1/3)(1) = 5/3`` and
    ``(2/3)(3) + (1/3)(5/3) = 23/9``.

    Args:
        price: Input series on a unique, monotonic `DatetimeIndex`.
        span: Span *s* of the average; larger spans smooth more.
        min_periods: Observations required before a value is emitted. Defaults
            to 1, so a value is emitted from the first observation.

    Returns:
        The exponentially weighted average, carrying ``price``'s index.
    """
    validated = ensure_series(price, "price")
    periods = ensure_positive_int(span, "span")
    floor = 1 if min_periods is None else ensure_positive_int(min_periods, "min_periods")
    return validated.ewm(span=periods, adjust=False, min_periods=floor).mean()
