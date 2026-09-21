# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Shared causal correlation.

A standalone module: it operates on plain `pandas` series and imports nothing
from `rules`, `forecasts` or `data`. The forecast diversification multiplier
(and, later, the instrument diversification multiplier and handcrafting) build on
it, so the causal correlation estimate is written once. All estimates are causal:
the value at time *t* uses only observations up to and including *t*.
"""

from __future__ import annotations

from typing import cast

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_positive_int, ensure_series

__all__ = ["rolling_correlation"]

_MIN_CORRELATION_PERIODS = 2


def rolling_correlation(
    a: FloatSeries,
    b: FloatSeries,
    *,
    window: int | None = None,
    min_periods: int | None = None,
) -> FloatSeries:
    r"""Causal Pearson correlation of two aligned series.

    Over an expanding window (the default) or a trailing window of ``window``
    observations, the Pearson correlation coefficient of ``a`` and ``b``:

    $$\rho_t = \frac{\sum (a - \bar{a})(b - \bar{b})}
      {\sqrt{\sum (a - \bar{a})^2}\,\sqrt{\sum (b - \bar{b})^2}} \in [-1, 1]$$

    where the sums run over the window ending at *t*. A correlation needs at least
    two observations, so warmup is `NaN`.

    Worked example: for ``a = [1, 2, 3]`` and ``b = [1, 3, 2]`` over an expanding
    window the first two points correlate perfectly (``+1``); over all three the
    numerator is ``1`` and each spread is ``2``, so the correlation is ``0.5``.

    Reference: Karl Pearson, the product-moment correlation coefficient (1895).

    Args:
        a: A series on a unique, monotonic `DatetimeIndex`.
        b: A series indexed identically to ``a``.
        window: Trailing-window length. `None` uses an expanding window that grows
            with the history.
        min_periods: Observations required before a value is emitted. Defaults to
            ``window`` for a trailing window, or 2 for an expanding window.

    Returns:
        The rolling correlation, carrying the shared index with `NaN` through
        warmup.

    Raises:
        ConfigurationError: If ``window`` or ``min_periods`` is not a positive
            integer.
        DataValidationError: If ``a`` and ``b`` are not indexed identically.
    """
    first = ensure_series(a, "a")
    second = ensure_series(b, "b")
    ensure_aligned(a=first, b=second)
    if window is None:
        floor = (
            _MIN_CORRELATION_PERIODS
            if min_periods is None
            else ensure_positive_int(min_periods, "min_periods")
        )
        return cast("FloatSeries", first.expanding(min_periods=floor).corr(second))
    length = ensure_positive_int(window, "window")
    floor = length if min_periods is None else ensure_positive_int(min_periods, "min_periods")
    return cast("FloatSeries", first.rolling(window=length, min_periods=floor).corr(second))
