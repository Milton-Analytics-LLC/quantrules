# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Volatility estimators and true-range measures.

Two families live here. The return-based estimators (`rolling_std`,
`ewma_volatility`, `realized_volatility`) summarise the dispersion of a return
or price series and can annualise it through an explicit ``periods_per_year``.
The bar-based measures (`true_range`, `atr`, `atr_percent`) summarise the range
of a high/low/close bar. All are causal.
"""

from __future__ import annotations

import math
from typing import cast

import pandas as pd

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_positive_int, ensure_series
from quantrules.defaults import (
    BUSINESS_DAYS_PER_YEAR,
    VOLATILITY_EWMA_SPAN,
    VOLATILITY_MIN_PERIODS,
)
from quantrules.exceptions import ConfigurationError
from quantrules.indicators._returns import _log_returns

__all__ = [
    "atr",
    "atr_percent",
    "ewma_volatility",
    "realized_volatility",
    "rolling_std",
    "true_range",
]


def rolling_std(
    series: FloatSeries,
    *,
    window: int,
    ddof: int = 0,
    min_periods: int | None = None,
) -> FloatSeries:
    r"""Standard deviation over a trailing window.

    Worked example: for ``[1, 2, 3, 4]`` with ``window=3`` and ``ddof=0`` both
    full windows have mean 2 (resp. 3) and variance
    :math:`((-1)^2 + 0 + 1^2) / 3 = 2/3`, so the standard deviation is
    :math:`\sqrt{2/3}`.

    Args:
        series: Input series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.
        ddof: Delta degrees of freedom. ``0`` is the population standard
            deviation; ``1`` is the sample standard deviation.
        min_periods: Observations required before a value is emitted. Defaults
            to ``window``.

    Returns:
        The rolling standard deviation, carrying ``series``'s index.
    """
    validated = ensure_series(series, "series")
    length = ensure_positive_int(window, "window")
    floor = length if min_periods is None else ensure_positive_int(min_periods, "min_periods")
    return validated.rolling(window=length, min_periods=floor).std(ddof=ddof)


def ewma_volatility(
    returns: FloatSeries,
    *,
    span: int = VOLATILITY_EWMA_SPAN,
    min_periods: int = VOLATILITY_MIN_PERIODS,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
) -> FloatSeries:
    r"""Exponentially weighted volatility of a return series.

    The RiskMetrics estimator: an exponentially weighted moving average of
    squared returns (a zero-mean variance, appropriate for the near-zero mean
    of high-frequency returns), then a square root, then annualisation. With
    smoothing factor :math:`\alpha = 2 / (s + 1)` for span *s*:

    $$v_0 = r_0^2, \quad v_t = \alpha r_t^2 + (1 - \alpha) v_{t-1}, \quad
      \sigma_t = \sqrt{v_t}\,\sqrt{\text{periods\_per\_year}}$$

    Worked example: returns ``[0.01, -0.02, 0.03]`` with ``span=2`` give squares
    ``[0.0001, 0.0004, 0.0009]``; the EWMA of those is ``0.0001``, ``0.0003``,
    ``0.0007``, whose roots are the per-period volatilities.

    Reference: J.P. Morgan/Reuters, *RiskMetrics Technical Document*, 4th ed.
    (1996), the exponentially weighted volatility estimator.

    Args:
        returns: A return series (for example ``price.pct_change()``) on a
            unique, monotonic `DatetimeIndex`.
        span: Span of the exponential weighting; larger spans react more slowly.
        min_periods: Observations required before a value is emitted.
        periods_per_year: Periods per year used to annualise. Pass ``1`` to
            leave the estimate per-period.

    Returns:
        The annualised volatility, carrying ``returns``'s index.
    """
    validated = ensure_series(returns, "returns")
    length = ensure_positive_int(span, "span")
    floor = ensure_positive_int(min_periods, "min_periods")
    yearly = ensure_positive_int(periods_per_year, "periods_per_year")
    variance = (validated**2).ewm(span=length, adjust=False, min_periods=floor).mean()
    return cast("FloatSeries", variance.pow(0.5) * math.sqrt(yearly))


def realized_volatility(
    price: FloatSeries,
    *,
    window: int,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
    ddof: int = 1,
) -> FloatSeries:
    r"""Annualised standard deviation of log returns over a trailing window.

    Log returns :math:`\ln(p_t / p_{t-1})` are taken, their standard deviation
    is measured over ``window`` returns, then annualised by
    :math:`\sqrt{\text{periods\_per\_year}}`.

    Worked example: prices ``[1, e, 1, e]`` give log returns ``[1, -1, 1]``; a
    two-return window has mean 0 and sample variance
    :math:`(1^2 + (-1)^2) / (2 - 1) = 2`, so the volatility is :math:`\sqrt{2}`.

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        window: Number of returns in the trailing window.
        periods_per_year: Periods per year used to annualise. Pass ``1`` to
            leave the estimate per-period.
        ddof: Delta degrees of freedom of the standard deviation. ``1`` is the
            sample standard deviation.

    Returns:
        The annualised realised volatility, carrying ``price``'s index.
    """
    validated = ensure_series(price, "price")
    length = ensure_positive_int(window, "window")
    yearly = ensure_positive_int(periods_per_year, "periods_per_year")
    log_returns = _log_returns(validated)
    dispersion = log_returns.rolling(window=length, min_periods=length).std(ddof=ddof)
    return cast("FloatSeries", dispersion * math.sqrt(yearly))


def true_range(high: FloatSeries, low: FloatSeries, close: FloatSeries) -> FloatSeries:
    r"""True range of each bar.

    The true range is the widest of the current high-low range and the gaps
    from the previous close:

    $$\mathrm{TR}_t = \max\bigl(h_t - l_t,\ |h_t - c_{t-1}|,\ |l_t - c_{t-1}|\bigr)$$

    On the first bar there is no previous close, so the true range is simply
    ``high - low``.

    Worked example: with highs ``[10, 12, 11]``, lows ``[8, 9, 7]`` and closes
    ``[9, 11, 8]`` the true ranges are ``10 - 8 = 2``, then
    ``max(3, |12 - 9|, |9 - 9|) = 3``, then ``max(4, |11 - 11|, |7 - 11|) = 4``.

    Args:
        high: Bar highs on a unique, monotonic `DatetimeIndex`.
        low: Bar lows, indexed identically to ``high``.
        close: Bar closes, indexed identically to ``high``.

    Returns:
        The true range, carrying the shared input index.
    """
    highs = ensure_series(high, "high")
    lows = ensure_series(low, "low")
    closes = ensure_series(close, "close")
    ensure_aligned(high=highs, low=lows, close=closes)
    prev_close = closes.shift(1)
    ranges = pd.DataFrame(
        {
            "high_low": highs - lows,
            "high_close": (highs - prev_close).abs(),
            "low_close": (lows - prev_close).abs(),
        }
    )
    return cast("FloatSeries", ranges.max(axis=1))


def atr(
    high: FloatSeries,
    low: FloatSeries,
    close: FloatSeries,
    *,
    window: int = 14,
    method: str = "wilder",
) -> FloatSeries:
    r"""Average true range.

    Smooths the [`true_range`][quantrules.indicators.volatility.true_range] over
    ``window`` bars. Two conventions:

    - ``"wilder"``: Wilder's smoothing, an exponential average with factor
      :math:`\alpha = 1 / n`, seeded from the first true range.
    - ``"simple"``: a simple moving average of the true range.

    Worked example: true ranges ``[2, 3, 4]`` with ``window=2``. Wilder
    (:math:`\alpha = 1/2`) gives ``0.5(3) + 0.5(2) = 2.5`` then
    ``0.5(4) + 0.5(2.5) = 3.25``; the simple average gives ``2.5`` then
    ``(3 + 4) / 2 = 3.5``.

    Reference: J. Welles Wilder Jr., *New Concepts in Technical Trading Systems*
    (1978).

    Args:
        high: Bar highs on a unique, monotonic `DatetimeIndex`.
        low: Bar lows, indexed identically to ``high``.
        close: Bar closes, indexed identically to ``high``.
        window: Number of bars to smooth over.
        method: ``"wilder"`` or ``"simple"``.

    Returns:
        The average true range, carrying the shared input index.

    Raises:
        ConfigurationError: If ``method`` is not ``"wilder"`` or ``"simple"``.
    """
    length = ensure_positive_int(window, "window")
    true_ranges = true_range(high, low, close)
    if method == "wilder":
        smoothed = true_ranges.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
        return cast("FloatSeries", smoothed)
    if method == "simple":
        return true_ranges.rolling(window=length, min_periods=length).mean()
    message = f"method must be 'wilder' or 'simple', got {method!r}"
    raise ConfigurationError(message)


def atr_percent(
    high: FloatSeries,
    low: FloatSeries,
    close: FloatSeries,
    *,
    window: int = 14,
    method: str = "wilder",
) -> FloatSeries:
    r"""Average true range as a fraction of the close.

    The [`atr`][quantrules.indicators.volatility.atr] divided by the current
    close, giving a unit-free volatility that is comparable across instruments
    at different price levels.

    Args:
        high: Bar highs on a unique, monotonic `DatetimeIndex`.
        low: Bar lows, indexed identically to ``high``.
        close: Bar closes, indexed identically to ``high``.
        window: Number of bars to smooth over.
        method: ``"wilder"`` or ``"simple"``.

    Returns:
        The ATR as a fraction of the close, carrying the shared input index.
    """
    average_true_range = atr(high, low, close, window=window, method=method)
    closes = ensure_series(close, "close")
    return average_true_range / closes
