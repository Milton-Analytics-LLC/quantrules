# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Trend measures: MACD, the raw EWMA crossover, and rolling slope.

All are causal. The EWMA-based measures use the recursive convention of
[`averages.ewma`][quantrules.indicators.averages.ewma] (pandas ``adjust=False``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrules._typing import FloatSeries, Frame
from quantrules._validation import ensure_positive_int, ensure_series
from quantrules.exceptions import ConfigurationError

__all__ = ["ewma_crossover", "macd", "rolling_slope"]

_MIN_SLOPE_WINDOW = 2


def _ewma(series: FloatSeries, span: int) -> FloatSeries:
    """Recursive EWMA seeded from the first observation (pandas ``adjust=False``)."""
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def macd(price: FloatSeries, *, fast: int = 12, slow: int = 26, signal: int = 9) -> Frame:
    r"""Moving-average convergence/divergence.

    Three series: the MACD *line* is the difference of a fast and a slow EWMA of
    the price; the *signal* is an EWMA of the line; the *histogram* is the line
    minus the signal.

    $$\text{line} = \mathrm{EWMA}_{\text{fast}}(p) - \mathrm{EWMA}_{\text{slow}}(p),
      \quad \text{signal} = \mathrm{EWMA}_{\text{signal}}(\text{line}),
      \quad \text{hist} = \text{line} - \text{signal}$$

    Worked example: for ``[1, 2, 3]`` with ``fast=1`` (so the fast EWMA is the
    price), ``slow=2`` and ``signal=2`` the line is ``0, 1/3, 4/9``, the signal
    is ``0, 2/9, 10/27`` and the histogram is ``0, 1/9, 2/27``.

    Reference: Gerald Appel, *Technical Analysis: Power Tools for Active
    Investors* (2005).

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        fast: Span of the fast EWMA.
        slow: Span of the slow EWMA.
        signal: Span of the EWMA applied to the line.

    Returns:
        A frame with columns ``line``, ``signal`` and ``histogram``, carrying
        ``price``'s index.
    """
    validated = ensure_series(price, "price")
    fast_span = ensure_positive_int(fast, "fast")
    slow_span = ensure_positive_int(slow, "slow")
    signal_span = ensure_positive_int(signal, "signal")
    line = _ewma(validated, fast_span) - _ewma(validated, slow_span)
    signal_line = _ewma(line, signal_span)
    histogram = line - signal_line
    return pd.DataFrame({"line": line, "signal": signal_line, "histogram": histogram})


def ewma_crossover(price: FloatSeries, *, fast: int, slow: int) -> FloatSeries:
    r"""Raw EWMA crossover: a fast EWMA minus a slow EWMA of the price.

    This is the unscaled core of Carver's EWMAC trend rule, before it is
    normalised by volatility and scaled to a target forecast (both later
    phases). It is positive when the fast average is above the slow one.

    Worked example: for ``[1, 2, 3]`` with ``fast=1`` (identity) and ``slow=2``
    the crossover is ``0, 1/3, 4/9``.

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        fast: Span of the fast EWMA.
        slow: Span of the slow EWMA.

    Returns:
        The crossover series, carrying ``price``'s index.
    """
    validated = ensure_series(price, "price")
    fast_span = ensure_positive_int(fast, "fast")
    slow_span = ensure_positive_int(slow, "slow")
    return _ewma(validated, fast_span) - _ewma(validated, slow_span)


def rolling_slope(series: FloatSeries, *, window: int) -> FloatSeries:
    r"""Ordinary-least-squares slope over a trailing window.

    Fits a straight line to the ``window`` most recent observations against the
    within-window position :math:`x = 0, 1, \dots, n-1` and returns its slope:

    $$\text{slope}_t = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}$$

    For a series that is linear in time the slope is recovered exactly. Worked
    example: for ``y = [1, 3, 2]`` with ``window=3`` the slope is
    :math:`(0\cdot1 + 1\cdot3 + 2\cdot2 - 1\cdot6) / 2 = 0.5`.

    Args:
        series: Input series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.

    Returns:
        The rolling slope, carrying ``series``'s index with `NaN` through warmup.

    Raises:
        ConfigurationError: If ``window`` is less than 2.
    """
    validated = ensure_series(series, "series")
    length = ensure_positive_int(window, "window")
    if length < _MIN_SLOPE_WINDOW:
        message = f"window must be at least {_MIN_SLOPE_WINDOW} to fit a slope, got {length}"
        raise ConfigurationError(message)
    values = validated.to_numpy()
    positions = np.arange(length)
    centred = positions - positions.mean()
    sum_squares = float((centred**2).sum())
    slopes = np.full(len(values), np.nan)
    if len(values) >= length:
        windows = np.lib.stride_tricks.sliding_window_view(values, length)
        numerator = windows @ centred
        slopes[length - 1 :] = numerator / sum_squares
    return pd.Series(slopes, index=validated.index)
