# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Price channels: rolling extremes, Donchian, Bollinger and Keltner.

Each channel returns bands that carry the input index, NaN-padded through
warmup. All are causal.
"""

from __future__ import annotations

import pandas as pd

from quantrules._typing import FloatSeries, Frame
from quantrules._validation import (
    ensure_non_negative,
    ensure_positive,
    ensure_positive_int,
    ensure_series,
)
from quantrules.indicators.averages import ewma
from quantrules.indicators.volatility import atr

__all__ = [
    "bollinger_bands",
    "donchian_channels",
    "keltner_channels",
    "keltner_position",
    "rolling_max",
    "rolling_min",
]


def rolling_max(series: FloatSeries, *, window: int) -> FloatSeries:
    """Maximum over a trailing window.

    Worked example: for ``[1, 3, 2, 5, 4]`` with ``window=2`` the maxima are
    ``NaN, 3, 3, 5, 5``.

    Args:
        series: Input series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.

    Returns:
        The rolling maximum, carrying ``series``'s index.
    """
    validated = ensure_series(series, "series")
    length = ensure_positive_int(window, "window")
    return validated.rolling(window=length, min_periods=length).max()


def rolling_min(series: FloatSeries, *, window: int) -> FloatSeries:
    """Minimum over a trailing window.

    Worked example: for ``[1, 3, 2, 5, 4]`` with ``window=2`` the minima are
    ``NaN, 1, 2, 2, 4``.

    Args:
        series: Input series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.

    Returns:
        The rolling minimum, carrying ``series``'s index.
    """
    validated = ensure_series(series, "series")
    length = ensure_positive_int(window, "window")
    return validated.rolling(window=length, min_periods=length).min()


def donchian_channels(price: FloatSeries, *, window: int) -> Frame:
    r"""Donchian channel: the rolling high, low and their midpoint.

    Worked example: for ``[1, 3, 2, 5, 4]`` with ``window=2`` the upper band is
    ``NaN, 3, 3, 5, 5``, the lower band is ``NaN, 1, 2, 2, 4`` and the mid is
    their average, ``NaN, 2, 2.5, 3.5, 4.5``.

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.

    Returns:
        A frame with columns ``upper``, ``mid`` and ``lower``, carrying
        ``price``'s index.
    """
    upper = rolling_max(price, window=window)
    lower = rolling_min(price, window=window)
    return pd.DataFrame({"upper": upper, "mid": (upper + lower) / 2.0, "lower": lower})


def bollinger_bands(
    price: FloatSeries,
    *,
    window: int = 20,
    num_std: float = 2.0,
    ddof: int = 0,
) -> Frame:
    r"""Bollinger bands: a moving average and symmetric standard-deviation bands.

    The mid band is a simple moving average; the outer bands sit ``num_std``
    standard deviations away; the ``zscore`` column reports how many standard
    deviations the price is from the mid band.

    Worked example: for ``[1, 2, 3, 4]`` with ``window=3`` and ``ddof=0`` the
    mid at the third point is 2 and the standard deviation is :math:`\sqrt{2/3}`,
    so the z-score is :math:`(3 - 2)/\sqrt{2/3} = \sqrt{3/2}`.

    Reference: John Bollinger, *Bollinger on Bollinger Bands* (2001).

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.
        num_std: Number of standard deviations for the outer bands.
        ddof: Delta degrees of freedom of the standard deviation.

    Returns:
        A frame with columns ``mid``, ``upper``, ``lower`` and ``zscore``,
        carrying ``price``'s index.
    """
    validated = ensure_series(price, "price")
    length = ensure_positive_int(window, "window")
    width = ensure_non_negative(num_std, "num_std")
    mid = validated.rolling(window=length, min_periods=length).mean()
    std = validated.rolling(window=length, min_periods=length).std(ddof=ddof)
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": mid + width * std,
            "lower": mid - width * std,
            "zscore": (validated - mid) / std,
        }
    )


def _keltner_parts(
    high: FloatSeries,
    low: FloatSeries,
    close: FloatSeries,
    window: int,
    atr_method: str,
) -> tuple[FloatSeries, FloatSeries, FloatSeries]:
    """Return the validated close, the EWMA mid line and the ATR band."""
    closes = ensure_series(close, "close")
    mid = ewma(closes, span=window)
    band = atr(high, low, close, window=window, method=atr_method)
    return closes, mid, band


def keltner_channels(
    high: FloatSeries,
    low: FloatSeries,
    close: FloatSeries,
    *,
    window: int = 20,
    atr_mult: float = 2.0,
    atr_method: str = "wilder",
) -> Frame:
    r"""Keltner channel: an EWMA of the close with average-true-range bands.

    The mid line is an EWMA of the close; the bands sit ``atr_mult`` average
    true ranges above and below it.

    Worked example: with highs ``[10, 12, 11]``, lows ``[8, 9, 7]`` and closes
    ``[9, 11, 8]``, ``window=2`` and a simple ATR, the mid at the second bar is
    the span-2 EWMA of the close, :math:`31/3`, and the ATR is 2.5, so the upper
    band is :math:`31/3 + 2(2.5)`.

    Reference: Chester Keltner, *How to Make Money in Commodities* (1960), in the
    average-true-range form popularised by Linda Bradford Raschke.

    Args:
        high: Bar highs on a unique, monotonic `DatetimeIndex`.
        low: Bar lows, indexed identically to ``high``.
        close: Bar closes, indexed identically to ``high``.
        window: Span of the EWMA mid line and window of the ATR.
        atr_mult: Number of average true ranges for the bands.
        atr_method: ATR convention, ``"wilder"`` or ``"simple"``.

    Returns:
        A frame with columns ``mid``, ``upper`` and ``lower``, carrying the
        shared input index.
    """
    multiplier = ensure_non_negative(atr_mult, "atr_mult")
    _, mid, band = _keltner_parts(high, low, close, window, atr_method)
    offset = multiplier * band
    return pd.DataFrame({"mid": mid, "upper": mid + offset, "lower": mid - offset})


def keltner_position(
    high: FloatSeries,
    low: FloatSeries,
    close: FloatSeries,
    *,
    window: int = 20,
    atr_mult: float = 2.0,
    atr_method: str = "wilder",
) -> FloatSeries:
    r"""Where the close sits inside its Keltner channel, clamped to [-1, 1].

    Zero is the mid line, +1 the upper band and -1 the lower band; moves beyond
    the bands are clamped:

    $$\mathrm{pos}_t = \mathrm{clip}\!\left(
      \frac{c_t - \text{mid}_t}{\text{atr\_mult} \cdot \text{atr}_t},\ -1,\ 1\right)$$

    Worked example: with the bars above, ``window=2``, a simple ATR and
    ``atr_mult=2``, the second bar gives :math:`(11 - 31/3) / (2 \cdot 2.5) = 2/15`.

    Args:
        high: Bar highs on a unique, monotonic `DatetimeIndex`.
        low: Bar lows, indexed identically to ``high``.
        close: Bar closes, indexed identically to ``high``.
        window: Span of the EWMA mid line and window of the ATR.
        atr_mult: Number of average true ranges spanning the channel half-width.
        atr_method: ATR convention, ``"wilder"`` or ``"simple"``.

    Returns:
        The clamped channel position, carrying the shared input index.
    """
    multiplier = ensure_positive(atr_mult, "atr_mult")
    closes, mid, band = _keltner_parts(high, low, close, window, atr_method)
    position = (closes - mid) / (multiplier * band)
    return position.clip(lower=-1.0, upper=1.0)
