# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Volume-flow indicators: OBV, accumulation/distribution and Chaikin money flow.

Carver's own rules do not use volume, but downstream trend and confirmation
rules do, so these are provided as first-class causal indicators.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_positive_int, ensure_series

__all__ = ["accumulation_distribution", "chaikin_money_flow", "obv"]


def _money_flow_volume(
    high: FloatSeries, low: FloatSeries, close: FloatSeries, volume: FloatSeries
) -> FloatSeries:
    r"""Money-flow volume: the money-flow multiplier times volume.

    The multiplier :math:`((c - l) - (h - c)) / (h - l)` lies in ``[-1, 1]`` and
    is zero when the bar has no range. A bar with a missing high, low or close
    stays `NaN` so it is not counted as zero flow.
    """
    spread = high - low
    multiplier = ((close - low) - (high - close)) / spread
    # A finite zero-range bar (high == low) carries no directional information,
    # so force its multiplier to 0. Missing inputs leave a NaN spread, which the
    # comparison treats as not-equal-to-zero, so the bar stays unknown.
    multiplier = multiplier.mask(spread == 0.0, 0.0)
    return cast("FloatSeries", multiplier * volume)


def obv(close: FloatSeries, volume: FloatSeries) -> FloatSeries:
    r"""On-balance volume: a running total of volume signed by the price move.

    Volume is added on an up day, subtracted on a down day and ignored when the
    close is unchanged; the first bar contributes nothing. A later missing close
    or volume leaves the running total unknown (`NaN`) from that bar on.

    $$\mathrm{OBV}_t = \mathrm{OBV}_{t-1} + \operatorname{sign}(c_t - c_{t-1})\, v_t$$

    Worked example: closes ``[10, 11, 10, 10, 12]`` with volumes
    ``[100, 200, 150, 300, 250]`` give ``0, 200, 50, 50, 300``.

    Reference: Joseph Granville, *Granville's New Key to Stock Market Profits*
    (1963).

    Args:
        close: Close series on a unique, monotonic `DatetimeIndex`.
        volume: Volume series, indexed identically to ``close``.

    Returns:
        The on-balance volume, carrying the shared input index.
    """
    closes = ensure_series(close, "close")
    volumes = ensure_series(volume, "volume")
    ensure_aligned(close=closes, volume=volumes)
    direction = np.sign(closes.diff().to_numpy())
    signed_volume = pd.Series(direction, index=closes.index) * volumes
    # The first bar has no prior close and contributes nothing; a later missing
    # close or volume leaves the cumulative total unknown from that point on.
    first_bar = pd.Series(np.arange(len(closes)) == 0, index=closes.index)
    return signed_volume.mask(first_bar, 0.0).cumsum(skipna=False)


def accumulation_distribution(
    high: FloatSeries, low: FloatSeries, close: FloatSeries, volume: FloatSeries
) -> FloatSeries:
    r"""Accumulation/distribution line: a running total of money-flow volume.

    Each bar contributes volume weighted by where the close sits in the bar's
    range: +volume at the high, -volume at the low, zero at the midpoint. A bar
    with missing inputs leaves the line unknown (`NaN`) from that bar on.

    Worked example: three bars with high 10, low 8 and volumes ``[100, 200,
    300]`` closing at the high, the low and the midpoint give ``100``, then
    ``100 - 200 = -100``, then ``-100`` unchanged.

    Reference: Marc Chaikin's accumulation/distribution line.

    Args:
        high: Bar highs on a unique, monotonic `DatetimeIndex`.
        low: Bar lows, indexed identically to ``high``.
        close: Bar closes, indexed identically to ``high``.
        volume: Bar volumes, indexed identically to ``high``.

    Returns:
        The accumulation/distribution line, carrying the shared input index.
    """
    highs = ensure_series(high, "high")
    lows = ensure_series(low, "low")
    closes = ensure_series(close, "close")
    volumes = ensure_series(volume, "volume")
    ensure_aligned(high=highs, low=lows, close=closes, volume=volumes)
    # skipna=False so a bar with missing inputs leaves the line unknown onward.
    return _money_flow_volume(highs, lows, closes, volumes).cumsum(skipna=False)


def chaikin_money_flow(
    high: FloatSeries,
    low: FloatSeries,
    close: FloatSeries,
    volume: FloatSeries,
    *,
    window: int = 20,
) -> FloatSeries:
    r"""Chaikin money flow: money-flow volume over total volume in a window.

    $$\mathrm{CMF}_t = \frac{\sum_{\text{window}} \text{money-flow volume}}
                            {\sum_{\text{window}} \text{volume}} \in [-1, 1]$$

    Worked example: with the bars above and ``window=2`` the money-flow volumes
    are ``[100, -200, 0]``, so the ratios are ``-100/300 = -1/3`` and
    ``-200/500 = -0.4``.

    Reference: Marc Chaikin's money-flow oscillator.

    Args:
        high: Bar highs on a unique, monotonic `DatetimeIndex`.
        low: Bar lows, indexed identically to ``high``.
        close: Bar closes, indexed identically to ``high``.
        volume: Bar volumes, indexed identically to ``high``.
        window: Number of bars in the trailing window.

    Returns:
        The Chaikin money flow, carrying the shared input index.
    """
    highs = ensure_series(high, "high")
    lows = ensure_series(low, "low")
    closes = ensure_series(close, "close")
    volumes = ensure_series(volume, "volume")
    ensure_aligned(high=highs, low=lows, close=closes, volume=volumes)
    length = ensure_positive_int(window, "window")
    money_flow = _money_flow_volume(highs, lows, closes, volumes)
    flow_sum = money_flow.rolling(window=length, min_periods=length).sum()
    volume_sum = volumes.rolling(window=length, min_periods=length).sum()
    return flow_sum / volume_sum
