# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Market-breadth indicators computed from advance/decline data.

These operate on breadth counts and volumes (not a single instrument's price),
but obey the same data contract: numeric series on a shared `DatetimeIndex`,
index-preserving output, no look-ahead. Optional smoothing is left to the
caller to compose with [`averages`][quantrules.indicators.averages].
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_positive_int, ensure_series
from quantrules.indicators.averages import ewma

__all__ = [
    "advance_decline_line",
    "mcclellan_oscillator",
    "new_high_new_low_index",
    "trin",
]

_MCCLELLAN_FAST_SPAN = 19
_MCCLELLAN_SLOW_SPAN = 39


def advance_decline_line(advances: FloatSeries, declines: FloatSeries) -> FloatSeries:
    r"""Advance/decline line: the running total of net advancing issues.

    $$\mathrm{ADL}_t = \sum_{i \le t} (\text{advances}_i - \text{declines}_i)$$

    Worked example: net advances ``20, 20, -20`` accumulate to ``20, 40, 20``.

    Args:
        advances: Count of advancing issues on a unique, monotonic `DatetimeIndex`.
        declines: Count of declining issues, indexed identically to ``advances``.

    Returns:
        The advance/decline line, carrying the shared input index.
    """
    advancing = ensure_series(advances, "advances")
    declining = ensure_series(declines, "declines")
    ensure_aligned(advances=advancing, declines=declining)
    return (advancing - declining).cumsum()


def mcclellan_oscillator(
    advances: FloatSeries,
    declines: FloatSeries,
    *,
    fast_span: int = _MCCLELLAN_FAST_SPAN,
    slow_span: int = _MCCLELLAN_SLOW_SPAN,
) -> FloatSeries:
    r"""McClellan oscillator: the difference of two EWMAs of net advances.

    $$\text{McClellan}_t =
      \mathrm{EWMA}_{\text{fast}}(\text{net})_t - \mathrm{EWMA}_{\text{slow}}(\text{net})_t,
      \qquad \text{net} = \text{advances} - \text{declines}$$

    Worked example: with ``fast_span=1`` (so the fast EWMA is net advances
    itself) and ``slow_span=2`` on net advances ``[20, 20, -20]`` the oscillator
    is ``0``, ``0`` then :math:`-40/3`.

    Reference: Sherman and Marian McClellan's breadth oscillator (1969).

    Args:
        advances: Count of advancing issues on a unique, monotonic `DatetimeIndex`.
        declines: Count of declining issues, indexed identically to ``advances``.
        fast_span: Span of the fast EWMA (conventionally 19).
        slow_span: Span of the slow EWMA (conventionally 39).

    Returns:
        The McClellan oscillator, carrying the shared input index.
    """
    advancing = ensure_series(advances, "advances")
    declining = ensure_series(declines, "declines")
    ensure_aligned(advances=advancing, declines=declining)
    fast = ensure_positive_int(fast_span, "fast_span")
    slow = ensure_positive_int(slow_span, "slow_span")
    net = advancing - declining
    return ewma(net, span=fast) - ewma(net, span=slow)


def new_high_new_low_index(new_highs: FloatSeries, new_lows: FloatSeries) -> FloatSeries:
    r"""Net new highs: new highs minus new lows.

    Worked example: new highs ``[5, 3, 8]`` less new lows ``[2, 4, 1]`` give
    ``3, -1, 7``.

    Args:
        new_highs: Count of new highs on a unique, monotonic `DatetimeIndex`.
        new_lows: Count of new lows, indexed identically to ``new_highs``.

    Returns:
        The net new-high count, carrying the shared input index.
    """
    highs = ensure_series(new_highs, "new_highs")
    lows = ensure_series(new_lows, "new_lows")
    ensure_aligned(new_highs=highs, new_lows=lows)
    return highs - lows


def trin(
    advances: FloatSeries,
    declines: FloatSeries,
    advancing_volume: FloatSeries,
    declining_volume: FloatSeries,
) -> FloatSeries:
    r"""Arms index (TRIN): the advance/decline ratio over the up/down volume ratio.

    $$\mathrm{TRIN}_t = \frac{\text{advances}_t / \text{declines}_t}
                             {\text{advancing volume}_t / \text{declining volume}_t}$$

    A value below one is broadly bullish (volume concentrated in advancers).

    Worked example: ``(100/50) / (1000/2000) = 4`` and
    ``(60/120) / (800/400) = 0.25``.

    Reference: Richard Arms, *The Arms Index (TRIN)* (1989).

    Args:
        advances: Count of advancing issues on a unique, monotonic `DatetimeIndex`.
        declines: Count of declining issues, indexed identically to ``advances``.
        advancing_volume: Volume in advancing issues, indexed identically.
        declining_volume: Volume in declining issues, indexed identically.

    Returns:
        The Arms index, carrying the shared input index.
    """
    advancing = ensure_series(advances, "advances")
    declining = ensure_series(declines, "declines")
    advancing_vol = ensure_series(advancing_volume, "advancing_volume")
    declining_vol = ensure_series(declining_volume, "declining_volume")
    ensure_aligned(
        advances=advancing,
        declines=declining,
        advancing_volume=advancing_vol,
        declining_volume=declining_vol,
    )
    return (advancing / declining) / (advancing_vol / declining_vol)
