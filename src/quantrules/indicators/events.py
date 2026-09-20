# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Event and state detectors.

Small causal building blocks for trading rules: where one series crosses
another, and how long since an event last fired. Each returns a numeric series
carrying the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_series

__all__ = ["bars_since", "cross_down", "cross_up"]


def cross_up(fast: FloatSeries, slow: FloatSeries) -> FloatSeries:
    r"""Flag (1.0) each bar where ``fast`` crosses from at-or-below to above ``slow``.

    The flag is 1.0 when ``fast > slow`` now and ``fast <= slow`` on the prior
    bar, else 0.0.

    Worked example: ``fast = [1, 2, 1, 3]`` against ``slow = [2, 1, 2, 2]``
    crosses up at bars 1 and 3, giving ``0, 1, 0, 1``.

    Args:
        fast: The faster series on a unique, monotonic `DatetimeIndex`.
        slow: The slower series, indexed identically to ``fast``.

    Returns:
        A 0/1 series, carrying the shared input index.
    """
    faster = ensure_series(fast, "fast")
    slower = ensure_series(slow, "slow")
    ensure_aligned(fast=faster, slow=slower)
    difference = faster - slower
    previous = difference.shift(1)
    crossed = (difference > 0.0) & (previous <= 0.0)
    return crossed.astype("float64")


def cross_down(fast: FloatSeries, slow: FloatSeries) -> FloatSeries:
    r"""Flag (1.0) each bar where ``fast`` crosses from at-or-above to below ``slow``.

    The flag is 1.0 when ``fast < slow`` now and ``fast >= slow`` on the prior
    bar, else 0.0.

    Worked example: ``fast = [1, 2, 1, 3]`` against ``slow = [2, 1, 2, 2]``
    crosses down at bar 2, giving ``0, 0, 1, 0``.

    Args:
        fast: The faster series on a unique, monotonic `DatetimeIndex`.
        slow: The slower series, indexed identically to ``fast``.

    Returns:
        A 0/1 series, carrying the shared input index.
    """
    faster = ensure_series(fast, "fast")
    slower = ensure_series(slow, "slow")
    ensure_aligned(fast=faster, slow=slower)
    difference = faster - slower
    previous = difference.shift(1)
    crossed = (difference < 0.0) & (previous >= 0.0)
    return crossed.astype("float64")


def bars_since(events: FloatSeries) -> FloatSeries:
    r"""Number of bars since the most recent event.

    An event is any non-zero, non-`NaN` value. The count is 0 on an event bar
    and climbs by one each bar after; it is `NaN` before the first event.

    Worked example: events ``[0, 1, 0, 0, 1, 0]`` give ``NaN, 0, 1, 2, 0, 1``.

    Args:
        events: A series whose non-zero entries mark events, on a unique,
            monotonic `DatetimeIndex`.

    Returns:
        The bar count since the last event, carrying ``events``'s index.
    """
    validated = ensure_series(events, "events")
    occurred = validated.fillna(0.0) != 0.0
    positions = pd.Series(np.arange(len(validated), dtype="float64"), index=validated.index)
    last_event = positions.where(occurred, np.nan).ffill()
    return positions - last_event
