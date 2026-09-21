# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""The breakout rule: where price sits within its recent high-low channel.

Container-free maths. Causal: the value at time *t* uses only observations up to
and including *t*.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_positive_int, ensure_series
from quantrules.exceptions import ConfigurationError
from quantrules.indicators.averages import ewma
from quantrules.indicators.channels import donchian_channels

__all__ = ["breakout"]

_MIN_BREAKOUT_SPAN = 2


def breakout(
    price: FloatSeries,
    *,
    span: int,
    smoothing_span: int | None = None,
) -> FloatSeries:
    r"""Raw breakout forecast: price's position within its rolling channel.

    Carver's breakout rule. Over a trailing window of ``span`` observations it
    measures where the price sits between the channel low and high, relative to
    the channel mid-point:

    $$\text{breakout}_t = \frac{p_t - \text{mid}_t}{\text{upper}_t - \text{lower}_t}
      \in \left[-\tfrac{1}{2}, \tfrac{1}{2}\right]$$

    where ``upper`` and ``lower`` are the rolling maximum and minimum and ``mid``
    is their average (a
    [`donchian_channels`][quantrules.indicators.channels.donchian_channels]). The
    forecast is ``+1/2`` at a new high, ``-1/2`` at a new low and ``0`` at the
    mid-point. It is *raw*: scaling and capping happen in `quantrules.forecasts`.
    An optional EWMA of span ``smoothing_span`` damps the whipsaw at the channel
    edges.

    Worked example: for ``price = [1, 3, 2, 5, 4]`` with ``span=2`` the mid is
    ``-, 2, 2.5, 3.5, 4.5`` and the range is ``-, 2, 1, 3, 1``, so the forecast is
    ``NaN, 0.5, -0.5, 0.5, -0.5``.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), the
    breakout trading rule.

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        span: Number of observations in the trailing channel window.
        smoothing_span: Span of an optional EWMA applied to the raw forecast.
            ``None`` leaves it unsmoothed.

    Returns:
        The raw breakout forecast, carrying ``price``'s index with `NaN` through
        warmup.

    Raises:
        ConfigurationError: If ``span`` is below 2, or ``smoothing_span`` is not a
            positive integer.
    """
    validated = ensure_series(price, "price")
    length = ensure_positive_int(span, "span")
    if length < _MIN_BREAKOUT_SPAN:
        message = f"span must be at least {_MIN_BREAKOUT_SPAN} for a channel, got {length}"
        raise ConfigurationError(message)
    channel = donchian_channels(validated, window=length)
    raw = (validated - channel["mid"]) / (channel["upper"] - channel["lower"])
    if smoothing_span is None:
        return raw
    smoothing = ensure_positive_int(smoothing_span, "smoothing_span")
    return ewma(raw, span=smoothing)
