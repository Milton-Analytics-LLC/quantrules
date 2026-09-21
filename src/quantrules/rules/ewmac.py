# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""The EWMAC trend rule: a volatility-normalised moving-average crossover.

Container-free maths. Causal: the value at time *t* uses only observations up to
and including *t*.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_positive_int, ensure_series
from quantrules.indicators.trend import ewma_crossover

__all__ = ["ewmac"]


def ewmac(
    price: FloatSeries,
    *,
    fast_span: int,
    slow_span: int,
    volatility: FloatSeries,
) -> FloatSeries:
    r"""Raw EWMAC forecast: a fast-minus-slow EWMA crossover divided by volatility.

    Carver's exponentially weighted moving-average crossover rule. The
    unscaled crossover
    [`ewma_crossover`][quantrules.indicators.trend.ewma_crossover] is in price
    units, so dividing it by the price's volatility (also in price units) gives a
    volatility-normalised, roughly unit-free forecast that is comparable across
    instruments and over time. The result is *raw*: scaling to a target average
    absolute forecast and capping happen in `quantrules.forecasts`.

    $$\text{ewmac}_t = \frac{\mathrm{EWMA}_{\text{fast}}(p)_t
      - \mathrm{EWMA}_{\text{slow}}(p)_t}{\sigma_t}$$

    ``volatility`` is supplied by the caller in the same (price) units as the
    crossover; it is not annualised or converted here. A common choice is a
    trailing standard deviation of price changes.

    Worked example: for ``price = [1, 2, 3]`` with ``fast_span=1`` (an identity
    EWMA) and ``slow_span=2`` the crossover is ``0, 1/3, 4/9``; dividing by a flat
    volatility of ``2`` gives ``0, 1/6, 2/9``.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), the
    EWMAC trading rule.

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        fast_span: Span of the fast EWMA.
        slow_span: Span of the slow EWMA.
        volatility: Price volatility, in price units, indexed identically to
            ``price``.

    Returns:
        The raw EWMAC forecast, carrying ``price``'s index.

    Raises:
        ConfigurationError: If either span is not a positive integer.
        DataValidationError: If ``price`` and ``volatility`` are not indexed
            identically.
    """
    prices = ensure_series(price, "price")
    sigma = ensure_series(volatility, "volatility")
    fast = ensure_positive_int(fast_span, "fast_span")
    slow = ensure_positive_int(slow_span, "slow_span")
    ensure_aligned(price=prices, volatility=sigma)
    crossover = ewma_crossover(prices, fast=fast, slow=slow)
    return crossover / sigma
