# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Instrument volatility for position sizing.

`instrument_volatility` is the floored volatility estimate a position is sized
against: the RiskMetrics exponentially weighted estimate from
[`ewma_volatility`][quantrules.indicators.volatility.ewma_volatility], floored at
its own trailing low percentile so an unusually quiet period cannot blow the
position up. `instrument_value_volatility` turns a fractional return volatility
into the cash volatility of one contract. Both are causal: the value at time *t*
uses only observations up to and including *t*.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from quantrules._typing import FloatSeries
from quantrules._validation import (
    ensure_aligned,
    ensure_fraction,
    ensure_positive,
    ensure_positive_int,
    ensure_series,
)
from quantrules.defaults import (
    BUSINESS_DAYS_PER_YEAR,
    VOLATILITY_EWMA_SPAN,
    VOLATILITY_FLOOR_PERCENTILE,
    VOLATILITY_FLOOR_WINDOW,
    VOLATILITY_MIN_PERIODS,
)
from quantrules.indicators.volatility import ewma_volatility

__all__ = ["instrument_value_volatility", "instrument_volatility"]


def instrument_volatility(
    returns: FloatSeries,
    *,
    span: int = VOLATILITY_EWMA_SPAN,
    min_periods: int = VOLATILITY_MIN_PERIODS,
    floor_percentile: float = VOLATILITY_FLOOR_PERCENTILE,
    floor_window: int = VOLATILITY_FLOOR_WINDOW,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
) -> FloatSeries:
    r"""Exponentially weighted volatility, floored at its trailing low percentile.

    The estimate is
    [`ewma_volatility`][quantrules.indicators.volatility.ewma_volatility] raised
    to a floor equal to its own ``floor_percentile`` quantile over the trailing
    ``floor_window`` observations:

    $$\sigma^{\text{floored}}_t =
      \max\bigl(\sigma_t,\ Q_p\{\sigma_\tau : t - w + 1 \le \tau \le t\}\bigr)$$

    A position is inversely proportional to volatility, so an unusually quiet
    period would otherwise produce an unboundedly large position; the floor
    bounds that. A ``floor_percentile`` of ``0`` disables the floor and returns
    the raw estimate. The estimate is annualised unless ``periods_per_year`` is
    ``1``; see the period-basis note on
    [`subsystem_position`][quantrules.sizing.position.subsystem_position].

    Worked example: suppose the trailing volatility window is
    ``[0.10, 0.12, 0.11, 0.09, 0.20]`` with ``floor_window=5`` and
    ``floor_percentile=0.5``; the median is ``0.11``, so the current ``0.20`` is
    left unchanged. Had the current estimate instead been ``0.08``, the window
    ``[0.10, 0.12, 0.11, 0.09, 0.08]`` has median ``0.10``, lifting it to ``0.10``.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), the
    volatility floor; see
    [`VOLATILITY_FLOOR_PERCENTILE`][quantrules.defaults.VOLATILITY_FLOOR_PERCENTILE]
    and [`VOLATILITY_FLOOR_WINDOW`][quantrules.defaults.VOLATILITY_FLOOR_WINDOW].

    Args:
        returns: A return series on a unique, monotonic `DatetimeIndex`.
        span: Span of the exponential weighting.
        min_periods: Observations required before an estimate is emitted.
        floor_percentile: Quantile of trailing volatility used as the floor, in
            ``[0, 1]``. ``0`` disables the floor.
        floor_window: Number of observations the floor percentile is measured
            over.
        periods_per_year: Periods per year used to annualise. Pass ``1`` to
            leave the estimate per-period.

    Returns:
        The floored volatility, carrying ``returns``'s index.
    """
    validated = ensure_series(returns, "returns")
    percentile = ensure_fraction(floor_percentile, "floor_percentile")
    window = ensure_positive_int(floor_window, "floor_window")
    estimate = ewma_volatility(
        validated, span=span, min_periods=min_periods, periods_per_year=periods_per_year
    )
    if percentile == 0.0:
        return estimate
    floor = estimate.rolling(window=window, min_periods=1).quantile(percentile)
    return cast("FloatSeries", np.maximum(estimate, floor))


def instrument_value_volatility(
    price: FloatSeries,
    volatility: FloatSeries,
    *,
    block_size: float = 1.0,
    exchange_rate: float = 1.0,
) -> FloatSeries:
    r"""Cash volatility of a single contract, in the account currency.

    Converts a fractional return volatility into the cash standard deviation of
    one contract's value:

    $$\sigma^{\$}_t = P_t \cdot B \cdot X \cdot \sigma_t$$

    for price :math:`P`, contract block size :math:`B` and the exchange rate
    :math:`X` from the instrument's currency to the account's. The result shares
    ``volatility``'s period basis: an annualised return volatility gives an
    annualised value volatility, a per-period one gives a per-period value
    volatility.

    Worked example: a price of ``100`` with return volatility ``0.01``, a block
    size of ``10`` and an exchange rate of ``2`` gives
    :math:`100 \cdot 10 \cdot 2 \cdot 0.01 = 20`.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015),
    instrument value volatility.

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        volatility: Fractional return volatility, indexed identically to
            ``price``.
        block_size: Contract multiplier (units of the instrument per contract).
        exchange_rate: Rate from the instrument's currency to the account's.

    Returns:
        The per-contract cash volatility, carrying the shared input index.
    """
    prices = ensure_series(price, "price")
    volatilities = ensure_series(volatility, "volatility")
    ensure_aligned(price=prices, volatility=volatilities)
    block = ensure_positive(block_size, "block_size")
    rate = ensure_positive(exchange_rate, "exchange_rate")
    return cast("FloatSeries", prices * block * rate * volatilities)
