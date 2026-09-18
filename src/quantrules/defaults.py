# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Default constants used across quantrules.

Every default lives here, once, so that a system built on quantrules can state
exactly where its parameters came from. Nothing in the library reads these
constants implicitly: they are the defaults on configuration objects and
function signatures, and every one of them can be overridden.

The conventions follow Robert Carver's *Systematic Trading* (Harriman House,
2015). Where a value is a quantrules choice rather than a published convention,
the comment says so.

!!! warning "Defaults are a starting point, not a recommendation"

    These values suit a diversified, medium-speed futures system. They are not
    tuned for your instruments, your capital, or your costs.
"""

from __future__ import annotations

__all__ = [
    "ANNUAL_VOL_TARGET",
    "BUFFER_FRACTION",
    "BUSINESS_DAYS_PER_YEAR",
    "CARRY_SMOOTHING_SPAN",
    "CORRELATION_FLOOR",
    "EWMAC_SPEED_PAIRS",
    "FORECAST_CAP",
    "MAX_DIVERSIFICATION_MULTIPLIER",
    "MONTHS_PER_YEAR",
    "ROOT_BUSINESS_DAYS_PER_YEAR",
    "SCALAR_ESTIMATION_MIN_PERIODS",
    "TARGET_AVG_ABS_FORECAST",
    "VOLATILITY_EWMA_SPAN",
    "VOLATILITY_FLOOR_PERCENTILE",
    "VOLATILITY_FLOOR_WINDOW",
    "VOLATILITY_MIN_PERIODS",
    "WEEKS_PER_YEAR",
]

# --------------------------------------------------------------------------
# Annualisation
# --------------------------------------------------------------------------

BUSINESS_DAYS_PER_YEAR = 256
"""Business days in a trading year.

256 rather than 252 because its square root is exactly 16, which makes
annualising and de-annualising volatility arithmetic you can do in your head.
"""

ROOT_BUSINESS_DAYS_PER_YEAR = 16.0
"""Square root of [`BUSINESS_DAYS_PER_YEAR`][quantrules.defaults.BUSINESS_DAYS_PER_YEAR].

Daily volatility times this value is annualised volatility.
"""

WEEKS_PER_YEAR = 52
"""Weeks in a trading year, for weekly-sampled series."""

MONTHS_PER_YEAR = 12
"""Months in a year, for monthly-sampled series."""

# --------------------------------------------------------------------------
# Forecasts
# --------------------------------------------------------------------------

TARGET_AVG_ABS_FORECAST = 10.0
"""Target average absolute value of a scaled forecast.

Forecasts are scaled so that their long-run average absolute value is this
number. Fixing it makes forecasts from different rules directly comparable and
lets forecast weights be a simple weighted average.
"""

FORECAST_CAP = 20.0
"""Absolute value at which a scaled forecast is clipped.

Twice [`TARGET_AVG_ABS_FORECAST`][quantrules.defaults.TARGET_AVG_ABS_FORECAST].
Capping limits the position an unusually extreme signal can produce.
"""

SCALAR_ESTIMATION_MIN_PERIODS = 256
"""Minimum observations before an estimated forecast scalar is emitted.

Roughly one year of business days. Before this point the estimate is `NaN`
rather than a number computed from too little history.
"""

# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------

ANNUAL_VOL_TARGET = 0.25
"""Target annualised volatility of the whole system's returns, as a fraction.

0.25 means a 25% annualised standard deviation of returns.
"""

VOLATILITY_EWMA_SPAN = 36
"""Span of the exponentially weighted estimate of return volatility.

A span of 36 business days corresponds to a half-life of roughly 25 days:
recent volatility dominates, but the estimate is not so jumpy that positions
churn.
"""

VOLATILITY_MIN_PERIODS = 10
"""Minimum observations before a volatility estimate is emitted.

A quantrules choice. Below this the exponentially weighted estimate is
dominated by one or two returns.
"""

VOLATILITY_FLOOR_PERCENTILE = 0.05
"""Percentile of past volatility used as a floor on the current estimate.

Position size is inversely proportional to volatility, so an unusually quiet
period would otherwise produce an unboundedly large position. Flooring the
estimate at its own low percentile bounds that.
"""

VOLATILITY_FLOOR_WINDOW = 2560
"""Observations over which the volatility floor percentile is measured.

Ten years of business days. A quantrules choice: long enough that the floor
reflects a full volatility cycle rather than the recent regime.
"""

# --------------------------------------------------------------------------
# Trading rules
# --------------------------------------------------------------------------

EWMAC_SPEED_PAIRS = ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256))
"""Standard (fast span, slow span) pairs for the EWMAC crossover rule.

Each pair holds a 1:4 ratio, and each is twice the speed of the next. Together
they span horizons from a few days to most of a year. Faster pairs are usually
uneconomic after costs on all but the cheapest instruments.
"""

CARRY_SMOOTHING_SPAN = 90
"""Span used to smooth a raw carry forecast.

Carry is a slow signal measured from noisy price differences, so it is smoothed
over roughly a quarter before use.
"""

# --------------------------------------------------------------------------
# Portfolio construction
# --------------------------------------------------------------------------

MAX_DIVERSIFICATION_MULTIPLIER = 2.5
"""Upper bound applied to both the forecast and instrument diversification multipliers.

Diversification multipliers are estimated from correlations, and correlation
estimates are noisy. Capping the multiplier bounds how much an optimistic
estimate can scale the whole portfolio up.
"""

CORRELATION_FLOOR = 0.0
"""Lower bound applied to correlations before diversification maths.

Negative correlation estimates would inflate a diversification multiplier, on
the strength of a relationship unlikely to survive out of sample.
"""

BUFFER_FRACTION = 0.10
"""Half-width of the no-trade buffer, as a fraction of the average position.

The optimal position is surrounded by a buffer; trading happens only when the
current position leaves it, and then only back to the nearest edge. This trades
a small tracking error against a large reduction in turnover and cost.
"""
