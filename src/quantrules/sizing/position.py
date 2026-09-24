# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""The subsystem position for a single instrument.

Turns a combined forecast into a position: how many contracts to hold so that a
forecast of the target average strength produces the per-period cash volatility
target. Causal: the value at time *t* uses only observations up to and including
*t*.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_positive, ensure_series
from quantrules.defaults import TARGET_AVG_ABS_FORECAST

__all__ = ["subsystem_position"]


def subsystem_position(
    forecast: FloatSeries,
    instrument_value_volatility: FloatSeries,
    *,
    cash_vol_target: float,
    target_avg_abs_forecast: float = TARGET_AVG_ABS_FORECAST,
) -> FloatSeries:
    r"""Position (in contracts) for one instrument's own subsystem.

    Scales the per-period cash volatility target by the forecast's strength
    relative to its target average, then divides by the cash volatility of one
    contract:

    $$N_t = \frac{f_t}{F} \cdot \frac{V_{\text{target}}}{\sigma^{\$}_t}$$

    for combined forecast :math:`f` (on the average-absolute scale :math:`F`),
    cash volatility target :math:`V_{\text{target}}` and per-contract cash
    volatility :math:`\sigma^{\$}` (see
    [`instrument_value_volatility`][quantrules.sizing.instrument_vol.instrument_value_volatility]).

    ``cash_vol_target`` and ``instrument_value_volatility`` must be on the same
    period basis: pair an annualised value volatility with
    [`annual_cash_vol_target`][quantrules.sizing.volatility_target.annual_cash_vol_target],
    or a per-period value volatility (from ``instrument_volatility(...,
    periods_per_year=1)``) with
    [`periodic_cash_vol_target`][quantrules.sizing.volatility_target.periodic_cash_vol_target].
    Mixing the two silently mis-sizes the position by a factor of
    :math:`\sqrt{\text{periods\_per\_year}}`. The value volatility must be
    strictly positive (a zero denominator yields a non-finite position); the
    volatility floor in
    [`instrument_volatility`][quantrules.sizing.instrument_vol.instrument_volatility]
    keeps it positive for any instrument whose returns ever move.

    Worked example: a forecast of ``10`` at :math:`F = 10`, a cash volatility
    target of ``1000`` and a per-contract value volatility of ``100`` give
    :math:`(10 / 10) \cdot (1000 / 100) = 10` contracts.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015),
    volatility scaling of the position; see
    [`TARGET_AVG_ABS_FORECAST`][quantrules.defaults.TARGET_AVG_ABS_FORECAST].

    Args:
        forecast: The combined forecast on a unique, monotonic `DatetimeIndex`.
        instrument_value_volatility: Per-contract cash volatility, indexed
            identically to ``forecast`` and strictly positive.
        cash_vol_target: The cash volatility target, on the same period basis as
            ``instrument_value_volatility`` (see the note above).
        target_avg_abs_forecast: The forecast's target average absolute value.

    Returns:
        The position in contracts, carrying the shared input index.
    """
    forecasts = ensure_series(forecast, "forecast")
    value_volatility = ensure_series(instrument_value_volatility, "instrument_value_volatility")
    ensure_aligned(forecast=forecasts, instrument_value_volatility=value_volatility)
    target = ensure_positive(cash_vol_target, "cash_vol_target")
    average = ensure_positive(target_avg_abs_forecast, "target_avg_abs_forecast")
    return forecasts * (target / average) / value_volatility
