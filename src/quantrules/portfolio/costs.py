# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Trading costs and turnover.

Turnover measures how much a position moves relative to its own average size;
multiplied by a per-trade cost it gives the risk-adjusted drag a strategy pays.
`position_turnover` is a causal series; `average_turnover` and `trading_cost` are
whole-sample scalars.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_non_negative, ensure_positive_int, ensure_series
from quantrules.defaults import BUSINESS_DAYS_PER_YEAR
from quantrules.exceptions import DataValidationError

__all__ = ["average_turnover", "position_turnover", "trading_cost"]


def position_turnover(
    positions: FloatSeries,
    *,
    window: int | None = None,
    min_periods: int | None = None,
) -> FloatSeries:
    r"""Per-period turnover: the position change relative to the average position.

    $$\text{turnover}_t = \frac{|p_t - p_{t-1}|}{\operatorname{mean}(|p|)_t}$$

    where the denominator is a causal expanding (default) or trailing average of the
    absolute position, so the ratio uses only past data and warms up to `NaN`.

    Worked example: positions ``[1, 1, 2, 2, 1]`` have absolute changes
    ``[nan, 0, 1, 0, 1]`` and an expanding mean absolute position of
    ``[1, 1, 4/3, 3/2, 7/5]``, so the turnover is ``[nan, 0, 0.75, 0, 5/7]``.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), turnover.

    Args:
        positions: A position series on a unique, monotonic `DatetimeIndex`.
        window: Trailing-window length for the average position. `None` uses an
            expanding window.
        min_periods: Observations required before an average is emitted.

    Returns:
        The per-period turnover, carrying ``positions``'s index.
    """
    validated = ensure_series(positions, "positions")
    changes = validated.diff().abs()
    magnitude = validated.abs()
    if window is None:
        floor = 1 if min_periods is None else ensure_positive_int(min_periods, "min_periods")
        average = magnitude.expanding(min_periods=floor).mean()
    else:
        length = ensure_positive_int(window, "window")
        floor = length if min_periods is None else ensure_positive_int(min_periods, "min_periods")
        average = magnitude.rolling(window=length, min_periods=floor).mean()
    return changes / average


def average_turnover(
    positions: FloatSeries,
    *,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
) -> float:
    r"""Annualised average turnover over the whole sample.

    $$\overline{\text{turnover}} = \frac{\operatorname{mean}(|\Delta p|)}
      {\operatorname{mean}(|p|)} \times \text{periods\_per\_year}$$

    Worked example: positions ``[1, 1, 2, 2, 1]`` give a mean absolute change of
    ``0.5`` and a mean absolute position of ``1.4``, so over 256 periods the average
    turnover is :math:`0.5 / 1.4 \times 256 \approx 91.4`.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), turnover.

    Args:
        positions: A position series on a unique, monotonic `DatetimeIndex`.
        periods_per_year: Periods per year used to annualise.

    Returns:
        The annualised average turnover.

    Raises:
        DataValidationError: If every position is zero, so turnover is undefined.
    """
    validated = ensure_series(positions, "positions")
    periods = ensure_positive_int(periods_per_year, "periods_per_year")
    mean_change = float(validated.diff().abs().mean())
    mean_magnitude = float(validated.abs().mean())
    if mean_magnitude == 0.0:
        message = "positions must contain a non-zero position to measure turnover"
        raise DataValidationError(message)
    return mean_change / mean_magnitude * periods


def trading_cost(
    positions: FloatSeries,
    cost_per_trade: float,
    *,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
) -> float:
    r"""Annualised trading cost, in Sharpe-ratio units.

    $$c_{\text{SR}} = \overline{\text{turnover}} \times \text{cost\_per\_trade}$$

    where ``cost_per_trade`` is the risk-adjusted cost of one full turnover of the
    average position, in annualised Sharpe-ratio units.

    Worked example: an average turnover of ``91.4`` at a ``0.001`` cost per trade
    costs :math:`\approx 0.091` Sharpe-ratio units a year.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015),
    risk-adjusted trading costs.

    Args:
        positions: A position series on a unique, monotonic `DatetimeIndex`.
        cost_per_trade: Risk-adjusted cost of one full turnover, in Sharpe-ratio
            units. `0` disables costs.
        periods_per_year: Periods per year used to annualise turnover.

    Returns:
        The annualised trading cost, in Sharpe-ratio units.

    Raises:
        DataValidationError: If every position is zero, so turnover is undefined.
    """
    rate = ensure_non_negative(cost_per_trade, "cost_per_trade")
    return average_turnover(positions, periods_per_year=periods_per_year) * rate
