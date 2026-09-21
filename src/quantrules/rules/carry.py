# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""The carry rule: a volatility-normalised, smoothed carry signal.

Container-free maths. Causal: the value at time *t* uses only observations up to
and including *t*.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_positive_int, ensure_series
from quantrules.defaults import CARRY_SMOOTHING_SPAN
from quantrules.indicators.averages import ewma

__all__ = ["carry"]


def carry(
    raw_carry: FloatSeries,
    *,
    volatility: FloatSeries,
    smoothing_span: int = CARRY_SMOOTHING_SPAN,
) -> FloatSeries:
    r"""Raw carry forecast: normalise a carry measure by volatility, then smooth it.

    Carry is the return earned from simply holding a position when prices do not
    move (a futures roll yield, an FX forward-points differential, a bond's
    roll-down). ``quantrules`` is data-agnostic, so the caller supplies the
    already-computed ``raw_carry`` in the same units as ``volatility``; the rule
    normalises it to a risk-adjusted, unit-free signal and smooths it, because
    carry measured from noisy prices is jumpy:

    $$\text{carry}_t = \mathrm{EWMA}_{\text{smoothing\_span}}
      \!\left(\frac{\text{raw\_carry}}{\sigma}\right)_t$$

    The result is *raw*: scaling to a target average absolute forecast and
    capping happen in `quantrules.forecasts`.

    Worked example: ``raw_carry = [2, 4, 6]`` over a flat volatility of ``2``
    normalises to ``[1, 2, 3]``; the span-2 EWMA of that is ``1, 5/3, 23/9``.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), the
    carry trading rule; smoothing span
    [`CARRY_SMOOTHING_SPAN`][quantrules.defaults.CARRY_SMOOTHING_SPAN].

    Args:
        raw_carry: The carry measure, in the same units as ``volatility``, on a
            unique, monotonic `DatetimeIndex`.
        volatility: Volatility to normalise by, indexed identically to
            ``raw_carry``.
        smoothing_span: Span of the EWMA that smooths the normalised carry.

    Returns:
        The raw carry forecast, carrying ``raw_carry``'s index.

    Raises:
        ConfigurationError: If ``smoothing_span`` is not a positive integer.
        DataValidationError: If ``raw_carry`` and ``volatility`` are not indexed
            identically.
    """
    numerator = ensure_series(raw_carry, "raw_carry")
    sigma = ensure_series(volatility, "volatility")
    span = ensure_positive_int(smoothing_span, "smoothing_span")
    ensure_aligned(raw_carry=numerator, volatility=sigma)
    return ewma(numerator / sigma, span=span)
