# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Combining several forecasts into one.

Causal: the value at time *t* uses only observations up to and including *t*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from quantrules._typing import FloatSeries, Frame
from quantrules._validation import ensure_aligned, ensure_frame, ensure_positive, ensure_series
from quantrules.defaults import FORECAST_CAP
from quantrules.forecasts._inputs import active_weights
from quantrules.forecasts.diversification import forecast_diversification_multiplier
from quantrules.indicators.normalize import clip

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["combine"]


def combine(
    forecasts: Frame,
    weights: Mapping[str, float],
    *,
    window: int | None = None,
    min_periods: int | None = None,
    fdm: FloatSeries | None = None,
    cap: float = FORECAST_CAP,
) -> FloatSeries:
    r"""Combine forecasts into one: a weighted average scaled up for diversification.

    The individual (scaled, capped) forecasts are averaged with ``weights``, scaled
    by the forecast diversification multiplier, then capped again so the combined
    forecast respects the same bound:

    $$\text{combined}_t = \operatorname{clip}\!\left(
      \mathrm{FDM}_t \sum_i w_i\, f_{i,t},\ -\text{cap},\ \text{cap}\right)$$

    The multiplier is estimated causally from the forecasts unless an ``fdm`` series
    is supplied.

    Worked example: two equal-weight forecasts both equal to ``[1, 2, 3, 4]`` have a
    multiplier of 1, so their combination is ``[1, 2, 3, 4]`` once the multiplier has
    warmed up.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), combining
    forecasts with forecast weights and the diversification multiplier.

    Args:
        forecasts: A frame of scaled, capped forecasts, one column per rule, on a
            unique, monotonic `DatetimeIndex`.
        weights: Forecast weights keyed by column name; non-negative and summing to 1.
        window: Trailing-window length for the multiplier's correlations. `None` uses
            an expanding window.
        min_periods: Observations required before the multiplier is emitted.
        fdm: An explicit diversification multiplier. When omitted it is estimated with
            [`forecast_diversification_multiplier`][quantrules.forecasts.diversification.forecast_diversification_multiplier].
        cap: The absolute value at which to clip the combined forecast.

    Returns:
        The combined forecast, carrying the forecasts' index.

    Raises:
        ConfigurationError: If the weights are invalid or ``cap`` is not positive.
        DataValidationError: If ``forecasts`` is not a valid frame, or ``fdm`` is not
            indexed identically to it.
    """
    frame = ensure_frame(forecasts, "forecasts")
    active_columns, weight_array = active_weights(weights, list(frame.columns))
    limit = ensure_positive(cap, "cap")
    weighted = cast("FloatSeries", frame[active_columns] @ weight_array)
    if fdm is None:
        multiplier: FloatSeries = forecast_diversification_multiplier(
            frame, weights, window=window, min_periods=min_periods
        )
    else:
        multiplier = ensure_series(fdm, "fdm")
        ensure_aligned(forecasts=weighted, fdm=multiplier)
    return clip(weighted * multiplier, lower=-limit, upper=limit)
