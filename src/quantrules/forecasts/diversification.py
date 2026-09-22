# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""The forecast diversification multiplier.

Causal: the value at time *t* uses only observations up to and including *t*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantrules._typing import FloatSeries, Frame
from quantrules._validation import ensure_finite, ensure_frame, ensure_positive
from quantrules.correlation import rolling_correlation
from quantrules.defaults import CORRELATION_FLOOR, MAX_DIVERSIFICATION_MULTIPLIER
from quantrules.forecasts._inputs import active_weights

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["forecast_diversification_multiplier"]


def forecast_diversification_multiplier(
    forecasts: Frame,
    weights: Mapping[str, float],
    *,
    window: int | None = None,
    min_periods: int | None = None,
    correlation_floor: float = CORRELATION_FLOOR,
    max_multiplier: float = MAX_DIVERSIFICATION_MULTIPLIER,
) -> FloatSeries:
    r"""The multiplier that restores a combined forecast's average absolute value.

    Averaging positively correlated forecasts shrinks their combined average
    absolute value below the target, so the combination is scaled back up by

    $$\mathrm{FDM}_t = \min\!\left(\frac{1}{\sqrt{\mathbf{w}^\top C_t\, \mathbf{w}}},
      \ \text{max\_multiplier}\right)$$

    where ``w`` are the forecast weights and :math:`C_t` is the causal correlation
    matrix of the forecasts, its entries floored at ``correlation_floor`` (negative
    correlations are noise and would inflate the multiplier). The quadratic form is
    accumulated from the causal pairwise
    [`rolling_correlation`][quantrules.correlation.rolling_correlation]s, so the
    multiplier uses only past data and warms up to `NaN`. It is capped because
    correlation estimates are noisy.

    Worked example: two equally weighted forecasts with correlation 1 give
    :math:`\mathbf{w}^\top C\,\mathbf{w} = 1` and a multiplier of 1; floored to 0
    (uncorrelated or anticorrelated) it is :math:`1/\sqrt{0.5} = \sqrt{2}`.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), the
    forecast diversification multiplier; cap
    [`MAX_DIVERSIFICATION_MULTIPLIER`][quantrules.defaults.MAX_DIVERSIFICATION_MULTIPLIER].

    Args:
        forecasts: A frame of scaled forecasts, one column per rule, on a unique,
            monotonic `DatetimeIndex`.
        weights: Forecast weights keyed by column name; non-negative and summing to 1.
        window: Trailing-window length for the correlations. `None` uses an
            expanding window.
        min_periods: Observations required before a correlation is emitted.
        correlation_floor: Lower bound applied to each correlation.
        max_multiplier: Upper bound on the multiplier.

    Returns:
        The diversification multiplier, carrying the forecasts' index.

    Raises:
        ConfigurationError: If the weights are invalid, ``max_multiplier`` is not
            positive, or ``correlation_floor`` is not finite.
        DataValidationError: If ``forecasts`` is not a valid frame.
    """
    frame = ensure_frame(forecasts, "forecasts")
    columns, weight_array = active_weights(weights, list(frame.columns))
    floor = ensure_finite(correlation_floor, "correlation_floor")
    cap = ensure_positive(max_multiplier, "max_multiplier")
    quadratic = pd.Series(float(np.sum(weight_array**2)), index=frame.index, dtype="float64")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            correlation = rolling_correlation(
                frame[columns[i]], frame[columns[j]], window=window, min_periods=min_periods
            )
            floored = correlation.clip(lower=floor)
            quadratic = quadratic + 2.0 * weight_array[i] * weight_array[j] * floored
    multiplier = 1.0 / quadratic.pow(0.5)
    return multiplier.clip(upper=cap)
