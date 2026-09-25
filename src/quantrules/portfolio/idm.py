# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""The instrument diversification multiplier.

Causal: the value at time *t* uses only observations up to and including *t*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantrules._typing import FloatSeries, Frame
from quantrules._validation import ensure_finite, ensure_frame, ensure_positive
from quantrules._weights import active_weights
from quantrules.correlation import rolling_correlation
from quantrules.defaults import CORRELATION_FLOOR, MAX_DIVERSIFICATION_MULTIPLIER

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["instrument_diversification_multiplier"]


def instrument_diversification_multiplier(
    returns: Frame,
    weights: Mapping[str, float],
    *,
    window: int | None = None,
    min_periods: int | None = None,
    correlation_floor: float = CORRELATION_FLOOR,
    max_multiplier: float = MAX_DIVERSIFICATION_MULTIPLIER,
) -> FloatSeries:
    r"""The multiplier that restores a diversified portfolio's risk to target.

    A portfolio of positively correlated instruments realises less risk than the
    weighted sum of their individual risks, so the whole book is scaled up by

    $$\mathrm{IDM}_t = \min\!\left(\frac{1}{\sqrt{\mathbf{w}^\top C_t\, \mathbf{w}}},
      \ \text{max\_multiplier}\right)$$

    where ``w`` are the instrument weights and :math:`C_t` is the causal correlation
    matrix of the instrument returns, its entries floored at ``correlation_floor``
    (negative correlations are noise and would inflate the multiplier). The quadratic
    form is accumulated from the causal pairwise
    [`rolling_correlation`][quantrules.correlation.rolling_correlation]s — the same
    construction as the forecast diversification multiplier — so it uses only past
    data and warms up to `NaN`. It is capped because correlation estimates are noisy.

    Worked example: two equally weighted instruments with return correlation 1 give
    :math:`\mathbf{w}^\top C\,\mathbf{w} = 1` and a multiplier of 1; floored to 0
    (uncorrelated or anticorrelated) it is :math:`1/\sqrt{0.5} = \sqrt{2}`.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), the
    instrument diversification multiplier; cap
    [`MAX_DIVERSIFICATION_MULTIPLIER`][quantrules.defaults.MAX_DIVERSIFICATION_MULTIPLIER].

    Args:
        returns: A frame of instrument returns, one column per instrument, on a
            unique, monotonic `DatetimeIndex`.
        weights: Instrument weights keyed by column name; non-negative, summing to 1.
        window: Trailing-window length for the correlations. `None` uses an
            expanding window.
        min_periods: Observations required before a correlation is emitted.
        correlation_floor: Lower bound applied to each correlation.
        max_multiplier: Upper bound on the multiplier.

    Returns:
        The diversification multiplier, carrying the returns' index.

    Raises:
        ConfigurationError: If the weights are invalid, ``max_multiplier`` is not
            positive, or ``correlation_floor`` is not finite.
        DataValidationError: If ``returns`` is not a valid frame.
    """
    frame = ensure_frame(returns, "returns")
    columns, weight_values = active_weights(weights, list(frame.columns))
    floor = ensure_finite(correlation_floor, "correlation_floor")
    cap = ensure_positive(max_multiplier, "max_multiplier")
    quadratic = pd.Series(float(np.sum(weight_values**2)), index=frame.index, dtype="float64")
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            correlation = rolling_correlation(
                frame[columns[i]], frame[columns[j]], window=window, min_periods=min_periods
            )
            floored = correlation.clip(lower=floor)
            quadratic = quadratic + 2.0 * weight_values[i] * weight_values[j] * floored
    multiplier = 1.0 / quadratic.pow(0.5)
    return multiplier.clip(upper=cap)
