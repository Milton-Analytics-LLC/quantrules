# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""The mean-reversion rule: bet against the recent deviation from equilibrium.

Container-free maths. Causal: the value at time *t* uses only observations up to
and including *t*.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules.indicators.normalize import zscore

__all__ = ["mean_reversion"]


def mean_reversion(price: FloatSeries, *, window: int) -> FloatSeries:
    r"""Raw mean-reversion forecast: the negated rolling z-score of the price.

    A price stretched above its trailing mean is expected to fall back, and one
    stretched below it to recover, so the forecast is the negation of the rolling
    [`zscore`][quantrules.indicators.normalize.zscore]: how many trailing standard
    deviations the price sits from its trailing mean, with the sign flipped.

    $$\text{mr}_t = -\,\frac{p_t - \text{mean}_t}{\text{std}_t}$$

    Normalising by the trailing standard deviation makes the forecast unit-free
    and invariant to a positive affine transform of the price. It is *raw*:
    scaling and capping happen in `quantrules.forecasts`.

    Worked example: for ``price = [1, 2, 3, 4]`` with ``window=3`` (population
    standard deviation) the z-score is :math:`\sqrt{3/2}` at each of the last two
    points, so the forecast is :math:`-\sqrt{3/2}`.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015); the
    z-score construction of a mean-reversion signal.

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.

    Returns:
        The raw mean-reversion forecast, carrying ``price``'s index with `NaN`
        through warmup.

    Raises:
        ConfigurationError: If ``window`` is not a positive integer.
    """
    return -zscore(price, window=window)
