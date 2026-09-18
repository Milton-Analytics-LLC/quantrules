# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Normalisation primitives: z-score, clipping, rolling rank and percentile.

These map a raw series onto a comparable scale. All are causal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_positive_int, ensure_series
from quantrules.exceptions import ConfigurationError

__all__ = ["clip", "rolling_percentile", "rolling_rank", "zscore"]


def zscore(
    series: FloatSeries,
    *,
    window: int,
    ddof: int = 0,
    min_periods: int | None = None,
) -> FloatSeries:
    r"""Rolling z-score: distance from the trailing mean in trailing standard deviations.

    $$z_t = \frac{x_t - \text{mean}_t}{\text{std}_t}$$

    where the mean and standard deviation are taken over the trailing window. It
    is invariant to a positive affine transform of the input.

    Worked example: for ``[1, 2, 3, 4]`` with ``window=3`` and ``ddof=0`` the
    third point has mean 2 and standard deviation :math:`\sqrt{2/3}`, so its
    z-score is :math:`(3 - 2)/\sqrt{2/3} = \sqrt{3/2}`.

    Args:
        series: Input series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.
        ddof: Delta degrees of freedom of the standard deviation.
        min_periods: Observations required before a value is emitted. Defaults
            to ``window``.

    Returns:
        The rolling z-score, carrying ``series``'s index.
    """
    validated = ensure_series(series, "series")
    length = ensure_positive_int(window, "window")
    floor = length if min_periods is None else ensure_positive_int(min_periods, "min_periods")
    rolling = validated.rolling(window=length, min_periods=floor)
    return (validated - rolling.mean()) / rolling.std(ddof=ddof)


def clip(series: FloatSeries, *, lower: float, upper: float) -> FloatSeries:
    """Clamp every value into the closed interval ``[lower, upper]``.

    This is the primitive behind forecast capping: a scaled forecast is clipped
    to a symmetric cap so that one extreme reading cannot dominate a position.

    Worked example: ``[-3, 0, 3]`` clipped to ``[-1, 1]`` becomes ``[-1, 0, 1]``.

    Args:
        series: Input series on a unique, monotonic `DatetimeIndex`.
        lower: Lower bound.
        upper: Upper bound.

    Returns:
        The clamped series, carrying ``series``'s index.

    Raises:
        ConfigurationError: If ``lower`` exceeds ``upper``.
    """
    validated = ensure_series(series, "series")
    if lower > upper:
        message = f"lower ({lower}) must not exceed upper ({upper})"
        raise ConfigurationError(message)
    return validated.clip(lower=lower, upper=upper)


def rolling_rank(series: FloatSeries, *, window: int) -> FloatSeries:
    r"""Where the current value sits between the trailing minimum and maximum.

    $$\text{rank}_t = \frac{x_t - \min_t}{\max_t - \min_t} \in [0, 1]$$

    Zero means the current value is the window's low, one means its high. This
    generalises an implied-volatility rank.

    Worked example: for ``[1, 3, 2, 5, 4]`` with ``window=3`` the ranks are
    ``0.5`` (window ``[1, 3, 2]``), ``1.0`` (window ``[3, 2, 5]``) and ``2/3``
    (window ``[2, 5, 4]``).

    Args:
        series: Input series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.

    Returns:
        The rolling rank, carrying ``series``'s index.
    """
    validated = ensure_series(series, "series")
    length = ensure_positive_int(window, "window")
    rolling = validated.rolling(window=length, min_periods=length)
    low = rolling.min()
    high = rolling.max()
    return (validated - low) / (high - low)


def _empirical_cdf_of_last(values: NDArray[np.float64]) -> float:
    """Share of a window at or below its last value."""
    return float(np.mean(values <= values[-1]))


def rolling_percentile(series: FloatSeries, *, window: int) -> FloatSeries:
    r"""Empirical percentile of the current value within its trailing window.

    The share of the ``window`` most recent observations at or below the current
    value, in ``(0, 1]``. This generalises an implied-volatility percentile.

    Worked example: for ``[1, 3, 2, 5, 4]`` with ``window=3`` the percentiles
    are ``2/3`` (``2`` is at or above two of ``[1, 3, 2]``), ``1.0`` (``5`` is
    the largest of ``[3, 2, 5]``) and ``2/3`` (window ``[2, 5, 4]``).

    Args:
        series: Input series on a unique, monotonic `DatetimeIndex`.
        window: Number of observations in the trailing window.

    Returns:
        The rolling percentile, carrying ``series``'s index.
    """
    validated = ensure_series(series, "series")
    length = ensure_positive_int(window, "window")
    rolling = validated.rolling(window=length, min_periods=length)
    return rolling.apply(_empirical_cdf_of_last, raw=True)
