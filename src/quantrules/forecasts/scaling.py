# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Scaling a raw forecast to a target average absolute forecast.

Causal: the value at time *t* uses only observations up to and including *t*.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_positive, ensure_positive_int, ensure_series
from quantrules.defaults import SCALAR_ESTIMATION_MIN_PERIODS, TARGET_AVG_ABS_FORECAST

__all__ = ["forecast_scalar", "scale"]


def forecast_scalar(
    raw: FloatSeries,
    *,
    target: float = TARGET_AVG_ABS_FORECAST,
    window: int | None = None,
    min_periods: int = SCALAR_ESTIMATION_MIN_PERIODS,
) -> FloatSeries:
    r"""The causal forecast scalar that rescales a raw forecast to ``target``.

    The scalar is the target average absolute forecast divided by the running
    mean absolute value of the raw forecast, over an expanding window (the
    default) or a trailing window:

    $$\text{scalar}_t = \frac{\text{target}}{\operatorname{mean}(|r|)_t}$$

    Estimated per instrument from its own history; pooling across instruments is
    deferred. Because the estimate needs history, warmup is `NaN`.

    Worked example: for a raw forecast whose absolute value is a flat ``2`` the
    running mean is ``2``, so with ``target=10`` the scalar is ``5``.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), forecast
    scaling to a target average absolute forecast.

    Args:
        raw: The raw forecast on a unique, monotonic `DatetimeIndex`.
        target: Target long-run average absolute forecast.
        window: Trailing-window length for the mean absolute value. `None` uses an
            expanding window.
        min_periods: Observations required before a scalar is emitted.

    Returns:
        The forecast scalar, carrying ``raw``'s index with `NaN` through warmup.

    Raises:
        ConfigurationError: If ``target`` is not positive, or ``window``/
            ``min_periods`` is not a positive integer.
    """
    validated = ensure_series(raw, "raw")
    goal = ensure_positive(target, "target")
    floor = ensure_positive_int(min_periods, "min_periods")
    absolute = validated.abs()
    if window is None:
        mean_abs = absolute.expanding(min_periods=floor).mean()
    else:
        length = ensure_positive_int(window, "window")
        mean_abs = absolute.rolling(window=length, min_periods=floor).mean()
    return goal / mean_abs


def scale(
    raw: FloatSeries,
    *,
    target: float = TARGET_AVG_ABS_FORECAST,
    scalar: float | None = None,
    window: int | None = None,
    min_periods: int = SCALAR_ESTIMATION_MIN_PERIODS,
) -> FloatSeries:
    r"""Scale a raw forecast to a target average absolute forecast.

    Pass an explicit ``scalar`` to use a known value (for example a published
    forecast scalar for a standard rule), or omit it to estimate the scalar
    causally from the raw forecast's own history with
    [`forecast_scalar`][quantrules.forecasts.scaling.forecast_scalar]:

    $$\text{scaled}_t = r_t \cdot \text{scalar}, \qquad
      \text{scaled}_t = r_t \cdot \frac{\text{target}}{\operatorname{mean}(|r|)_t}$$

    Worked example: ``[1, -2, 3]`` with ``scalar=2`` becomes ``[2, -4, 6]``.

    Args:
        raw: The raw forecast on a unique, monotonic `DatetimeIndex`.
        target: Target average absolute forecast, used when ``scalar`` is omitted.
        scalar: An explicit forecast scalar. When given, ``target``, ``window`` and
            ``min_periods`` are ignored.
        window: Trailing-window length for the estimate. `None` uses an expanding
            window.
        min_periods: Observations required before an estimated value is emitted.

    Returns:
        The scaled forecast, carrying ``raw``'s index.

    Raises:
        ConfigurationError: If ``scalar`` or ``target`` is not positive, or
            ``window``/``min_periods`` is not a positive integer.
    """
    validated = ensure_series(raw, "raw")
    if scalar is not None:
        return validated * ensure_positive(scalar, "scalar")
    return validated * forecast_scalar(
        validated, target=target, window=window, min_periods=min_periods
    )
