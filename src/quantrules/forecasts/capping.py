# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Capping a scaled forecast to a symmetric bound.

Causal: the value at time *t* uses only the observation at *t*.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_positive, ensure_series
from quantrules.defaults import FORECAST_CAP
from quantrules.indicators.normalize import clip

__all__ = ["cap"]


def cap(scaled: FloatSeries, *, cap: float = FORECAST_CAP) -> FloatSeries:
    r"""Clip a scaled forecast to the symmetric interval ``[-cap, cap]``.

    Once a forecast is scaled to a target average absolute value, an unusually
    extreme reading is clipped so that one signal cannot dominate a position:

    $$\text{capped}_t = \operatorname{clip}(f_t,\ -\text{cap},\ \text{cap})$$

    The default cap is twice the target average absolute forecast.

    Worked example: ``[-30, -10, 0, 10, 30]`` capped at ``20`` becomes
    ``[-20, -10, 0, 10, 20]``.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), forecast
    capping; default [`FORECAST_CAP`][quantrules.defaults.FORECAST_CAP].

    Args:
        scaled: The scaled forecast on a unique, monotonic `DatetimeIndex`.
        cap: The absolute value at which to clip. Must be positive.

    Returns:
        The capped forecast, carrying ``scaled``'s index.

    Raises:
        ConfigurationError: If ``cap`` is not positive.
    """
    validated = ensure_series(scaled, "scaled")
    limit = ensure_positive(cap, "cap")
    return clip(validated, lower=-limit, upper=limit)
