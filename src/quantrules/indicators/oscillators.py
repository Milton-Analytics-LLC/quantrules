# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Bounded momentum oscillators.

Currently the relative strength index. Causal.
"""

from __future__ import annotations

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_positive_int, ensure_series
from quantrules.exceptions import ConfigurationError

__all__ = ["rsi"]

_RSI_MAX = 100.0


def rsi(price: FloatSeries, *, window: int = 14, method: str = "wilder") -> FloatSeries:
    r"""Relative strength index, bounded in [0, 100].

    Splits each step into a gain and a loss, smooths each over ``window`` steps,
    and reports the share of movement that was upward:

    $$\mathrm{RSI}_t = 100 \cdot
      \frac{\overline{\text{gain}}_t}{\overline{\text{gain}}_t + \overline{\text{loss}}_t}$$

    - ``"wilder"``: Wilder's smoothing, an exponential average with factor
      :math:`\alpha = 1 / n`.
    - ``"simple"``: a simple moving average of the gains and losses.

    Worked example: prices ``[1, 2, 3, 4, 3, 2]`` with ``window=2`` and the
    simple method give an RSI of ``100`` while every step is a gain, ``50`` at
    the bar with one gain and one loss of equal size, and ``0`` once both steps
    are losses.

    Reference: J. Welles Wilder Jr., *New Concepts in Technical Trading Systems*
    (1978).

    Args:
        price: Price series on a unique, monotonic `DatetimeIndex`.
        window: Number of steps to smooth over.
        method: ``"wilder"`` or ``"simple"``.

    Returns:
        The RSI, carrying ``price``'s index with `NaN` through warmup.

    Raises:
        ConfigurationError: If ``method`` is not ``"wilder"`` or ``"simple"``.
    """
    validated = ensure_series(price, "price")
    length = ensure_positive_int(window, "window")
    delta = validated.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    if method == "wilder":
        avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    elif method == "simple":
        avg_gain = gain.rolling(window=length, min_periods=length).mean()
        avg_loss = loss.rolling(window=length, min_periods=length).mean()
    else:
        message = f"method must be 'wilder' or 'simple', got {method!r}"
        raise ConfigurationError(message)
    return _RSI_MAX * avg_gain / (avg_gain + avg_loss)
