# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Position buffering.

Trading to the exact optimal position every period is expensive. Buffering
surrounds the optimal position with a no-trade band whose half-width is a
fraction of the average position, and trades only when the held position leaves
the band, and then only back to the nearest edge. This gives up a little
tracking accuracy for a large reduction in turnover. Causal: the held position at
time *t* depends only on the held position at *t - 1* and the inputs at *t*.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantrules._typing import FloatSeries
from quantrules._validation import ensure_aligned, ensure_fraction, ensure_series
from quantrules.defaults import BUFFER_FRACTION

__all__ = ["buffer_position"]


def _run_buffer(
    optimal: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Walk the no-trade band forward, holding inside it and trading to its edge.

    The held position is carried across periods, so this recursion cannot be
    vectorised; it is a single causal left-to-right pass.
    """
    held = np.full(optimal.shape, np.nan)
    position = math.nan
    for index in range(len(optimal)):
        low = lower[index]
        high = upper[index]
        if math.isnan(low) or math.isnan(high):
            held[index] = position
            continue
        if math.isnan(position):
            position = float(optimal[index])
        elif position < low:
            position = float(low)
        elif position > high:
            position = float(high)
        held[index] = position
    return held


def buffer_position(
    optimal: FloatSeries,
    average_position: FloatSeries,
    *,
    fraction: float = BUFFER_FRACTION,
) -> FloatSeries:
    r"""Buffer a position with a no-trade band around the optimal position.

    The band half-width is a fraction of the (absolute) average position; the
    held position is the previous held position clipped into the band:

    $$b_t = \varphi\,|A_t|, \qquad
      \text{held}_t = \min\bigl(\max(\text{held}_{t-1},\ N_t - b_t),\ N_t + b_t\bigr)$$

    for optimal position :math:`N`, average position :math:`A` and fraction
    :math:`\varphi`. On the first period with a defined band the book is flat, so
    it trades straight to the optimal position; while buffering is active, a period
    whose optimal or average position is missing holds the previous position. A
    ``fraction`` of ``0`` disables buffering: the held position is the optimal
    position every period, independent of ``average_position``.

    Worked example: an average position of ``100`` and ``fraction`` ``0.1`` give a
    half-width of ``10``. Starting from optimal ``100`` (held ``100``), an optimal
    of ``105`` keeps the held ``100`` inside ``[95, 115]``; an optimal of ``130``
    pulls the lower edge to ``120``, so the held moves up to ``120``; an optimal
    of ``90`` pushes the upper edge to ``100``, so the held moves down to ``100``.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015),
    position buffering; see [`BUFFER_FRACTION`][quantrules.defaults.BUFFER_FRACTION].

    Args:
        optimal: The optimal (unbuffered) position on a unique, monotonic
            `DatetimeIndex`.
        average_position: The average position (the position at an average-strength
            forecast), indexed identically to ``optimal``.
        fraction: Half-width of the band as a fraction of the average position, in
            ``[0, 1]``. ``0`` disables buffering.

    Returns:
        The buffered (held) position, carrying the shared input index.
    """
    optimals = ensure_series(optimal, "optimal")
    averages = ensure_series(average_position, "average_position")
    ensure_aligned(optimal=optimals, average_position=averages)
    width = ensure_fraction(fraction, "fraction")
    if width == 0.0:
        # Buffering is disabled: follow the optimal position exactly, independent
        # of average_position (whose NaNs would otherwise poison a zero-width band).
        return pd.Series(optimals.to_numpy(), index=optimals.index, dtype="float64")
    half = averages.abs() * width
    lower = optimals - half
    upper = optimals + half
    held = _run_buffer(
        np.asarray(optimals, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    return pd.Series(held, index=optimals.index, dtype="float64")
