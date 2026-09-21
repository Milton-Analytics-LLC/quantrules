# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared input validation for the forecast-combination functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from quantrules._validation import ensure_non_negative
from quantrules.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

_WEIGHT_SUM_TOLERANCE = 1e-6


def aligned_weights(weights: Mapping[str, float], columns: Sequence[str]) -> NDArray[np.float64]:
    """Return the weights as an array in ``columns`` order.

    Args:
        weights: Forecast weights keyed by column name.
        columns: The forecast columns, in order.

    Returns:
        The weights as a `float64` array aligned to ``columns``.

    Raises:
        ConfigurationError: If the keys do not match ``columns`` exactly, any
            weight is negative, or the weights do not sum to 1.
    """
    if set(weights) != set(columns):
        message = (
            f"weights keys {sorted(weights)} must match the forecast columns {sorted(columns)}"
        )
        raise ConfigurationError(message)
    values = np.array(
        [ensure_non_negative(weights[column], f"weights[{column!r}]") for column in columns],
        dtype="float64",
    )
    total = float(values.sum())
    if not math.isclose(total, 1.0, abs_tol=_WEIGHT_SUM_TOLERANCE):
        message = f"weights must sum to 1, got {total}"
        raise ConfigurationError(message)
    return values
