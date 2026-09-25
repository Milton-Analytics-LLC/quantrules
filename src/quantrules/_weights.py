# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared validation for weight mappings.

Forecast and instrument weights are validated in one place. `aligned_weights` and
`active_weights` match a mapping against a set of columns; `weight_array` and
`positive_weight_array` validate a bare mapping with no columns to match against.
"""

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


def _non_negative_values(weights: Mapping[str, float], keys: Sequence[str]) -> NDArray[np.float64]:
    """Return ``weights`` over ``keys`` as a float64 array, each validated non-negative."""
    return np.array(
        [ensure_non_negative(weights[key], f"weights[{key!r}]") for key in keys],
        dtype="float64",
    )


def weight_array(weights: Mapping[str, float]) -> tuple[list[str], NDArray[np.float64]]:
    """Validate a bare weight mapping and return its keys and values.

    Args:
        weights: Weights keyed by name.

    Returns:
        The keys (in mapping order) and their weights as a `float64` array.

    Raises:
        ConfigurationError: If any weight is negative or they do not sum to 1.
    """
    keys = list(weights)
    values = _non_negative_values(weights, keys)
    total = float(values.sum())
    if not math.isclose(total, 1.0, abs_tol=_WEIGHT_SUM_TOLERANCE):
        message = f"weights must sum to 1, got {total}"
        raise ConfigurationError(message)
    return keys, values


def positive_weight_array(weights: Mapping[str, float]) -> tuple[list[str], NDArray[np.float64]]:
    """Validate a non-negative weight mapping with a strictly positive sum.

    Unlike `weight_array`, the weights need not sum to 1; the caller normalises
    them, so only a positive sum (a defined normalisation) is required.

    Args:
        weights: Weights keyed by name.

    Returns:
        The keys (in mapping order) and their weights as a `float64` array.

    Raises:
        ConfigurationError: If any weight is negative or the sum is not positive.
    """
    keys = list(weights)
    values = _non_negative_values(weights, keys)
    total = float(values.sum())
    if total <= 0.0:
        message = f"weights must have a positive sum, got {total}"
        raise ConfigurationError(message)
    return keys, values


def aligned_weights(weights: Mapping[str, float], columns: Sequence[str]) -> NDArray[np.float64]:
    """Return the weights as an array in ``columns`` order.

    Args:
        weights: Weights keyed by column (or instrument) name.
        columns: The columns, in order.

    Returns:
        The weights as a `float64` array aligned to ``columns``.

    Raises:
        ConfigurationError: If the keys do not match ``columns`` exactly, any
            weight is negative, or the weights do not sum to 1.
    """
    if set(weights) != set(columns):
        message = f"weights keys {sorted(weights)} must match the columns {sorted(columns)}"
        raise ConfigurationError(message)
    values = _non_negative_values(weights, list(columns))
    total = float(values.sum())
    if not math.isclose(total, 1.0, abs_tol=_WEIGHT_SUM_TOLERANCE):
        message = f"weights must sum to 1, got {total}"
        raise ConfigurationError(message)
    return values


def active_weights(
    weights: Mapping[str, float], columns: Sequence[str]
) -> tuple[list[str], NDArray[np.float64]]:
    """Validate ``weights`` and return the positive-weight columns and their weights.

    Zero-weight entries are dropped. They do not affect a weighted average or the
    diversification quadratic form mathematically, but a disabled entry is often
    missing or flat, and ``NaN * 0`` is `NaN`, so keeping it would poison the result.

    Args:
        weights: Weights keyed by column (or instrument) name.
        columns: The columns, in order.

    Returns:
        The positive-weight columns and their weights, in ``columns`` order.

    Raises:
        ConfigurationError: If ``weights`` fails `aligned_weights`' validation.
    """
    full = aligned_weights(weights, columns)
    mask = full > 0.0
    active_columns = [column for column, keep in zip(columns, mask, strict=True) if keep]
    return active_columns, full[mask]
