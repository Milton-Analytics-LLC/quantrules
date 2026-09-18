# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Validation helpers applied at every public boundary.

These helpers exist so that a bad parameter or a malformed index fails loudly at
the edge of the library, with the offending parameter named, rather than
producing silent `NaN`s several layers deeper.

quantrules never mutates an input series. `ensure_series` returns the caller's
object unchanged when it already satisfies the contract, and a converted copy
otherwise.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from quantrules._typing import FloatSeries
from quantrules.exceptions import ConfigurationError, DataValidationError

__all__ = [
    "ensure_aligned",
    "ensure_ddof",
    "ensure_finite",
    "ensure_fraction",
    "ensure_non_negative",
    "ensure_positive",
    "ensure_positive_int",
    "ensure_series",
]

_REAL_TYPES = (int, float, np.integer, np.floating)

# Alignment is only meaningful once there is something to align against.
_MIN_SERIES_TO_COMPARE = 2


def _as_finite_float(value: Any, name: str) -> float:  # noqa: ANN401
    """Coerce ``value`` to a finite float or raise naming ``name``."""
    if isinstance(value, bool) or not isinstance(value, _REAL_TYPES):
        message = f"{name} must be a real number, got {type(value).__name__}"
        raise ConfigurationError(message)
    number = float(value)
    if not math.isfinite(number):
        message = f"{name} must be finite, got {value!r}"
        raise ConfigurationError(message)
    return number


def ensure_finite(value: float, name: str) -> float:
    """Return ``value`` as a finite float, rejecting `NaN` and infinities.

    Args:
        value: The value to check.
        name: Parameter name, used in the error message.

    Returns:
        ``value`` as a finite float.

    Raises:
        ConfigurationError: If ``value`` is not a finite real number.
    """
    return _as_finite_float(value, name)


def ensure_positive(value: float, name: str) -> float:
    """Return ``value`` as a float, requiring it to be strictly positive.

    Args:
        value: The value to check.
        name: Parameter name, used in the error message.

    Returns:
        ``value`` as a finite float.

    Raises:
        ConfigurationError: If ``value`` is not a finite real number greater
            than zero.
    """
    number = _as_finite_float(value, name)
    if number <= 0.0:
        message = f"{name} must be greater than 0, got {number}"
        raise ConfigurationError(message)
    return number


def ensure_non_negative(value: float, name: str) -> float:
    """Return ``value`` as a float, requiring it to be zero or greater.

    Args:
        value: The value to check.
        name: Parameter name, used in the error message.

    Returns:
        ``value`` as a finite float.

    Raises:
        ConfigurationError: If ``value`` is not a finite real number of zero or more.
    """
    number = _as_finite_float(value, name)
    if number < 0.0:
        message = f"{name} must be 0 or greater, got {number}"
        raise ConfigurationError(message)
    return number


def ensure_fraction(value: float, name: str) -> float:
    """Return ``value`` as a float, requiring it to lie in the closed interval [0, 1].

    Args:
        value: The value to check.
        name: Parameter name, used in the error message.

    Returns:
        ``value`` as a finite float.

    Raises:
        ConfigurationError: If ``value`` lies outside [0, 1].
    """
    number = _as_finite_float(value, name)
    if not 0.0 <= number <= 1.0:
        message = f"{name} must lie between 0 and 1 inclusive, got {number}"
        raise ConfigurationError(message)
    return number


def ensure_positive_int(value: int, name: str) -> int:
    """Return ``value`` as an int, requiring it to be strictly positive.

    `bool` is rejected even though it subclasses `int`, because passing `True`
    where a span is expected is always a mistake.

    Args:
        value: The value to check.
        name: Parameter name, used in the error message.

    Returns:
        ``value`` as an int.

    Raises:
        ConfigurationError: If ``value`` is not an integer greater than zero.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        message = f"{name} must be an integer, got {type(value).__name__}"
        raise ConfigurationError(message)
    number = int(value)
    if number <= 0:
        message = f"{name} must be greater than 0, got {number}"
        raise ConfigurationError(message)
    return number


def ensure_ddof(ddof: int, window: int, name: str = "ddof") -> int:
    """Validate delta degrees of freedom against a window size.

    A rolling standard deviation over ``window`` observations needs
    ``ddof`` strictly below ``window``; otherwise every window has no degrees of
    freedom left and the result is silently all `NaN`.

    Args:
        ddof: Delta degrees of freedom to check.
        window: The window size ``ddof`` will be used with.
        name: Parameter name, used in the error message.

    Returns:
        ``ddof`` as an int.

    Raises:
        ConfigurationError: If ``ddof`` is not an integer of zero or more that is
            strictly less than ``window``.
    """
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)):
        message = f"{name} must be an integer, got {type(ddof).__name__}"
        raise ConfigurationError(message)
    value = int(ddof)
    if value < 0:
        message = f"{name} must be 0 or greater, got {value}"
        raise ConfigurationError(message)
    if value >= window:
        message = f"{name} ({value}) must be less than window ({window})"
        raise ConfigurationError(message)
    return value


def ensure_series(values: object, name: str) -> FloatSeries:
    """Validate an input series against the quantrules data contract.

    A conforming series is a `pandas.Series` of numeric dtype indexed by a
    unique, monotonically increasing `pandas.DatetimeIndex`. Missing values are
    allowed: real price histories have gaps.

    Args:
        values: The object to validate. Deliberately untyped: this function is
            the boundary at which an unknown object becomes a known series.
        name: Parameter name, used in the error message.

    Returns:
        The series as `float64`. The caller's object is returned unchanged when
        it is already `float64`; otherwise a converted copy is returned. The
        input is never mutated.

    Raises:
        DataValidationError: If ``values`` violates any part of the contract.
    """
    if not isinstance(values, pd.Series):
        message = f"{name} must be a pandas Series, got {type(values).__name__}"
        raise DataValidationError(message)

    index = values.index
    if not isinstance(index, pd.DatetimeIndex):
        message = f"{name} must be indexed by a DatetimeIndex, got {type(index).__name__}"
        raise DataValidationError(message)
    if not index.is_monotonic_increasing:
        message = f"{name} must have a monotonic increasing index"
        raise DataValidationError(message)
    if index.has_duplicates:
        message = f"{name} must not contain duplicate index entries"
        raise DataValidationError(message)
    if not is_numeric_dtype(values):
        message = f"{name} must have a numeric dtype, got {values.dtype}"
        raise DataValidationError(message)

    if values.dtype == np.float64:
        return cast("FloatSeries", values)
    return values.astype("float64")


def ensure_aligned(**named_series: FloatSeries) -> None:
    """Require every supplied series to share one index.

    Args:
        **named_series: Series to compare, keyed by parameter name. The first
            keyword is treated as the reference.

    Raises:
        DataValidationError: If any series has an index differing from the first.
    """
    items = list(named_series.items())
    if len(items) < _MIN_SERIES_TO_COMPARE:
        return

    reference_name, reference = items[0]
    for other_name, other in items[1:]:
        if not reference.index.equals(other.index):
            message = (
                f"{other_name} must be indexed identically to {reference_name} "
                f"({len(other)} rows vs {len(reference)})"
            )
            raise DataValidationError(message)
