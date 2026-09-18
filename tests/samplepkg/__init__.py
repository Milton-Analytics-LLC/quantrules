# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""A miniature package used to prove the causality discovery walker works.

It deliberately mixes time-series functions with functions that are not, and
hides one of each inside a subpackage and a private module.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["exported_scalar_function", "exported_series_function"]


def exported_series_function(values: pd.Series) -> pd.Series:
    """A time-series function: must be discovered."""
    return values


def exported_scalar_function(values: pd.Series) -> float:
    """Not a time-series function: must be ignored."""
    return float(values.mean())


def unexported_series_function(values: pd.Series) -> pd.Series:
    """Absent from __all__, so not public: must be ignored."""
    return values
