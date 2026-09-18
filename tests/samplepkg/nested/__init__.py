# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""A subpackage, to prove the walker recurses."""

from __future__ import annotations

import pandas as pd

__all__ = ["nested_series_function"]


def nested_series_function(values: pd.Series) -> pd.Series:
    """Lives one level down: must be discovered."""
    return values
