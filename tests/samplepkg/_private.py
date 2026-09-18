# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Private modules are internal, so their contents are not walked."""

from __future__ import annotations

import pandas as pd

__all__ = ["private_series_function"]


def private_series_function(values: pd.Series) -> pd.Series:
    """Lives in a private module: must be ignored."""
    return values
