# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Frame-returning indicators are time-series functions too."""

from __future__ import annotations

import pandas as pd

__all__ = ["channel"]


def channel(values: pd.Series) -> pd.DataFrame:
    """A frame-returning time-series function: must be discovered."""
    return pd.DataFrame({"value": values})
