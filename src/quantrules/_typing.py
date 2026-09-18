# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Internal type aliases.

`pandas.Series` is only subscriptable to a type checker, so the parameterised
alias is defined under `TYPE_CHECKING` and falls back to the bare class at
runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import pandas as pd

if TYPE_CHECKING:
    FloatSeries: TypeAlias = "pd.Series[float]"
else:  # pragma: no cover - runtime fallback, exercised implicitly by every import
    FloatSeries: TypeAlias = pd.Series

Frame: TypeAlias = pd.DataFrame

__all__ = ["FloatSeries", "Frame"]
