# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Shared input validation for the forecast-combination functions.

The weight helpers moved to :mod:`quantrules._weights` (now shared with the portfolio
layer); they are re-exported here so the forecast layer's imports are unchanged.
"""

from __future__ import annotations

from quantrules._weights import active_weights, aligned_weights

__all__ = ["active_weights", "aligned_weights"]
