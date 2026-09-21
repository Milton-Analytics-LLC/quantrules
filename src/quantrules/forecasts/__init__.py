# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""The `forecasts` layer of quantrules.

Scaling, capping, the diversification multiplier and combination, re-exported here
for convenience as ``quantrules.forecasts.<name>`` while their canonical public home
stays the leaf module.
"""

from __future__ import annotations

from quantrules.forecasts.capping import cap as cap
from quantrules.forecasts.combine import combine as combine
from quantrules.forecasts.diversification import (
    forecast_diversification_multiplier as forecast_diversification_multiplier,
)
from quantrules.forecasts.scaling import forecast_scalar as forecast_scalar
from quantrules.forecasts.scaling import scale as scale

__all__: list[str] = []
