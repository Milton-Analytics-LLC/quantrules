# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Composable building blocks for systematic trading systems.

`quantrules` provides four independently usable layers:

1. **Indicators** - technical indicators over price series.
2. **Rules** - trading rules that turn indicators into raw forecasts.
3. **Forecasts** - scaling, capping, weighting and combining forecasts.
4. **Sizing and portfolio** - volatility targeting, position sizing and
   portfolio construction.

Every public function takes and returns `pandas` objects, uses only
information available at or before each timestamp, and never downloads data.

!!! warning "Not investment advice"

    This library is provided for research and educational purposes only.
    Nothing it produces is a recommendation to buy or sell any security.
    Trading involves substantial risk of loss.
"""

from quantrules import forecasts, portfolio, rules, sizing
from quantrules.data import MarketData

__version__ = "0.1.0.dev0"

__all__ = ["MarketData", "__version__", "forecasts", "portfolio", "rules", "sizing"]
