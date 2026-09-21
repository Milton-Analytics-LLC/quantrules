# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""The `rules` layer of quantrules.

The rule maths (`ewmac`, `carry`, `breakout`, `mean_reversion`) are container-free
functions, re-exported here for convenience as ``quantrules.rules.<name>`` while
their canonical public home stays the leaf module. The registry API (`Rule`,
`register_rule`, `get_rule`, `available_rules`) is the package-level public
surface. Importing this package registers the built-in rules: the six standard
EWMAC speeds and carry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantrules.defaults import EWMAC_SPEED_PAIRS
from quantrules.rules.base import Rule, available_rules, get_rule, register_rule
from quantrules.rules.breakout import breakout as breakout
from quantrules.rules.carry import carry as carry
from quantrules.rules.ewmac import ewmac as ewmac
from quantrules.rules.mean_reversion import mean_reversion as mean_reversion

if TYPE_CHECKING:
    from quantrules._typing import FloatSeries
    from quantrules.data import MarketData
    from quantrules.rules.base import RuleFunction

__all__ = ["Rule", "available_rules", "get_rule", "register_rule"]


def _make_ewmac(fast: int, slow: int) -> RuleFunction:
    def rule(data: MarketData) -> FloatSeries:
        return ewmac(data["price"], fast_span=fast, slow_span=slow, volatility=data["volatility"])

    return rule


def _carry_rule(data: MarketData) -> FloatSeries:
    return carry(data["raw_carry"], volatility=data["volatility"])


def _register_builtin_rules() -> None:
    """Register the standard EWMAC speeds and carry as named rules."""
    for fast, slow in EWMAC_SPEED_PAIRS:
        register_rule(f"ewmac{fast}_{slow}", required_fields=("price", "volatility"))(
            _make_ewmac(fast, slow)
        )
    register_rule("carry", required_fields=("raw_carry", "volatility"))(_carry_rule)


_register_builtin_rules()
