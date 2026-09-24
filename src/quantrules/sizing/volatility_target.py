# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Cash volatility targets.

The volatility target says how much risk the whole system may take, as an
annualised standard deviation of returns, converted into a cash amount. These
helpers are the primitive-typed twins of the
[`VolatilityTargetConfig`][quantrules.config.VolatilityTargetConfig] properties,
for callers who work with plain numbers rather than a config object. They return
scalars, so there is no time series and nothing to prove causal.
"""

from __future__ import annotations

import math

from quantrules._validation import ensure_positive, ensure_positive_int
from quantrules.defaults import ANNUAL_VOL_TARGET, BUSINESS_DAYS_PER_YEAR

__all__ = ["annual_cash_vol_target", "periodic_cash_vol_target"]


def annual_cash_vol_target(
    *,
    capital: float,
    annual_vol_target: float = ANNUAL_VOL_TARGET,
) -> float:
    r"""Annual cash volatility target.

    The capital multiplied by the fractional annual volatility target:

    $$V^{\text{annual}} = C \cdot \tau$$

    Worked example: a capital of ``1_000_000`` at a ``0.25`` target gives
    :math:`1{,}000{,}000 \cdot 0.25 = 250{,}000`.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), the
    volatility target; see
    [`ANNUAL_VOL_TARGET`][quantrules.defaults.ANNUAL_VOL_TARGET].

    Args:
        capital: Account capital, in the account currency.
        annual_vol_target: Fractional annualised volatility target (``0.25`` is
            25% per year).

    Returns:
        The annual cash volatility target.
    """
    money = ensure_positive(capital, "capital")
    target = ensure_positive(annual_vol_target, "annual_vol_target")
    return money * target


def periodic_cash_vol_target(
    *,
    capital: float,
    annual_vol_target: float = ANNUAL_VOL_TARGET,
    periods_per_year: int = BUSINESS_DAYS_PER_YEAR,
) -> float:
    r"""Per-period cash volatility target.

    The annual cash volatility target de-annualised to one period:

    $$V_{\text{target}} = \frac{C \cdot \tau}{\sqrt{Y}}$$

    Worked example: with a capital of ``1_000_000``, a ``0.25`` target and
    ``256`` periods per year, this is :math:`250{,}000 / \sqrt{256} = 15{,}625`.

    Reference: Robert Carver, *Systematic Trading* (Harriman House, 2015), the
    volatility target.

    Args:
        capital: Account capital, in the account currency.
        annual_vol_target: Fractional annualised volatility target.
        periods_per_year: Periods per year used to de-annualise.

    Returns:
        The per-period cash volatility target.
    """
    yearly = annual_cash_vol_target(capital=capital, annual_vol_target=annual_vol_target)
    periods = ensure_positive_int(periods_per_year, "periods_per_year")
    return yearly / math.sqrt(periods)
