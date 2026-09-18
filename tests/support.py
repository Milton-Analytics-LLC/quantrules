# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic data builders shared across the test suite.

Nothing here touches the network or the filesystem. Every generator takes an
explicit ``seed`` so that a failing test reproduces exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START = "2020-01-01"


def business_days(periods: int, start: str = DEFAULT_START) -> pd.DatetimeIndex:
    """Return ``periods`` consecutive business days starting at ``start``."""
    return pd.date_range(start=start, periods=periods, freq="B", name="date")


def series(values: list[float], start: str = DEFAULT_START) -> pd.Series:
    """Build a float series on a business-day index from explicit ``values``."""
    return pd.Series(values, index=business_days(len(values), start), dtype="float64")


def random_walk(
    periods: int,
    *,
    seed: int,
    annual_drift: float = 0.0,
    annual_vol: float = 0.20,
    start_price: float = 100.0,
    periods_per_year: int = 256,
    start: str = DEFAULT_START,
) -> pd.Series:
    """Build a geometric random walk with a known drift and volatility.

    Args:
        periods: Number of observations.
        seed: Seed for ``numpy.random.default_rng``; identical seeds give
            identical output.
        annual_drift: Annualised log drift.
        annual_vol: Annualised volatility of log returns.
        start_price: Price at the first observation.
        periods_per_year: Periods used to de-annualise drift and volatility.
        start: First date on the index.

    Returns:
        A float price series on a business-day index.
    """
    rng = np.random.default_rng(seed)
    step_vol = annual_vol / np.sqrt(periods_per_year)
    step_drift = annual_drift / periods_per_year
    shocks = rng.normal(loc=step_drift, scale=step_vol, size=periods)
    shocks[0] = 0.0
    prices = start_price * np.exp(np.cumsum(shocks))
    return pd.Series(prices, index=business_days(periods, start), dtype="float64")


def trending_walk(periods: int, *, seed: int, annual_drift: float = 0.30) -> pd.Series:
    """Build a random walk with a strong upward drift, for trend-following tests."""
    return random_walk(periods, seed=seed, annual_drift=annual_drift)
