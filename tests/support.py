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


def ohlcv(
    periods: int,
    *,
    seed: int,
    start_price: float = 100.0,
    annual_vol: float = 0.20,
    start: str = DEFAULT_START,
) -> pd.DataFrame:
    """Build a deterministic OHLCV frame with valid bar geometry.

    ``close`` is a geometric random walk; ``open`` jitters around it and the
    high and low bracket both, so ``high >= max(open, close) >= min(open,
    close) >= low`` holds on every bar and every value stays positive. Volume
    is a positive random series.
    """
    rng = np.random.default_rng(seed)
    close = random_walk(
        periods, seed=seed, annual_vol=annual_vol, start_price=start_price, start=start
    ).to_numpy()
    open_ = close * (1.0 + rng.normal(0.0, 0.003, periods))
    top = np.maximum(open_, close)
    bottom = np.minimum(open_, close)
    high = top * (1.0 + np.abs(rng.normal(0.0, 0.003, periods)))
    low = bottom * (1.0 - np.abs(rng.normal(0.0, 0.003, periods)))
    volume = rng.uniform(1e5, 1e6, periods)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=business_days(periods, start),
        dtype="float64",
    )


def breadth_data(periods: int, *, seed: int, start: str = DEFAULT_START) -> pd.DataFrame:
    """Build a deterministic market-breadth frame.

    Advancing and declining issue counts and volumes are strictly positive (so
    ratio-of-ratio measures never divide by zero); new-high and new-low counts
    may be zero.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "advances": rng.integers(50, 500, periods),
            "declines": rng.integers(50, 500, periods),
            "advancing_volume": rng.uniform(1e6, 1e7, periods),
            "declining_volume": rng.uniform(1e6, 1e7, periods),
            "new_highs": rng.integers(0, 100, periods),
            "new_lows": rng.integers(0, 100, periods),
        },
        index=business_days(periods, start),
        dtype="float64",
    )


def crossing_pair(periods: int, *, seed: int, start: str = DEFAULT_START) -> pd.DataFrame:
    """Build two aligned random walks as ``fast`` and ``slow`` columns."""
    return pd.DataFrame(
        {
            "fast": random_walk(periods, seed=seed, start=start),
            "slow": random_walk(periods, seed=seed + 1, start=start),
        }
    )


def events_series(
    periods: int, *, seed: int, rate: float = 0.2, start: str = DEFAULT_START
) -> pd.Series:
    """Build a deterministic 0/1 event series, roughly ``rate`` of the bars firing."""
    rng = np.random.default_rng(seed)
    flags = (rng.uniform(0.0, 1.0, periods) < rate).astype(float)
    return pd.Series(flags, index=business_days(periods, start), dtype="float64")


def forecast_panel(
    periods: int, *, seed: int, columns: int = 3, start: str = DEFAULT_START
) -> pd.DataFrame:
    """Build correlated forecast series as columns ``rule_0`` .. ``rule_{columns-1}``.

    Each column is a shared factor plus idiosyncratic noise, so the columns are
    positively correlated and non-degenerate (their variances are strictly positive).
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 1.0, periods)
    data = {f"rule_{index}": common + rng.normal(0.0, 1.0, periods) for index in range(columns)}
    return pd.DataFrame(data, index=business_days(periods, start), dtype="float64")


def carry_data(periods: int, *, seed: int, start: str = DEFAULT_START) -> pd.DataFrame:
    """Build aligned ``raw_carry`` and ``volatility`` columns for the carry rule.

    ``raw_carry`` is a signed carry measure and ``volatility`` is strictly
    positive, so normalising one by the other never divides by zero.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "raw_carry": rng.normal(0.0, 1.0, periods),
            "volatility": rng.uniform(0.5, 2.0, periods),
        },
        index=business_days(periods, start),
        dtype="float64",
    )


def instrument_data(periods: int, *, seed: int, start: str = DEFAULT_START) -> pd.DataFrame:
    """Build aligned ``price`` and ``volatility`` columns for one instrument.

    ``price`` is a strictly positive random walk and ``volatility`` is a strictly
    positive fractional return volatility, so cash-volatility conversions never
    hit a zero.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "price": random_walk(periods, seed=seed, start=start),
            "volatility": pd.Series(
                rng.uniform(0.005, 0.02, periods),
                index=business_days(periods, start),
                dtype="float64",
            ),
        }
    )


def position_inputs(periods: int, *, seed: int, start: str = DEFAULT_START) -> pd.DataFrame:
    """Build aligned ``forecast`` and ``instrument_value_volatility`` columns.

    ``forecast`` is a signed forecast on roughly the +/-20 scale and
    ``instrument_value_volatility`` is a strictly positive cash volatility, so
    the position quotient never divides by zero.
    """
    rng = np.random.default_rng(seed)
    index = business_days(periods, start)
    return pd.DataFrame(
        {
            "forecast": pd.Series(rng.normal(0.0, 10.0, periods), index=index, dtype="float64"),
            "instrument_value_volatility": pd.Series(
                rng.uniform(50.0, 500.0, periods), index=index, dtype="float64"
            ),
        }
    )
