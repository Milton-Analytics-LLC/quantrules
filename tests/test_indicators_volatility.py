# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.volatility."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.indicators.volatility import (
    atr,
    atr_percent,
    ewma_volatility,
    realized_volatility,
    rolling_std,
    true_range,
)
from tests.support import ohlcv, random_walk, series


class TestRollingStd:
    def test_matches_hand_computed_population_std(self) -> None:
        """[1,2,3,4] window 3, ddof 0: both full windows have variance 2/3."""
        result = rolling_std(series([1.0, 2.0, 3.0, 4.0]), window=3)

        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert math.isclose(result.iloc[2], math.sqrt(2.0 / 3.0))
        assert math.isclose(result.iloc[3], math.sqrt(2.0 / 3.0))

    def test_min_periods_shortens_the_warmup(self) -> None:
        """Window 3, min_periods 2: [1,2] has population std 0.5."""
        result = rolling_std(series([1.0, 2.0, 3.0, 4.0]), window=3, min_periods=2)

        assert math.isnan(result.iloc[0])
        assert result.iloc[1] == 0.5

    def test_is_non_negative(self) -> None:
        result = rolling_std(random_walk(60, seed=8), window=10)

        assert (result.dropna() >= 0.0).all()

    @given(scale=st.floats(min_value=0.1, max_value=100.0))
    def test_scales_with_the_data(self, scale: float) -> None:
        base = random_walk(60, seed=7)

        scaled = rolling_std(base * scale, window=10)
        reference = rolling_std(base, window=10) * scale

        assert np.allclose(scaled.to_numpy(), reference.to_numpy(), equal_nan=True)

    def test_rejects_ddof_not_below_window(self) -> None:
        with pytest.raises(ConfigurationError, match="ddof"):
            rolling_std(series([1.0, 2.0, 3.0]), window=2, ddof=2)


class TestEwmaVolatility:
    def test_matches_hand_computed_riskmetrics_recursion(self) -> None:
        """[.01,-.02,.03] span 2, ppy 1: EWMA of squares gives .0001, .0003, .0007."""
        result = ewma_volatility(
            series([0.01, -0.02, 0.03]), span=2, min_periods=1, periods_per_year=1
        )

        assert math.isclose(result.iloc[0], 0.01)
        assert math.isclose(result.iloc[1], math.sqrt(0.0003))
        assert math.isclose(result.iloc[2], math.sqrt(0.0007))

    def test_annualises_by_root_periods_per_year(self) -> None:
        per_period = ewma_volatility(
            series([0.01, -0.02, 0.03]), span=2, min_periods=1, periods_per_year=1
        )
        annualised = ewma_volatility(
            series([0.01, -0.02, 0.03]), span=2, min_periods=1, periods_per_year=4
        )

        assert math.isclose(annualised.iloc[2], per_period.iloc[2] * 2.0)

    def test_is_non_negative(self) -> None:
        result = ewma_volatility(random_walk(60, seed=11).pct_change(fill_method=None), span=36)

        assert (result.dropna() >= 0.0).all()


class TestRealizedVolatility:
    def test_matches_hand_computed_log_return_std(self) -> None:
        """Prices 1, e, 1, e give log returns 1, -1, 1; a two-day sample std is sqrt(2)."""
        prices = series([1.0, math.e, 1.0, math.e])

        result = realized_volatility(prices, window=2, periods_per_year=1)

        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert math.isclose(result.iloc[2], math.sqrt(2.0))
        assert math.isclose(result.iloc[3], math.sqrt(2.0))

    def test_annualises_by_root_periods_per_year(self) -> None:
        prices = series([1.0, math.e, 1.0, math.e])

        result = realized_volatility(prices, window=2, periods_per_year=4)

        assert math.isclose(result.iloc[2], 2.0 * math.sqrt(2.0))

    def test_is_non_negative(self) -> None:
        result = realized_volatility(random_walk(120, seed=12), window=21)

        assert (result.dropna() >= 0.0).all()

    def test_rejects_non_positive_prices(self) -> None:
        """Log returns need positive prices; a zero or negative price is rejected."""
        with pytest.raises(DataValidationError, match="positive"):
            realized_volatility(series([100.0, 0.0, 101.0]), window=2)

    def test_rejects_ddof_not_below_window(self) -> None:
        with pytest.raises(ConfigurationError, match="ddof"):
            realized_volatility(series([100.0, 101.0]), window=1)


class TestTrueRange:
    def test_matches_hand_computed_ranges(self) -> None:
        """First bar is high-low; later bars take the widest of the three ranges."""
        high = series([10.0, 12.0, 11.0])
        low = series([8.0, 9.0, 7.0])
        close = series([9.0, 11.0, 8.0])

        result = true_range(high, low, close)

        assert result.iloc[0] == 2.0
        assert result.iloc[1] == 3.0
        assert result.iloc[2] == 4.0

    def test_never_below_the_high_low_range(self) -> None:
        frame = ohlcv(80, seed=9)

        result = true_range(frame["high"], frame["low"], frame["close"])
        span = (frame["high"] - frame["low"]).to_numpy()

        assert (result.to_numpy() >= span - 1e-9).all()

    def test_missing_high_or_low_is_unknown_not_zero(self) -> None:
        """A bar with a missing high (or low) has an unknown range, so TR is NaN there."""
        high = series([10.0, float("nan"), 11.0])
        low = series([8.0, 9.0, 7.0])
        close = series([9.0, 11.0, 8.0])

        result = true_range(high, low, close)

        assert result.iloc[0] == 2.0
        assert math.isnan(result.iloc[1])
        assert result.iloc[2] == 4.0

    def test_missing_prior_close_is_unknown_after_the_first_bar(self) -> None:
        """A later bar whose prior close is missing has an unknown gap, so TR is NaN."""
        high = series([10.0, 12.0, 11.0])
        low = series([8.0, 9.0, 7.0])
        close = series([9.0, float("nan"), 8.0])

        result = true_range(high, low, close)

        assert result.iloc[0] == 2.0
        assert result.iloc[1] == 3.0
        assert math.isnan(result.iloc[2])


class TestAtr:
    def test_wilder_matches_hand_computed_smoothing(self) -> None:
        """TR [2,3,4] window 2 (alpha 1/2): .5*3+.5*2=2.5, then .5*4+.5*2.5=3.25."""
        high = series([10.0, 12.0, 11.0])
        low = series([8.0, 9.0, 7.0])
        close = series([9.0, 11.0, 8.0])

        result = atr(high, low, close, window=2, method="wilder")

        assert math.isnan(result.iloc[0])
        assert math.isclose(result.iloc[1], 2.5)
        assert math.isclose(result.iloc[2], 3.25)

    def test_simple_matches_hand_computed_mean(self) -> None:
        """TR [2,3,4] window 2: (2+3)/2=2.5, then (3+4)/2=3.5."""
        high = series([10.0, 12.0, 11.0])
        low = series([8.0, 9.0, 7.0])
        close = series([9.0, 11.0, 8.0])

        result = atr(high, low, close, window=2, method="simple")

        assert math.isnan(result.iloc[0])
        assert math.isclose(result.iloc[1], 2.5)
        assert math.isclose(result.iloc[2], 3.5)

    def test_rejects_an_unknown_method(self) -> None:
        frame = ohlcv(5, seed=1)

        with pytest.raises(ConfigurationError, match="method"):
            atr(frame["high"], frame["low"], frame["close"], method="bogus")


class TestAtrPercent:
    def test_divides_atr_by_close(self) -> None:
        high = series([10.0, 12.0, 11.0])
        low = series([8.0, 9.0, 7.0])
        close = series([9.0, 11.0, 8.0])

        result = atr_percent(high, low, close, window=2, method="wilder")

        assert math.isnan(result.iloc[0])
        assert math.isclose(result.iloc[1], 2.5 / 11.0)
        assert math.isclose(result.iloc[2], 3.25 / 8.0)
