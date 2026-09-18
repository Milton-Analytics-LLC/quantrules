# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.trend."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.exceptions import ConfigurationError
from quantrules.indicators.trend import ewma_crossover, macd, rolling_slope
from tests.support import random_walk, series


class TestMacd:
    def test_matches_hand_computed_emas(self) -> None:
        """fast=1 makes the fast EMA the price; slow=2 and signal=2 are span-2 EMAs."""
        result = macd(series([1.0, 2.0, 3.0]), fast=1, slow=2, signal=2)

        assert result["line"].iloc[0] == 0.0
        assert math.isclose(result["line"].iloc[1], 1.0 / 3.0)
        assert math.isclose(result["line"].iloc[2], 4.0 / 9.0)
        assert math.isclose(result["signal"].iloc[1], 2.0 / 9.0)
        assert math.isclose(result["signal"].iloc[2], 10.0 / 27.0)
        assert math.isclose(result["histogram"].iloc[1], 1.0 / 9.0)
        assert math.isclose(result["histogram"].iloc[2], 2.0 / 27.0)

    def test_histogram_is_exactly_line_minus_signal(self) -> None:
        result = macd(random_walk(80, seed=3))

        expected = (result["line"] - result["signal"]).to_numpy()
        assert np.allclose(result["histogram"].to_numpy(), expected, equal_nan=True)

    def test_has_the_documented_columns_and_index(self) -> None:
        price = random_walk(30, seed=4)

        result = macd(price)

        assert list(result.columns) == ["line", "signal", "histogram"]
        pd.testing.assert_index_equal(result.index, price.index)

    def test_constant_price_gives_zero(self) -> None:
        result = macd(series([5.0] * 20))

        assert np.allclose(result.to_numpy(), 0.0)


class TestEwmaCrossover:
    def test_matches_hand_computed_difference(self) -> None:
        """fast=1 (identity) minus a span-2 EMA of [1,2,3] gives 0, 1/3, 4/9."""
        result = ewma_crossover(series([1.0, 2.0, 3.0]), fast=1, slow=2)

        assert result.iloc[0] == 0.0
        assert math.isclose(result.iloc[1], 1.0 / 3.0)
        assert math.isclose(result.iloc[2], 4.0 / 9.0)

    def test_constant_price_gives_zero(self) -> None:
        result = ewma_crossover(series([7.0] * 15), fast=4, slow=16)

        assert np.allclose(result.to_numpy(), 0.0)

    def test_positive_when_the_series_rises(self) -> None:
        rising = series([float(i) for i in range(30)])

        result = ewma_crossover(rising, fast=4, slow=16)

        assert result.iloc[-1] > 0.0


class TestRollingSlope:
    def test_recovers_the_slope_of_a_line(self) -> None:
        result = rolling_slope(series([0.0, 2.0, 4.0, 6.0, 8.0]), window=3)

        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert math.isclose(result.iloc[2], 2.0)
        assert math.isclose(result.iloc[3], 2.0)
        assert math.isclose(result.iloc[4], 2.0)

    def test_matches_hand_computed_slope(self) -> None:
        """x=[0,1,2], y=[1,3,2]: slope = (7 - 1*6) / 2 = 0.5."""
        result = rolling_slope(series([1.0, 3.0, 2.0]), window=3)

        assert math.isclose(result.iloc[2], 0.5)

    def test_preserves_the_index(self) -> None:
        price = random_walk(40, seed=6)

        result = rolling_slope(price, window=10)

        pd.testing.assert_index_equal(result.index, price.index)

    def test_returns_all_nan_when_shorter_than_the_window(self) -> None:
        result = rolling_slope(series([1.0, 2.0]), window=5)

        assert result.isna().all()
        assert len(result) == 2

    def test_rejects_a_window_below_two(self) -> None:
        """A slope needs at least two points; a one-point window has no x-variance."""
        with pytest.raises(ConfigurationError, match="window"):
            rolling_slope(series([1.0, 2.0, 3.0]), window=1)

    @given(slope=st.floats(min_value=-50.0, max_value=50.0))
    def test_recovers_any_linear_slope(self, slope: float) -> None:
        line = series([slope * i + 3.0 for i in range(20)])

        result = rolling_slope(line, window=5)

        assert np.allclose(result.dropna().to_numpy(), slope)
