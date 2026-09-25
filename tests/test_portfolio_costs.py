# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for trading costs and turnover."""

from __future__ import annotations

import pandas as pd
import pytest

from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.portfolio.costs import average_turnover, position_turnover, trading_cost
from tests.support import series


class TestPositionTurnover:
    def test_matches_the_hand_computed_series(self) -> None:
        # positions [1,1,2,2,1]: |dp|=[nan,0,1,0,1]; expanding mean|p|=[1,1,4/3,3/2,7/5];
        # turnover = [nan, 0, 0.75, 0, 5/7].
        result = position_turnover(series([1.0, 1.0, 2.0, 2.0, 1.0]))
        pd.testing.assert_series_equal(result, series([float("nan"), 0.0, 0.75, 0.0, 5.0 / 7.0]))

    def test_a_constant_position_never_trades(self) -> None:
        result = position_turnover(series([5.0, 5.0, 5.0]))
        pd.testing.assert_series_equal(result, series([float("nan"), 0.0, 0.0]))

    def test_trailing_window_defaults_min_periods_to_the_window(self) -> None:
        # window=2, min_periods=window: mean|p| of [1,1,3,3] = [nan,1,2,3]; turnover=[nan,0,1,0].
        result = position_turnover(series([1.0, 1.0, 3.0, 3.0]), window=2)
        pd.testing.assert_series_equal(result, series([float("nan"), 0.0, 1.0, 0.0]))

    def test_trailing_window_accepts_min_periods(self) -> None:
        result = position_turnover(series([1.0, 1.0, 3.0, 3.0]), window=2, min_periods=1)
        pd.testing.assert_series_equal(result, series([float("nan"), 0.0, 1.0, 0.0]))

    def test_is_non_negative(self) -> None:
        result = position_turnover(series([1.0, -2.0, 3.0, -4.0]))
        assert (result.dropna() >= 0.0).all()

    def test_preserves_the_index(self) -> None:
        positions = series([1.0, 2.0, 3.0])
        pd.testing.assert_index_equal(position_turnover(positions).index, positions.index)

    def test_rejects_a_non_series(self) -> None:
        with pytest.raises(DataValidationError, match="Series"):
            position_turnover([1.0, 2.0, 3.0])  # type: ignore[arg-type]

    def test_rejects_a_non_positive_window(self) -> None:
        with pytest.raises(ConfigurationError, match="window"):
            position_turnover(series([1.0, 2.0]), window=0)

    def test_rejects_a_non_positive_min_periods(self) -> None:
        with pytest.raises(ConfigurationError, match="min_periods"):
            position_turnover(series([1.0, 2.0]), min_periods=0)


class TestAverageTurnover:
    def test_matches_the_hand_computed_value(self) -> None:
        # mean|dp| = 0.5, mean|p| = 1.4, annualised over 256.
        result = average_turnover(series([1.0, 1.0, 2.0, 2.0, 1.0]))
        assert result == pytest.approx(0.5 / 1.4 * 256)

    def test_a_constant_position_has_zero_turnover(self) -> None:
        assert average_turnover(series([5.0, 5.0, 5.0])) == 0.0

    def test_scales_with_periods_per_year(self) -> None:
        assert average_turnover(
            series([1.0, 1.0, 2.0, 2.0, 1.0]), periods_per_year=1
        ) == pytest.approx(0.5 / 1.4)

    def test_rejects_a_non_series(self) -> None:
        with pytest.raises(DataValidationError, match="Series"):
            average_turnover([1.0, 2.0])  # type: ignore[arg-type]

    def test_rejects_a_non_positive_periods_per_year(self) -> None:
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            average_turnover(series([1.0, 2.0]), periods_per_year=0)

    def test_rejects_an_all_zero_book(self) -> None:
        with pytest.raises(DataValidationError, match="non-zero"):
            average_turnover(series([0.0, 0.0, 0.0]))


class TestTradingCost:
    def test_matches_the_hand_computed_cost(self) -> None:
        result = trading_cost(series([1.0, 1.0, 2.0, 2.0, 1.0]), cost_per_trade=0.001)
        assert result == pytest.approx(0.5 / 1.4 * 256 * 0.001)

    def test_zero_cost_per_trade_is_free(self) -> None:
        assert trading_cost(series([1.0, 1.0, 2.0, 2.0, 1.0]), cost_per_trade=0.0) == 0.0

    def test_is_linear_in_cost_per_trade(self) -> None:
        positions = series([1.0, 1.0, 2.0, 2.0, 1.0])
        assert trading_cost(positions, cost_per_trade=0.002) == pytest.approx(
            2.0 * trading_cost(positions, cost_per_trade=0.001)
        )

    def test_rejects_a_negative_cost_per_trade(self) -> None:
        with pytest.raises(ConfigurationError, match="cost_per_trade"):
            trading_cost(series([1.0, 2.0]), cost_per_trade=-0.1)
