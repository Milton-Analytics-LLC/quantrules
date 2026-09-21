# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the forecast diversification multiplier."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.forecasts.diversification import forecast_diversification_multiplier
from tests.support import business_days


def _frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=business_days(length), dtype="float64")


class TestForecastDiversificationMultiplier:
    def test_identical_forecasts_have_no_diversification(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        result = forecast_diversification_multiplier(forecasts, {"a": 0.5, "b": 0.5}, min_periods=2)
        assert math.isnan(result.iloc[0])
        assert np.allclose(result.to_numpy()[1:], 1.0)

    def test_anticorrelated_forecasts_reach_root_two(self) -> None:
        """Floored at 0, an anticorrelated equal-weight pair gives 1/sqrt(0.5) = sqrt(2)."""
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        result = forecast_diversification_multiplier(forecasts, {"a": 0.5, "b": 0.5}, min_periods=2)
        assert result.iloc[-1] == pytest.approx(math.sqrt(2.0))

    def test_is_capped_at_the_maximum(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        result = forecast_diversification_multiplier(
            forecasts, {"a": 0.5, "b": 0.5}, min_periods=2, max_multiplier=1.2
        )
        assert result.iloc[-1] == pytest.approx(1.2)

    def test_a_single_forecast_has_multiplier_one(self) -> None:
        result = forecast_diversification_multiplier(
            _frame({"a": [1.0, 2.0, 3.0, 4.0]}), {"a": 1.0}, min_periods=2
        )
        assert np.allclose(result.to_numpy(), 1.0)

    def test_is_bounded_between_one_and_the_cap(self) -> None:
        forecasts = _frame(
            {"a": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0], "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]}
        )
        result = forecast_diversification_multiplier(
            forecasts, {"a": 0.5, "b": 0.5}, min_periods=2
        ).dropna()
        assert (result >= 1.0 - 1e-9).all()
        assert (result <= 2.5 + 1e-9).all()

    def test_preserves_the_index(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        result = forecast_diversification_multiplier(forecasts, {"a": 0.5, "b": 0.5}, min_periods=2)
        pd.testing.assert_index_equal(result.index, forecasts.index)

    def test_rejects_weights_that_do_not_match_the_columns(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ConfigurationError, match="match"):
            forecast_diversification_multiplier(forecasts, {"a": 0.5, "c": 0.5})

    def test_rejects_negative_weights(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ConfigurationError, match="0 or greater"):
            forecast_diversification_multiplier(forecasts, {"a": -0.5, "b": 1.5})

    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ConfigurationError, match="sum to 1"):
            forecast_diversification_multiplier(forecasts, {"a": 0.3, "b": 0.3})

    def test_rejects_a_non_frame(self) -> None:
        with pytest.raises(DataValidationError, match="DataFrame"):
            forecast_diversification_multiplier([1.0, 2.0, 3.0], {"a": 1.0})  # type: ignore[arg-type]

    def test_rejects_a_non_positive_maximum(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ConfigurationError, match="max_multiplier"):
            forecast_diversification_multiplier(forecasts, {"a": 0.5, "b": 0.5}, max_multiplier=0.0)
