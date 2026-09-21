# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for combining forecasts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.forecasts.combine import combine
from tests.support import business_days


def _frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=business_days(length), dtype="float64")


class TestCombine:
    def test_identical_forecasts_combine_to_their_value(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        result = combine(forecasts, {"a": 0.5, "b": 0.5}, min_periods=2)
        assert math.isnan(result.iloc[0])  # diversification multiplier warms up
        assert np.allclose(result.to_numpy()[1:], [2.0, 3.0, 4.0])

    def test_applies_unequal_weights(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
        result = combine(forecasts, {"a": 0.25, "b": 0.75}, min_periods=2)
        assert result.iloc[-1] == pytest.approx(7.0)  # 0.25*4 + 0.75*8, multiplier 1

    def test_caps_the_combined_forecast(self) -> None:
        forecasts = _frame({"a": [10.0, 20.0, 30.0, 40.0], "b": [10.0, 20.0, 30.0, 40.0]})
        result = combine(forecasts, {"a": 0.5, "b": 0.5}, min_periods=2, cap=20.0)
        assert result.iloc[-1] == pytest.approx(20.0)

    def test_uses_an_explicit_multiplier(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        fdm = pd.Series(2.0, index=forecasts.index, dtype="float64")
        result = combine(forecasts, {"a": 0.5, "b": 0.5}, fdm=fdm)
        assert np.allclose(result.to_numpy(), [2.0, 4.0, 6.0, 8.0])

    def test_never_exceeds_the_cap(self) -> None:
        forecasts = _frame(
            {"a": [5.0, 30.0, 2.0, 40.0, 4.0, 60.0], "b": [3.0, 25.0, 8.0, 35.0, 6.0, 55.0]}
        )
        result = combine(forecasts, {"a": 0.5, "b": 0.5}, min_periods=2, cap=20.0).dropna()
        assert result.abs().max() <= 20.0 + 1e-9

    def test_preserves_the_index(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        result = combine(forecasts, {"a": 0.5, "b": 0.5}, min_periods=2)
        pd.testing.assert_index_equal(result.index, forecasts.index)

    def test_rejects_a_misaligned_multiplier(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        fdm = pd.Series([1.0, 1.0, 1.0], index=business_days(3), dtype="float64")
        with pytest.raises(DataValidationError, match="identically"):
            combine(forecasts, {"a": 0.5, "b": 0.5}, fdm=fdm)

    def test_rejects_a_non_positive_cap(self) -> None:
        forecasts = _frame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ConfigurationError, match="cap"):
            combine(forecasts, {"a": 0.5, "b": 0.5}, cap=0.0)

    def test_rejects_a_non_frame(self) -> None:
        with pytest.raises(DataValidationError, match="DataFrame"):
            combine([1.0, 2.0, 3.0], {"a": 1.0})  # type: ignore[arg-type]
