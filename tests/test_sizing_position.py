# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the subsystem position."""

from __future__ import annotations

import pandas as pd
import pytest

from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.sizing.position import subsystem_position
from tests.support import series


class TestSubsystemPosition:
    def test_matches_the_hand_computed_position(self) -> None:
        # N = (f / 10) * (1000 / sigma$) = f * 100 / sigma$
        # = [10*100/100, -10*100/200, 20*100/50] = [10, -5, 40].
        result = subsystem_position(
            series([10.0, -10.0, 20.0]), series([100.0, 200.0, 50.0]), cash_vol_target=1000.0
        )
        pd.testing.assert_series_equal(result, series([10.0, -5.0, 40.0]))

    def test_zero_forecast_is_flat(self) -> None:
        result = subsystem_position(
            series([0.0, 0.0]), series([100.0, 200.0]), cash_vol_target=1000.0
        )
        pd.testing.assert_series_equal(result, series([0.0, 0.0]))

    def test_scales_inversely_with_value_volatility(self) -> None:
        forecast = series([10.0, 20.0, -5.0])
        base = subsystem_position(forecast, series([100.0, 100.0, 100.0]), cash_vol_target=1000.0)
        doubled = subsystem_position(
            forecast, series([200.0, 200.0, 200.0]), cash_vol_target=1000.0
        )
        pd.testing.assert_series_equal(doubled, base / 2.0)

    def test_preserves_the_index(self) -> None:
        forecast = series([10.0, -10.0, 20.0])
        result = subsystem_position(forecast, series([100.0, 200.0, 50.0]), cash_vol_target=1000.0)
        pd.testing.assert_index_equal(result.index, forecast.index)

    def test_rejects_misaligned_inputs(self) -> None:
        forecast = series([10.0, -10.0])
        value_vol = pd.Series(
            [100.0, 200.0], index=pd.date_range("2021-06-01", periods=2, freq="B")
        )
        with pytest.raises(DataValidationError, match="identically"):
            subsystem_position(forecast, value_vol, cash_vol_target=1000.0)

    def test_rejects_a_non_series_forecast(self) -> None:
        with pytest.raises(DataValidationError, match="Series"):
            subsystem_position([10.0, 20.0], series([100.0, 200.0]), cash_vol_target=1000.0)  # type: ignore[arg-type]

    def test_rejects_a_non_positive_cash_vol_target(self) -> None:
        with pytest.raises(ConfigurationError, match="cash_vol_target"):
            subsystem_position(series([10.0]), series([100.0]), cash_vol_target=0.0)

    def test_rejects_a_non_positive_target_avg_abs_forecast(self) -> None:
        with pytest.raises(ConfigurationError, match="target_avg_abs_forecast"):
            subsystem_position(
                series([10.0]), series([100.0]), cash_vol_target=1000.0, target_avg_abs_forecast=0.0
            )
