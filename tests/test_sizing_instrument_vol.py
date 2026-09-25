# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the floored instrument volatility estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.indicators.volatility import ewma_volatility
from quantrules.sizing.instrument_vol import instrument_value_volatility, instrument_volatility
from tests.support import random_walk, series


class TestInstrumentVolatility:
    def test_floor_disabled_matches_ewma_volatility(self) -> None:
        returns = random_walk(120, seed=42)
        result = instrument_volatility(returns, floor_percentile=0.0)
        pd.testing.assert_series_equal(result, ewma_volatility(returns))

    def test_floor_lifts_a_quiet_tail_to_the_percentile(self) -> None:
        # span=1 makes the EWMA variance equal each squared return, so with
        # periods_per_year=1 the estimate is |returns|: [0.02, 0.02, 0.005, 0.005].
        # A 4-wide median floor is [0.02, 0.02, 0.02, 0.0125], so the two quiet
        # tail values are lifted from 0.005 to 0.02 and 0.0125.
        returns = series([0.02, 0.02, 0.005, 0.005])
        result = instrument_volatility(
            returns, span=1, min_periods=1, floor_window=4, floor_percentile=0.5, periods_per_year=1
        )
        pd.testing.assert_series_equal(result, series([0.02, 0.02, 0.02, 0.0125]))

    def test_floor_uses_the_lower_not_the_upper_percentile(self) -> None:
        # Decreasing sigma [0.04, 0.03, 0.02, 0.01] (span=1, ppy=1 => sigma=|r|).
        # The 25th percentile of each window sits below its max, pinning the
        # floor's direction: a 0.25 -> 0.75 flip would give the very different
        # [0.04, 0.0375, 0.035, 0.0325].
        returns = series([0.04, 0.03, 0.02, 0.01])
        result = instrument_volatility(
            returns,
            span=1,
            min_periods=1,
            floor_window=4,
            floor_percentile=0.25,
            periods_per_year=1,
        )
        pd.testing.assert_series_equal(result, series([0.04, 0.0325, 0.025, 0.0175]))

    def test_annualises_by_root_periods_per_year(self) -> None:
        returns = random_walk(120, seed=7)
        per_year = instrument_volatility(returns, floor_percentile=0.0, periods_per_year=256)
        per_period = instrument_volatility(returns, floor_percentile=0.0, periods_per_year=1)
        assert np.allclose(per_year.dropna(), per_period.dropna() * 16.0)

    def test_is_non_negative(self) -> None:
        result = instrument_volatility(random_walk(120, seed=9), floor_window=20)
        assert (result.dropna() >= 0.0).all()

    def test_preserves_the_index(self) -> None:
        returns = random_walk(120, seed=11)
        result = instrument_volatility(returns, floor_window=20)
        pd.testing.assert_index_equal(result.index, returns.index)

    def test_warms_up_to_nan_until_min_periods(self) -> None:
        result = instrument_volatility(random_walk(120, seed=13), min_periods=10, floor_window=20)
        assert result.iloc[:9].isna().all()
        assert result.iloc[9:].notna().all()

    def test_rejects_a_non_series(self) -> None:
        with pytest.raises(DataValidationError, match="Series"):
            instrument_volatility([0.01, 0.02, 0.03])  # type: ignore[arg-type]

    def test_rejects_a_floor_percentile_outside_the_unit_interval(self) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            instrument_volatility(series([0.01, 0.02]), floor_percentile=1.5)

    def test_rejects_a_non_positive_floor_window(self) -> None:
        with pytest.raises(ConfigurationError, match="floor_window"):
            instrument_volatility(series([0.01, 0.02]), floor_window=0)

    def test_rejects_a_non_positive_span(self) -> None:
        with pytest.raises(ConfigurationError, match="span"):
            instrument_volatility(series([0.01, 0.02]), span=0)


class TestInstrumentValueVolatility:
    def test_matches_the_hand_computed_product(self) -> None:
        result = instrument_value_volatility(
            series([100.0, 200.0]), series([0.01, 0.02]), block_size=10.0, exchange_rate=2.0
        )
        pd.testing.assert_series_equal(result, series([20.0, 80.0]))

    def test_defaults_leave_block_and_fx_at_one(self) -> None:
        result = instrument_value_volatility(series([100.0, 200.0]), series([0.01, 0.02]))
        pd.testing.assert_series_equal(result, series([1.0, 4.0]))

    def test_rejects_misaligned_inputs(self) -> None:
        price = series([100.0, 200.0])
        volatility = pd.Series([0.01, 0.02], index=pd.date_range("2021-06-01", periods=2, freq="B"))
        with pytest.raises(DataValidationError, match="identically"):
            instrument_value_volatility(price, volatility)

    def test_preserves_the_index(self) -> None:
        price = series([100.0, 200.0, 300.0])
        result = instrument_value_volatility(price, series([0.01, 0.02, 0.03]))
        pd.testing.assert_index_equal(result.index, price.index)

    def test_rejects_a_non_positive_block_size(self) -> None:
        with pytest.raises(ConfigurationError, match="block_size"):
            instrument_value_volatility(series([100.0]), series([0.01]), block_size=0.0)

    def test_rejects_a_non_positive_exchange_rate(self) -> None:
        with pytest.raises(ConfigurationError, match="exchange_rate"):
            instrument_value_volatility(series([100.0]), series([0.01]), exchange_rate=-1.0)
