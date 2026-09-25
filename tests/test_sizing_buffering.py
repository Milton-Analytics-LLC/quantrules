# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for position buffering."""

from __future__ import annotations

import pandas as pd
import pytest

from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.sizing.buffering import buffer_position
from tests.support import random_walk, series


class TestBufferPosition:
    def test_matches_the_hand_traced_buffer(self) -> None:
        # average 100, fraction 0.1 => half-width 10.
        # t0 seed 100; t1 band [95,115] holds 100; t2 band [120,140] trades up to
        # 120; t3 band [80,100] trades down to 100; t4 band [90,110] holds 100.
        result = buffer_position(
            series([100.0, 105.0, 130.0, 90.0, 100.0]),
            series([100.0, 100.0, 100.0, 100.0, 100.0]),
            fraction=0.1,
        )
        pd.testing.assert_series_equal(result, series([100.0, 100.0, 120.0, 100.0, 100.0]))

    def test_zero_fraction_trades_to_optimal(self) -> None:
        optimal = series([100.0, 105.0, 130.0, 90.0])
        result = buffer_position(optimal, series([100.0, 100.0, 100.0, 100.0]), fraction=0.0)
        pd.testing.assert_series_equal(result, optimal)

    def test_disabled_buffer_follows_optimal_despite_a_missing_average(self) -> None:
        # fraction=0 disables buffering, so a NaN in average_position must not cause
        # a hold: the held position follows the optimal position exactly.
        optimal = series([100.0, 105.0, 130.0])
        average = series([100.0, float("nan"), 100.0])
        result = buffer_position(optimal, average, fraction=0.0)
        pd.testing.assert_series_equal(result, optimal)

    def test_disabled_buffer_passes_through_a_missing_optimal(self) -> None:
        optimal = series([100.0, float("nan"), 130.0])
        result = buffer_position(optimal, series([100.0, 100.0, 100.0]), fraction=0.0)
        pd.testing.assert_series_equal(result, optimal)

    def test_holds_when_optimal_stays_inside_the_band(self) -> None:
        result = buffer_position(
            series([100.0, 105.0, 103.0, 98.0]),
            series([100.0, 100.0, 100.0, 100.0]),
            fraction=0.1,
        )
        pd.testing.assert_series_equal(result, series([100.0, 100.0, 100.0, 100.0]))

    def test_nan_optimal_propagates_and_holds(self) -> None:
        result = buffer_position(
            series([float("nan"), 5.0, float("nan"), 5.0]),
            series([10.0, 10.0, 10.0, 10.0]),
            fraction=0.1,
        )
        pd.testing.assert_series_equal(result, series([float("nan"), 5.0, 5.0, 5.0]))

    def test_stays_within_the_band(self) -> None:
        optimal = random_walk(100, seed=5) - 100.0
        average = pd.Series(50.0, index=optimal.index)
        result = buffer_position(optimal, average, fraction=0.1)
        half = average.abs() * 0.1
        assert ((result - optimal).abs() <= half + 1e-9).all()

    def test_reduces_turnover(self) -> None:
        optimal = random_walk(100, seed=6) - 100.0
        average = pd.Series(50.0, index=optimal.index)
        result = buffer_position(optimal, average, fraction=0.1)
        assert result.diff().abs().sum() <= optimal.diff().abs().sum()

    def test_preserves_the_index(self) -> None:
        optimal = series([100.0, 105.0, 130.0])
        result = buffer_position(optimal, series([100.0, 100.0, 100.0]), fraction=0.1)
        pd.testing.assert_index_equal(result.index, optimal.index)

    def test_rejects_misaligned_inputs(self) -> None:
        optimal = series([100.0, 105.0])
        average = pd.Series([100.0, 100.0], index=pd.date_range("2021-06-01", periods=2, freq="B"))
        with pytest.raises(DataValidationError, match="identically"):
            buffer_position(optimal, average, fraction=0.1)

    def test_rejects_a_fraction_outside_the_unit_interval(self) -> None:
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            buffer_position(series([100.0]), series([100.0]), fraction=1.5)
