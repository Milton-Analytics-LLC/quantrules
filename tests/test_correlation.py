# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared causal correlation primitive."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.correlation import rolling_correlation
from quantrules.exceptions import ConfigurationError, DataValidationError
from tests.support import random_walk, series


class TestRollingCorrelation:
    def test_matches_hand_computed_expanding_correlation(self) -> None:
        """[1, 2, 3] vs [1, 3, 2]: two points correlate +1, three points give 1/2/2 = 0.5."""
        result = rolling_correlation(
            series([1.0, 2.0, 3.0]), series([1.0, 3.0, 2.0]), min_periods=2
        )
        assert math.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(1.0)
        assert result.iloc[2] == pytest.approx(0.5)

    def test_perfectly_correlated_series_are_one(self) -> None:
        a = series([1.0, 2.0, 3.0, 4.0])
        result = rolling_correlation(a, a, min_periods=2)
        assert np.allclose(result.to_numpy()[1:], 1.0)

    def test_perfectly_anticorrelated_series_are_minus_one(self) -> None:
        result = rolling_correlation(
            series([1.0, 2.0, 3.0, 4.0]), series([4.0, 3.0, 2.0, 1.0]), min_periods=2
        )
        assert np.allclose(result.to_numpy()[1:], -1.0)

    def test_rolling_window_uses_only_the_window(self) -> None:
        a = series([1.0, 2.0, 3.0, 4.0])
        result = rolling_correlation(a, a, window=2)
        assert math.isnan(result.iloc[0])
        assert np.allclose(result.to_numpy()[1:], 1.0)

    @given(seed=st.integers(min_value=0, max_value=50))
    def test_is_bounded_in_plus_or_minus_one(self, seed: int) -> None:
        a = random_walk(60, seed=seed)
        b = random_walk(60, seed=seed + 100)
        result = rolling_correlation(a, b, min_periods=5).dropna()
        assert result.abs().max() <= 1.0 + 1e-9

    def test_is_symmetric(self) -> None:
        a = random_walk(60, seed=7)
        b = random_walk(60, seed=8)
        forward = rolling_correlation(a, b, min_periods=5)
        backward = rolling_correlation(b, a, min_periods=5)
        assert np.allclose(forward.to_numpy(), backward.to_numpy(), equal_nan=True)

    def test_preserves_the_index(self) -> None:
        a = random_walk(30, seed=9)
        b = random_walk(30, seed=10)
        result = rolling_correlation(a, b, min_periods=5)
        pd.testing.assert_index_equal(result.index, a.index)

    def test_rejects_misaligned_series(self) -> None:
        with pytest.raises(DataValidationError, match="identically"):
            rolling_correlation(series([1.0, 2.0, 3.0]), series([1.0, 2.0]))

    def test_rejects_a_non_positive_window(self) -> None:
        with pytest.raises(ConfigurationError, match="window"):
            rolling_correlation(series([1.0, 2.0, 3.0]), series([1.0, 2.0, 3.0]), window=0)
