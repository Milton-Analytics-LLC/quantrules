# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.averages."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from quantrules.indicators.averages import ewma, sma
from tests.support import random_walk, series


class TestSma:
    def test_matches_hand_computed_windows(self) -> None:
        """[1,2,3,4,5] window 3: warmup, then (1+2+3)/3, (2+3+4)/3, (3+4+5)/3."""
        result = sma(series([1.0, 2.0, 3.0, 4.0, 5.0]), window=3)

        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert result.iloc[2] == 2.0
        assert result.iloc[3] == 3.0
        assert result.iloc[4] == 4.0

    def test_min_periods_shortens_the_warmup(self) -> None:
        result = sma(series([1.0, 2.0, 3.0, 4.0]), window=3, min_periods=1)

        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 1.5

    def test_preserves_the_index(self) -> None:
        price = random_walk(50, seed=1)

        result = sma(price, window=10)

        pd.testing.assert_index_equal(result.index, price.index)

    @given(
        value=st.floats(min_value=-1e6, max_value=1e6),
        length=st.integers(min_value=5, max_value=50),
    )
    def test_average_of_a_constant_is_that_constant(self, value: float, length: int) -> None:
        result = sma(series([value] * length), window=3)

        assert np.allclose(result.to_numpy()[2:], value)


class TestEwma:
    def test_matches_hand_computed_recursion(self) -> None:
        """[1,2,3] span 2: alpha=2/3, so 1, (2/3)*2+(1/3)*1=5/3, (2/3)*3+(1/3)*(5/3)=23/9."""
        result = ewma(series([1.0, 2.0, 3.0]), span=2)

        assert result.iloc[0] == 1.0
        assert math.isclose(result.iloc[1], 5.0 / 3.0)
        assert math.isclose(result.iloc[2], 23.0 / 9.0)

    def test_min_periods_delays_the_first_value(self) -> None:
        result = ewma(series([1.0, 2.0, 3.0, 4.0]), span=2, min_periods=3)

        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert not math.isnan(result.iloc[2])

    def test_preserves_the_index(self) -> None:
        price = random_walk(50, seed=2)

        result = ewma(price, span=10)

        pd.testing.assert_index_equal(result.index, price.index)

    @given(
        value=st.floats(min_value=-1e6, max_value=1e6),
        length=st.integers(min_value=1, max_value=50),
    )
    def test_average_of_a_constant_is_that_constant(self, value: float, length: int) -> None:
        result = ewma(series([value] * length), span=5)

        assert np.allclose(result.to_numpy(), value)
