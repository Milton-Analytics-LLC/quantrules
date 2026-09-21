# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the mean-reversion rule."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.exceptions import ConfigurationError
from quantrules.indicators.normalize import zscore
from quantrules.rules.mean_reversion import mean_reversion
from tests.support import random_walk, series


class TestMeanReversion:
    def test_matches_negative_hand_computed_zscore(self) -> None:
        """Negated rolling z-score of [1, 2, 3, 4] window 3 is -sqrt(3/2) at the last two points."""
        result = mean_reversion(series([1.0, 2.0, 3.0, 4.0]), window=3)
        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(-math.sqrt(1.5))
        assert result.iloc[3] == pytest.approx(-math.sqrt(1.5))

    def test_is_the_negation_of_the_zscore(self) -> None:
        price = random_walk(120, seed=21)
        result = mean_reversion(price, window=20)
        expected = -zscore(price, window=20)
        assert np.allclose(result.to_numpy(), expected.to_numpy(), equal_nan=True)

    @given(
        scale=st.floats(min_value=0.1, max_value=10.0),
        shift=st.floats(min_value=-50.0, max_value=50.0),
    )
    def test_is_invariant_to_a_positive_affine_transform(self, scale: float, shift: float) -> None:
        price = random_walk(80, seed=22)
        base = mean_reversion(price, window=20)
        transformed = mean_reversion(price * scale + shift, window=20)
        assert np.allclose(transformed.to_numpy(), base.to_numpy(), equal_nan=True)

    def test_preserves_the_index(self) -> None:
        price = random_walk(30, seed=23)
        result = mean_reversion(price, window=10)
        pd.testing.assert_index_equal(result.index, price.index)

    def test_rejects_a_non_positive_window(self) -> None:
        with pytest.raises(ConfigurationError, match="window"):
            mean_reversion(series([1.0, 2.0, 3.0]), window=0)
