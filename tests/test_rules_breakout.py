# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the breakout rule."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantrules.exceptions import ConfigurationError
from quantrules.rules.breakout import breakout
from tests.support import random_walk, series


class TestBreakout:
    def test_matches_hand_computed_channel_position(self) -> None:
        """Span 2 on [1, 3, 2, 5, 4] gives (price - mid) / range [NaN, 0.5, -0.5, 0.5, -0.5]."""
        result = breakout(series([1.0, 3.0, 2.0, 5.0, 4.0]), span=2)
        assert math.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.5)
        assert result.iloc[2] == pytest.approx(-0.5)
        assert result.iloc[3] == pytest.approx(0.5)
        assert result.iloc[4] == pytest.approx(-0.5)

    def test_is_bounded_in_plus_or_minus_half(self) -> None:
        result = breakout(random_walk(120, seed=11), span=20).dropna()
        assert result.abs().max() <= 0.5 + 1e-9

    def test_smoothing_span_of_one_is_the_identity(self) -> None:
        price = random_walk(60, seed=12)
        unsmoothed = breakout(price, span=10)
        smoothed = breakout(price, span=10, smoothing_span=1)
        assert np.allclose(smoothed.to_numpy(), unsmoothed.to_numpy(), equal_nan=True)

    def test_preserves_the_index(self) -> None:
        price = random_walk(30, seed=13)
        result = breakout(price, span=10)
        pd.testing.assert_index_equal(result.index, price.index)

    def test_rejects_a_span_below_two(self) -> None:
        with pytest.raises(ConfigurationError, match="span"):
            breakout(series([1.0, 2.0, 3.0]), span=1)

    def test_rejects_a_non_positive_smoothing_span(self) -> None:
        with pytest.raises(ConfigurationError, match="smoothing_span"):
            breakout(series([1.0, 2.0, 3.0]), span=2, smoothing_span=0)
