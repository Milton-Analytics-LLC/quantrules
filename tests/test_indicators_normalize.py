# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.normalize."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.exceptions import ConfigurationError
from quantrules.indicators.normalize import clip, rolling_percentile, rolling_rank, zscore
from tests.support import random_walk, series


class TestZscore:
    def test_matches_hand_computed_values(self) -> None:
        """[1,2,3,4] window 3, ddof 0: z = (x - mid)/std = 1/sqrt(2/3) = sqrt(3/2)."""
        result = zscore(series([1.0, 2.0, 3.0, 4.0]), window=3)

        assert math.isnan(result.iloc[1])
        assert math.isclose(result.iloc[2], math.sqrt(1.5))
        assert math.isclose(result.iloc[3], math.sqrt(1.5))

    def test_min_periods_shortens_the_warmup(self) -> None:
        result = zscore(series([1.0, 2.0, 3.0, 4.0]), window=3, min_periods=2)

        assert math.isnan(result.iloc[0])
        assert not math.isnan(result.iloc[1])

    @given(
        scale=st.floats(min_value=0.1, max_value=100.0),
        shift=st.floats(min_value=-50, max_value=50),
    )
    def test_is_invariant_to_positive_affine_transforms(self, scale: float, shift: float) -> None:
        base = random_walk(60, seed=3)

        plain = zscore(base, window=10)
        transformed = zscore(base * scale + shift, window=10)

        assert np.allclose(plain.to_numpy(), transformed.to_numpy(), equal_nan=True)


class TestClip:
    def test_clamps_to_the_bounds(self) -> None:
        result = clip(series([-3.0, 0.0, 3.0]), lower=-1.0, upper=1.0)

        assert list(result) == [-1.0, 0.0, 1.0]

    def test_rejects_lower_above_upper(self) -> None:
        with pytest.raises(ConfigurationError, match="lower"):
            clip(series([1.0, 2.0]), lower=1.0, upper=-1.0)

    def test_is_idempotent(self) -> None:
        once = clip(random_walk(40, seed=4), lower=95.0, upper=105.0)
        twice = clip(once, lower=95.0, upper=105.0)

        assert np.allclose(once.to_numpy(), twice.to_numpy())


class TestRollingRank:
    def test_matches_hand_computed_position_in_range(self) -> None:
        """[1,3,2,5,4] window 3: (x - min)/(max - min) is 0.5, 1.0, 2/3."""
        result = rolling_rank(series([1.0, 3.0, 2.0, 5.0, 4.0]), window=3)

        assert math.isnan(result.iloc[1])
        assert math.isclose(result.iloc[2], 0.5)
        assert math.isclose(result.iloc[3], 1.0)
        assert math.isclose(result.iloc[4], 2.0 / 3.0)

    def test_stays_within_the_unit_interval(self) -> None:
        result = rolling_rank(random_walk(120, seed=5), window=20).dropna()

        assert (result >= 0.0).all()
        assert (result <= 1.0).all()


class TestRollingPercentile:
    def test_matches_hand_computed_empirical_cdf(self) -> None:
        """[1,3,2,5,4] window 3: fraction of the window at or below the last value."""
        result = rolling_percentile(series([1.0, 3.0, 2.0, 5.0, 4.0]), window=3)

        assert math.isnan(result.iloc[1])
        assert math.isclose(result.iloc[2], 2.0 / 3.0)
        assert math.isclose(result.iloc[3], 1.0)
        assert math.isclose(result.iloc[4], 2.0 / 3.0)

    def test_stays_within_the_unit_interval(self) -> None:
        result = rolling_percentile(random_walk(120, seed=6), window=20).dropna()

        assert (result > 0.0).all()
        assert (result <= 1.0).all()
