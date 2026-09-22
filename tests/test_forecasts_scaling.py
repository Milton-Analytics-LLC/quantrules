# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for forecast scaling."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.exceptions import ConfigurationError
from quantrules.forecasts.scaling import forecast_scalar, scale
from tests.support import random_walk, series

# Carver's published forecast scalars per EWMAC speed (from a backtested example),
# supplied by the maintainer. Used only as test fixtures, never reproduced in the docs.
PUBLISHED_EWMAC_SCALARS = {
    (2, 8): 10.6,
    (4, 16): 7.5,
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
    (64, 256): 1.87,
}


class TestForecastScalar:
    def test_matches_hand_computed_expanding_scalar(self) -> None:
        """|raw| is a flat 2, so the expanding mean is 2 and the scalar is 10 / 2 = 5."""
        result = forecast_scalar(series([2.0, -2.0, 2.0, -2.0]), target=10.0, min_periods=2)
        assert math.isnan(result.iloc[0])
        assert np.allclose(result.to_numpy()[1:], 5.0)

    def test_last_scalar_hits_the_target_on_average(self) -> None:
        raw = random_walk(120, seed=41) - 100.0  # centre so it takes both signs
        result = forecast_scalar(raw, target=10.0, min_periods=2)
        assert result.iloc[-1] == pytest.approx(10.0 / raw.abs().mean())

    def test_rolling_window_uses_only_the_window(self) -> None:
        """|raw| = [1, 3, 2, 4] with window 2: rolling means 2, 2.5, 3 give scalars 5, 4, 10/3."""
        result = forecast_scalar(
            series([1.0, -3.0, 2.0, -4.0]), target=10.0, window=2, min_periods=2
        )
        assert math.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(5.0)
        assert result.iloc[2] == pytest.approx(4.0)
        assert result.iloc[3] == pytest.approx(10.0 / 3.0)

    def test_rolling_window_below_the_expanding_warmup_does_not_raise(self) -> None:
        # A window smaller than the default expanding warmup must adapt, not raise.
        result = forecast_scalar(series([2.0] * 30), target=10.0, window=20)
        assert math.isnan(result.iloc[18])
        assert result.iloc[-1] == pytest.approx(5.0)

    def test_rejects_min_periods_above_the_window(self) -> None:
        with pytest.raises(ConfigurationError, match="window"):
            forecast_scalar(series([1.0] * 10), window=5, min_periods=10)


class TestScale:
    def test_explicit_scalar_multiplies(self) -> None:
        result = scale(series([1.0, -2.0, 3.0]), scalar=2.0)
        assert np.allclose(result.to_numpy(), [2.0, -4.0, 6.0])

    @pytest.mark.parametrize("scalar", sorted(PUBLISHED_EWMAC_SCALARS.values()))
    def test_published_scalars_scale_exactly(self, scalar: float) -> None:
        raw = series([1.0, -2.0, 3.0, -4.0])
        result = scale(raw, scalar=scalar)
        assert np.allclose(result.to_numpy(), raw.to_numpy() * scalar)

    def test_estimated_scaling_matches_hand_computed(self) -> None:
        result = scale(series([2.0, -2.0, 2.0, -2.0]), target=10.0, min_periods=2)
        assert math.isnan(result.iloc[0])
        assert np.allclose(result.to_numpy()[1:], [-10.0, 10.0, -10.0])

    @given(factor=st.floats(min_value=0.1, max_value=10.0))
    def test_estimated_scaling_is_homogeneous_in_target(self, factor: float) -> None:
        raw = series([2.0, -3.0, 1.0, -4.0, 2.5])
        base = scale(raw, target=10.0, min_periods=2)
        scaled = scale(raw, target=10.0 * factor, min_periods=2)
        assert np.allclose(scaled.to_numpy(), base.to_numpy() * factor, equal_nan=True)

    def test_preserves_the_index(self) -> None:
        raw = series([1.0, -2.0, 3.0, -4.0])
        result = scale(raw, target=10.0, min_periods=2)
        pd.testing.assert_index_equal(result.index, raw.index)

    def test_rejects_a_non_positive_scalar(self) -> None:
        with pytest.raises(ConfigurationError, match="scalar"):
            scale(series([1.0, 2.0, 3.0]), scalar=0.0)

    def test_rejects_a_non_positive_target(self) -> None:
        with pytest.raises(ConfigurationError, match="target"):
            scale(series([1.0, 2.0, 3.0]), target=0.0, min_periods=2)
