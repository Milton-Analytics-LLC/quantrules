# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the carry rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.defaults import CARRY_SMOOTHING_SPAN
from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.rules.carry import carry
from tests.support import random_walk, series


class TestCarry:
    def test_matches_hand_computed_smoothing(self) -> None:
        """[2, 4, 6] over a flat volatility of 2 is [1, 2, 3]; its span-2 EWMA is [1, 5/3, 23/9]."""
        raw = series([2.0, 4.0, 6.0])
        vol = series([2.0, 2.0, 2.0])
        result = carry(raw, volatility=vol, smoothing_span=2)
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[1] == pytest.approx(5.0 / 3.0)
        assert result.iloc[2] == pytest.approx(23.0 / 9.0)

    def test_smoothing_span_of_one_is_the_normalised_carry(self) -> None:
        raw = series([2.0, 4.0, 6.0])
        vol = series([2.0, 2.0, 2.0])
        result = carry(raw, volatility=vol, smoothing_span=1)
        assert np.allclose(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_default_smoothing_span_is_the_library_default(self) -> None:
        raw = random_walk(120, seed=31)
        vol = pd.Series(2.0, index=raw.index, dtype="float64")
        default = carry(raw, volatility=vol)
        explicit = carry(raw, volatility=vol, smoothing_span=CARRY_SMOOTHING_SPAN)
        assert np.allclose(default.to_numpy(), explicit.to_numpy(), equal_nan=True)

    @given(scale=st.floats(min_value=0.1, max_value=10.0))
    def test_forecast_scales_inversely_with_volatility(self, scale: float) -> None:
        raw = series([1.0, -2.0, 3.0, -4.0, 5.0])
        vol = pd.Series(2.0, index=raw.index, dtype="float64")
        base = carry(raw, volatility=vol, smoothing_span=4)
        scaled = carry(raw, volatility=vol * scale, smoothing_span=4)
        assert np.allclose(scaled.to_numpy(), (base / scale).to_numpy(), equal_nan=True)

    def test_preserves_the_index(self) -> None:
        raw = series([1.0, 2.0, 3.0, 4.0])
        vol = series([2.0, 2.0, 2.0, 2.0])
        result = carry(raw, volatility=vol, smoothing_span=2)
        pd.testing.assert_index_equal(result.index, raw.index)

    def test_rejects_misaligned_volatility(self) -> None:
        raw = series([1.0, 2.0, 3.0])
        vol = series([2.0, 2.0, 2.0, 2.0])
        with pytest.raises(DataValidationError, match="identically"):
            carry(raw, volatility=vol)

    def test_rejects_a_non_positive_smoothing_span(self) -> None:
        raw = series([1.0, 2.0, 3.0])
        vol = series([2.0, 2.0, 2.0])
        with pytest.raises(ConfigurationError, match="smoothing_span"):
            carry(raw, volatility=vol, smoothing_span=0)
