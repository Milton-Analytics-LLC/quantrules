# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for forecast capping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.exceptions import ConfigurationError
from quantrules.forecasts.capping import cap
from tests.support import random_walk, series


class TestCap:
    def test_clips_to_the_symmetric_cap(self) -> None:
        result = cap(series([-30.0, -10.0, 0.0, 10.0, 30.0]), cap=20.0)
        assert np.allclose(result.to_numpy(), [-20.0, -10.0, 0.0, 10.0, 20.0])

    def test_default_cap_is_plus_or_minus_twenty(self) -> None:
        result = cap(series([-30.0, 30.0]))
        assert np.allclose(result.to_numpy(), [-20.0, 20.0])

    @given(limit=st.floats(min_value=1.0, max_value=50.0))
    def test_result_never_exceeds_the_cap(self, limit: float) -> None:
        result = cap(random_walk(80, seed=51) - 100.0, cap=limit)
        assert result.abs().max() <= limit + 1e-9

    def test_is_idempotent(self) -> None:
        forecast = random_walk(80, seed=52) - 100.0
        once = cap(forecast, cap=20.0)
        twice = cap(once, cap=20.0)
        assert np.allclose(once.to_numpy(), twice.to_numpy(), equal_nan=True)

    def test_leaves_values_within_the_cap_unchanged(self) -> None:
        forecast = series([-5.0, 0.0, 12.0, 19.0])
        result = cap(forecast, cap=20.0)
        assert np.allclose(result.to_numpy(), forecast.to_numpy())

    def test_preserves_the_index(self) -> None:
        forecast = series([-30.0, 10.0, 30.0])
        result = cap(forecast, cap=20.0)
        pd.testing.assert_index_equal(result.index, forecast.index)

    def test_rejects_a_non_positive_cap(self) -> None:
        with pytest.raises(ConfigurationError, match="cap"):
            cap(series([1.0, 2.0, 3.0]), cap=0.0)
