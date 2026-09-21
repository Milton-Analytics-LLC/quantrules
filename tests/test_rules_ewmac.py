# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the EWMAC trend rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.rules.ewmac import ewmac
from tests.support import series


def _flat_vol(price: pd.Series, level: float = 2.0) -> pd.Series:
    return pd.Series(level, index=price.index, dtype="float64")


class TestEwmac:
    def test_matches_hand_computed_crossover_over_volatility(self) -> None:
        """Crossover [0, 1/3, 4/9] over a flat volatility of 2 gives [0, 1/6, 2/9]."""
        price = series([1.0, 2.0, 3.0])
        result = ewmac(price, fast_span=1, slow_span=2, volatility=_flat_vol(price))
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == pytest.approx(1.0 / 6.0)
        assert result.iloc[2] == pytest.approx(2.0 / 9.0)

    def test_is_positive_on_a_rising_price(self) -> None:
        price = series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = ewmac(price, fast_span=2, slow_span=4, volatility=_flat_vol(price))
        assert result.iloc[-1] > 0.0

    @given(scale=st.floats(min_value=0.1, max_value=10.0))
    def test_forecast_scales_inversely_with_volatility(self, scale: float) -> None:
        price = series([1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0])
        vol = _flat_vol(price)
        base = ewmac(price, fast_span=2, slow_span=4, volatility=vol)
        scaled = ewmac(price, fast_span=2, slow_span=4, volatility=vol * scale)
        assert np.allclose(scaled.to_numpy(), (base / scale).to_numpy(), equal_nan=True)

    def test_preserves_the_index(self) -> None:
        price = series([1.0, 2.0, 3.0, 4.0])
        result = ewmac(price, fast_span=2, slow_span=4, volatility=_flat_vol(price))
        pd.testing.assert_index_equal(result.index, price.index)

    def test_rejects_misaligned_volatility(self) -> None:
        price = series([1.0, 2.0, 3.0])
        vol = series([2.0, 2.0, 2.0, 2.0])
        with pytest.raises(DataValidationError, match="identically"):
            ewmac(price, fast_span=1, slow_span=2, volatility=vol)

    def test_rejects_a_non_positive_span(self) -> None:
        price = series([1.0, 2.0, 3.0])
        with pytest.raises(ConfigurationError, match="fast_span"):
            ewmac(price, fast_span=0, slow_span=2, volatility=_flat_vol(price))
