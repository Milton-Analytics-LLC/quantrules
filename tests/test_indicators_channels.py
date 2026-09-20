# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.channels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantrules.exceptions import ConfigurationError
from quantrules.indicators.channels import (
    bollinger_bands,
    donchian_channels,
    keltner_channels,
    keltner_position,
    rolling_max,
    rolling_min,
)
from tests.support import ohlcv, series


def _bars() -> tuple[pd.Series, pd.Series, pd.Series]:
    return (
        series([10.0, 12.0, 11.0]),
        series([8.0, 9.0, 7.0]),
        series([9.0, 11.0, 8.0]),
    )


class TestRollingMaxMin:
    def test_max_matches_hand_computed_windows(self) -> None:
        result = rolling_max(series([1.0, 3.0, 2.0, 5.0, 4.0]), window=2)

        assert math.isnan(result.iloc[0])
        assert list(result.iloc[1:]) == [3.0, 3.0, 5.0, 5.0]

    def test_min_matches_hand_computed_windows(self) -> None:
        result = rolling_min(series([1.0, 3.0, 2.0, 5.0, 4.0]), window=2)

        assert math.isnan(result.iloc[0])
        assert list(result.iloc[1:]) == [1.0, 2.0, 2.0, 4.0]


class TestDonchianChannels:
    def test_matches_hand_computed_channels(self) -> None:
        result = donchian_channels(series([1.0, 3.0, 2.0, 5.0, 4.0]), window=2)

        assert list(result.columns) == ["upper", "mid", "lower"]
        assert list(result["upper"].iloc[1:]) == [3.0, 3.0, 5.0, 5.0]
        assert list(result["lower"].iloc[1:]) == [1.0, 2.0, 2.0, 4.0]
        assert list(result["mid"].iloc[1:]) == [2.0, 2.5, 3.5, 4.5]

    def test_channels_are_ordered(self) -> None:
        result = donchian_channels(ohlcv(60, seed=1)["close"], window=10).dropna()

        assert (result["upper"] >= result["mid"]).all()
        assert (result["mid"] >= result["lower"]).all()


class TestBollingerBands:
    def test_matches_hand_computed_bands(self) -> None:
        """[1,2,3,4] window 3, ddof 0: mid 2, std sqrt(2/3), z (3-2)/std = sqrt(3/2)."""
        result = bollinger_bands(series([1.0, 2.0, 3.0, 4.0]), window=3, num_std=2.0)

        assert list(result.columns) == ["mid", "upper", "lower", "zscore"]
        std = math.sqrt(2.0 / 3.0)
        assert math.isclose(result["mid"].iloc[2], 2.0)
        assert math.isclose(result["upper"].iloc[2], 2.0 + 2.0 * std)
        assert math.isclose(result["lower"].iloc[2], 2.0 - 2.0 * std)
        assert math.isclose(result["zscore"].iloc[2], math.sqrt(1.5))

    def test_bands_are_ordered(self) -> None:
        result = bollinger_bands(ohlcv(60, seed=2)["close"], window=20).dropna()

        assert (result["upper"] >= result["mid"]).all()
        assert (result["mid"] >= result["lower"]).all()

    def test_rejects_ddof_not_below_window(self) -> None:
        with pytest.raises(ConfigurationError, match="ddof"):
            bollinger_bands(series([1.0, 2.0, 3.0]), window=2, ddof=2)


class TestKeltnerChannels:
    def test_mid_and_bands_match_hand_computed_values(self) -> None:
        """Window 2 simple ATR: mid EMA of close is 31/3 at bar 1, ATR is 2.5."""
        high, low, close = _bars()

        result = keltner_channels(high, low, close, window=2, atr_mult=2.0, atr_method="simple")

        assert list(result.columns) == ["mid", "upper", "lower"]
        assert math.isclose(result["mid"].iloc[1], 31.0 / 3.0)
        assert math.isclose(result["upper"].iloc[1], 31.0 / 3.0 + 2.0 * 2.5)
        assert math.isclose(result["lower"].iloc[1], 31.0 / 3.0 - 2.0 * 2.5)

    def test_channels_are_ordered(self) -> None:
        frame = ohlcv(60, seed=3)

        result = keltner_channels(frame["high"], frame["low"], frame["close"], window=10).dropna()

        assert (result["upper"] >= result["mid"]).all()
        assert (result["mid"] >= result["lower"]).all()


class TestKeltnerPosition:
    def test_matches_hand_computed_position(self) -> None:
        """(close - mid) / (mult * ATR) = (11 - 31/3) / (2 * 2.5) = 2/15."""
        high, low, close = _bars()

        result = keltner_position(high, low, close, window=2, atr_mult=2.0, atr_method="simple")

        assert math.isclose(result.iloc[1], 2.0 / 15.0)

    def test_is_clamped_to_the_unit_interval(self) -> None:
        frame = ohlcv(60, seed=4)

        result = keltner_position(
            frame["high"], frame["low"], frame["close"], window=10, atr_mult=0.01
        ).dropna()

        assert np.all(result.to_numpy() >= -1.0)
        assert np.all(result.to_numpy() <= 1.0)
        assert np.any(np.abs(result.to_numpy()) == 1.0)
