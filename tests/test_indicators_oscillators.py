# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.oscillators."""

from __future__ import annotations

import math

import pytest

from quantrules.exceptions import ConfigurationError
from quantrules.indicators.oscillators import rsi
from tests.support import random_walk, series


class TestRsi:
    def test_simple_matches_hand_computed_values(self) -> None:
        """Prices 1,2,3,4,3,2 window 2: all-up bars read 100, a balanced bar 50, all-down 0."""
        result = rsi(series([1.0, 2.0, 3.0, 4.0, 3.0, 2.0]), window=2, method="simple")

        assert math.isnan(result.iloc[1])
        assert result.iloc[2] == 100.0
        assert result.iloc[3] == 100.0
        assert result.iloc[4] == 50.0
        assert result.iloc[5] == 0.0

    def test_wilder_reads_one_hundred_when_only_gains(self) -> None:
        result = rsi(series([1.0, 2.0, 3.0, 4.0, 5.0]), window=2, method="wilder")

        assert result.iloc[-1] == 100.0

    def test_rejects_an_unknown_method(self) -> None:
        with pytest.raises(ConfigurationError, match="method"):
            rsi(random_walk(30, seed=1), method="bogus")

    def test_stays_within_zero_and_one_hundred(self) -> None:
        result = rsi(random_walk(200, seed=2), window=14).dropna()

        assert (result >= 0.0).all()
        assert (result <= 100.0).all()
