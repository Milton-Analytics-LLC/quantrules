# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.breadth."""

from __future__ import annotations

import math

import numpy as np

from quantrules.indicators.breadth import (
    advance_decline_line,
    mcclellan_oscillator,
    new_high_new_low_index,
    trin,
)
from tests.support import series


class TestAdvanceDeclineLine:
    def test_accumulates_net_advances(self) -> None:
        """Net advances 20, 20, -20 accumulate to 20, 40, 20."""
        result = advance_decline_line(series([100.0, 120.0, 90.0]), series([80.0, 100.0, 110.0]))

        assert list(result) == [20.0, 40.0, 20.0]


class TestMcClellanOscillator:
    def test_matches_hand_computed_difference_of_emas(self) -> None:
        """fast=1 (identity) minus a span-2 EMA of net advances [20,20,-20]."""
        result = mcclellan_oscillator(
            series([100.0, 120.0, 90.0]), series([80.0, 100.0, 110.0]), fast_span=1, slow_span=2
        )

        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0
        assert math.isclose(result.iloc[2], -40.0 / 3.0)

    def test_constant_breadth_gives_zero(self) -> None:
        result = mcclellan_oscillator(series([300.0] * 40), series([200.0] * 40))

        assert np.allclose(result.to_numpy(), 0.0)


class TestNewHighNewLowIndex:
    def test_is_the_net_of_highs_over_lows(self) -> None:
        result = new_high_new_low_index(series([5.0, 3.0, 8.0]), series([2.0, 4.0, 1.0]))

        assert list(result) == [3.0, -1.0, 7.0]


class TestTrin:
    def test_matches_hand_computed_ratio_of_ratios(self) -> None:
        """(advances/declines) / (advancing_volume/declining_volume)."""
        result = trin(
            series([100.0, 60.0]),
            series([50.0, 120.0]),
            series([1000.0, 800.0]),
            series([2000.0, 400.0]),
        )

        assert math.isclose(result.iloc[0], 4.0)
        assert math.isclose(result.iloc[1], 0.25)
