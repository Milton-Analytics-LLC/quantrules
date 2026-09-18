# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.volume."""

from __future__ import annotations

import math

from quantrules.indicators.volume import accumulation_distribution, chaikin_money_flow, obv
from tests.support import ohlcv, series


class TestObv:
    def test_matches_hand_computed_running_total(self) -> None:
        """Up bars add volume, down bars subtract it, unchanged bars add nothing."""
        close = series([10.0, 11.0, 10.0, 10.0, 12.0])
        volume = series([100.0, 200.0, 150.0, 300.0, 250.0])

        result = obv(close, volume)

        assert list(result) == [0.0, 200.0, 50.0, 50.0, 300.0]


class TestAccumulationDistribution:
    def test_matches_hand_computed_line(self) -> None:
        """Close at the high adds +volume, at the low subtracts it, at the mid adds 0."""
        high = series([10.0, 10.0, 10.0])
        low = series([8.0, 8.0, 8.0])
        close = series([10.0, 8.0, 9.0])
        volume = series([100.0, 200.0, 300.0])

        result = accumulation_distribution(high, low, close, volume)

        assert list(result) == [100.0, -100.0, -100.0]


class TestChaikinMoneyFlow:
    def test_matches_hand_computed_ratio(self) -> None:
        """Sum of money-flow volume over the window divided by summed volume."""
        high = series([10.0, 10.0, 10.0])
        low = series([8.0, 8.0, 8.0])
        close = series([10.0, 8.0, 9.0])
        volume = series([100.0, 200.0, 300.0])

        result = chaikin_money_flow(high, low, close, volume, window=2)

        assert math.isnan(result.iloc[0])
        assert math.isclose(result.iloc[1], -1.0 / 3.0)
        assert math.isclose(result.iloc[2], -0.4)

    def test_stays_within_the_unit_interval(self) -> None:
        frame = ohlcv(80, seed=1)

        result = chaikin_money_flow(
            frame["high"], frame["low"], frame["close"], frame["volume"], window=20
        ).dropna()

        assert (result >= -1.0).all()
        assert (result <= 1.0).all()
