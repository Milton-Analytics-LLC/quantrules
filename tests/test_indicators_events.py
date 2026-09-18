# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for quantrules.indicators.events."""

from __future__ import annotations

import math

import numpy as np

from quantrules.indicators.events import bars_since, cross_down, cross_up
from tests.support import crossing_pair, series


class TestCrossUp:
    def test_flags_the_bar_where_fast_rises_through_slow(self) -> None:
        result = cross_up(series([1.0, 2.0, 1.0, 3.0]), series([2.0, 1.0, 2.0, 2.0]))

        assert list(result) == [0.0, 1.0, 0.0, 1.0]

    def test_is_binary(self) -> None:
        pair = crossing_pair(120, seed=1)

        result = cross_up(pair["fast"], pair["slow"])

        assert set(np.unique(result.to_numpy())) <= {0.0, 1.0}


class TestCrossDown:
    def test_flags_the_bar_where_fast_falls_through_slow(self) -> None:
        result = cross_down(series([1.0, 2.0, 1.0, 3.0]), series([2.0, 1.0, 2.0, 2.0]))

        assert list(result) == [0.0, 0.0, 1.0, 0.0]


class TestBarsSince:
    def test_counts_bars_since_the_last_event(self) -> None:
        """Events at positions 1 and 4: the count resets to 0 there and climbs between."""
        result = bars_since(series([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]))

        assert math.isnan(result.iloc[0])
        assert list(result.iloc[1:]) == [0.0, 1.0, 2.0, 0.0, 1.0]
