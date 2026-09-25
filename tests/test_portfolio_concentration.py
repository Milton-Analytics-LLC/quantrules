# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for portfolio concentration measures."""

from __future__ import annotations

import pytest

from quantrules.exceptions import ConfigurationError
from quantrules.portfolio.concentration import (
    effective_number_of_instruments,
    herfindahl_hirschman_index,
)


class TestHerfindahlHirschmanIndex:
    def test_equal_weights_give_one_over_n(self) -> None:
        weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        assert herfindahl_hirschman_index(weights) == pytest.approx(0.25)

    def test_matches_a_hand_computed_value(self) -> None:
        # 0.5^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
        assert herfindahl_hirschman_index({"a": 0.5, "b": 0.3, "c": 0.2}) == pytest.approx(0.38)

    def test_a_single_instrument_is_one(self) -> None:
        assert herfindahl_hirschman_index({"a": 1.0}) == pytest.approx(1.0)

    def test_is_permutation_invariant(self) -> None:
        forward = herfindahl_hirschman_index({"a": 0.5, "b": 0.3, "c": 0.2})
        shuffled = herfindahl_hirschman_index({"c": 0.2, "a": 0.5, "b": 0.3})
        assert forward == pytest.approx(shuffled)

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ConfigurationError, match="0 or greater"):
            herfindahl_hirschman_index({"a": -0.1, "b": 1.1})

    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(ConfigurationError, match="sum to 1"):
            herfindahl_hirschman_index({"a": 0.3, "b": 0.3})


class TestEffectiveNumberOfInstruments:
    def test_equal_weights_give_n(self) -> None:
        weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        assert effective_number_of_instruments(weights) == pytest.approx(4.0)

    def test_is_the_reciprocal_of_the_index(self) -> None:
        assert effective_number_of_instruments({"a": 0.5, "b": 0.3, "c": 0.2}) == pytest.approx(
            1.0 / 0.38
        )

    def test_a_single_instrument_is_one(self) -> None:
        assert effective_number_of_instruments({"a": 1.0}) == pytest.approx(1.0)
