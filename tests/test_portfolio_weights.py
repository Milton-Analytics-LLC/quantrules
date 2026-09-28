# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for instrument weight builders."""

from __future__ import annotations

import pytest

from quantrules.exceptions import ConfigurationError
from quantrules.portfolio.weights import equal_weights, normalize_weights


class TestEqualWeights:
    def test_assigns_one_over_n(self) -> None:
        assert equal_weights(["a", "b", "c", "d"]) == pytest.approx(
            {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        )

    def test_a_single_instrument_gets_all_the_weight(self) -> None:
        assert equal_weights(["a"]) == {"a": 1.0}

    def test_rejects_an_empty_sequence(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one"):
            equal_weights([])

    def test_rejects_duplicate_instruments(self) -> None:
        with pytest.raises(ConfigurationError, match="duplicate"):
            equal_weights(["a", "b", "a"])


class TestNormalizeWeights:
    def test_scales_to_sum_to_one(self) -> None:
        assert normalize_weights({"a": 2.0, "b": 1.0, "c": 1.0}) == pytest.approx(
            {"a": 0.5, "b": 0.25, "c": 0.25}
        )

    def test_already_normalised_is_unchanged(self) -> None:
        assert normalize_weights({"a": 0.5, "b": 0.5}) == pytest.approx({"a": 0.5, "b": 0.5})

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ConfigurationError, match="0 or greater"):
            normalize_weights({"a": -1.0, "b": 2.0})

    def test_rejects_an_all_zero_mapping(self) -> None:
        with pytest.raises(ConfigurationError, match="positive sum"):
            normalize_weights({"a": 0.0, "b": 0.0})
