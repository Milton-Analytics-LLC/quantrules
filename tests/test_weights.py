# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared weight-mapping validators."""

from __future__ import annotations

import numpy as np
import pytest

from quantrules._weights import (
    active_weights,
    aligned_weights,
    positive_weight_array,
    weight_array,
)
from quantrules.exceptions import ConfigurationError


class TestWeightArray:
    def test_returns_keys_and_values(self) -> None:
        keys, values = weight_array({"a": 0.6, "b": 0.4})
        assert keys == ["a", "b"]
        assert np.allclose(values, [0.6, 0.4])

    def test_rejects_a_negative_weight(self) -> None:
        with pytest.raises(ConfigurationError, match="0 or greater"):
            weight_array({"a": -0.1, "b": 1.1})

    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        with pytest.raises(ConfigurationError, match="sum to 1"):
            weight_array({"a": 0.3, "b": 0.3})


class TestPositiveWeightArray:
    def test_allows_weights_that_do_not_sum_to_one(self) -> None:
        keys, values = positive_weight_array({"a": 2.0, "b": 1.0})
        assert keys == ["a", "b"]
        assert np.allclose(values, [2.0, 1.0])

    def test_rejects_a_negative_weight(self) -> None:
        with pytest.raises(ConfigurationError, match="0 or greater"):
            positive_weight_array({"a": -1.0, "b": 2.0})

    def test_rejects_a_non_positive_sum(self) -> None:
        with pytest.raises(ConfigurationError, match="positive sum"):
            positive_weight_array({"a": 0.0, "b": 0.0})


class TestAlignedAndActiveWeights:
    def test_aligned_weights_orders_by_columns(self) -> None:
        values = aligned_weights({"b": 0.25, "a": 0.75}, ["a", "b"])
        assert np.allclose(values, [0.75, 0.25])

    def test_aligned_weights_rejects_a_key_mismatch(self) -> None:
        with pytest.raises(ConfigurationError, match="match"):
            aligned_weights({"a": 1.0}, ["a", "b"])

    def test_active_weights_drops_zero_weight_columns(self) -> None:
        columns, values = active_weights({"a": 0.5, "b": 0.0, "c": 0.5}, ["a", "b", "c"])
        assert columns == ["a", "c"]
        assert np.allclose(values, [0.5, 0.5])
