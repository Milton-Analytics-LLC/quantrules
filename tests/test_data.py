# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `MarketData` container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantrules.data import MarketData
from quantrules.exceptions import DataValidationError
from tests.support import business_days, series


class TestConstruction:
    def test_exposes_fields_from_a_frame(self) -> None:
        frame = pd.DataFrame(
            {"price": [1.0, 2.0, 3.0], "carry": [0.1, 0.2, 0.3]},
            index=business_days(3),
        )
        data = MarketData(frame)
        assert set(data.fields) == {"price", "carry"}
        pd.testing.assert_series_equal(data["price"], frame["price"])

    def test_index_is_carried(self) -> None:
        frame = pd.DataFrame({"price": [1.0, 2.0, 3.0]}, index=business_days(3))
        data = MarketData(frame)
        pd.testing.assert_index_equal(data.index, frame.index)

    def test_integer_columns_are_coerced_to_float(self) -> None:
        frame = pd.DataFrame({"price": [1, 2, 3]}, index=business_days(3))
        data = MarketData(frame)
        assert data["price"].dtype == np.float64

    def test_does_not_mutate_the_callers_frame(self) -> None:
        frame = pd.DataFrame({"price": [1.0, 2.0, 3.0]}, index=business_days(3))
        before = frame.copy()
        data = MarketData(frame)
        column = data["price"]
        column.iloc[:] = -999.0  # mutating what we got back must not touch the source
        pd.testing.assert_frame_equal(frame, before)

    def test_from_fields_builds_from_aligned_series(self) -> None:
        data = MarketData.from_fields(price=series([1.0, 2.0]), carry=series([0.1, 0.2]))
        assert set(data.fields) == {"price", "carry"}

    def test_contains_reports_membership(self) -> None:
        data = MarketData(pd.DataFrame({"price": [1.0, 2.0]}, index=business_days(2)))
        assert "price" in data
        assert "carry" not in data


class TestValidation:
    def test_rejects_a_non_frame(self) -> None:
        with pytest.raises(DataValidationError, match="DataFrame"):
            MarketData(series([1.0, 2.0]))

    def test_rejects_a_non_datetime_index(self) -> None:
        frame = pd.DataFrame({"price": [1.0, 2.0, 3.0]})  # default RangeIndex
        with pytest.raises(DataValidationError, match="DatetimeIndex"):
            MarketData(frame)

    def test_rejects_a_non_monotonic_index(self) -> None:
        idx = business_days(3)[::-1]
        frame = pd.DataFrame({"price": [1.0, 2.0, 3.0]}, index=idx)
        with pytest.raises(DataValidationError, match="monotonic"):
            MarketData(frame)

    def test_rejects_duplicate_index_entries(self) -> None:
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
        frame = pd.DataFrame({"price": [1.0, 2.0]}, index=idx)
        with pytest.raises(DataValidationError, match="duplicate"):
            MarketData(frame)

    def test_rejects_a_non_numeric_column(self) -> None:
        frame = pd.DataFrame({"price": ["a", "b"]}, index=business_days(2))
        with pytest.raises(DataValidationError, match="numeric"):
            MarketData(frame)

    def test_rejects_an_empty_frame(self) -> None:
        frame = pd.DataFrame(index=business_days(3))
        with pytest.raises(DataValidationError, match="at least one field"):
            MarketData(frame)

    def test_from_fields_rejects_misaligned_series(self) -> None:
        with pytest.raises(DataValidationError, match="identically"):
            MarketData.from_fields(price=series([1.0, 2.0]), carry=series([0.1, 0.2, 0.3]))

    def test_from_fields_rejects_no_fields(self) -> None:
        with pytest.raises(DataValidationError, match="at least one field"):
            MarketData.from_fields()


class TestFieldAccess:
    def test_missing_field_access_raises(self) -> None:
        data = MarketData(pd.DataFrame({"price": [1.0, 2.0]}, index=business_days(2)))
        with pytest.raises(DataValidationError, match="carry"):
            _ = data["carry"]

    def test_require_passes_when_all_present(self) -> None:
        data = MarketData(pd.DataFrame({"price": [1.0], "carry": [0.1]}, index=business_days(1)))
        data.require(("price", "carry"))  # must not raise

    def test_require_names_the_missing_field(self) -> None:
        data = MarketData(pd.DataFrame({"price": [1.0]}, index=business_days(1)))
        with pytest.raises(DataValidationError, match="volatility"):
            data.require(("price", "volatility"))
