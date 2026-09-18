# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the input-validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantrules._validation import (
    ensure_aligned,
    ensure_fraction,
    ensure_non_negative,
    ensure_positive,
    ensure_positive_int,
    ensure_series,
)
from quantrules.exceptions import ConfigurationError, DataValidationError, QuantRulesError
from tests.support import business_days, series


class TestEnsurePositive:
    def test_returns_the_value_unchanged(self) -> None:
        assert ensure_positive(0.25, "annual_vol_target") == 0.25

    def test_rejects_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="annual_vol_target"):
            ensure_positive(0.0, "annual_vol_target")

    def test_rejects_negative(self) -> None:
        with pytest.raises(ConfigurationError, match="annual_vol_target"):
            ensure_positive(-1.0, "annual_vol_target")

    def test_rejects_nan(self) -> None:
        with pytest.raises(ConfigurationError, match="annual_vol_target"):
            ensure_positive(float("nan"), "annual_vol_target")

    def test_rejects_infinity(self) -> None:
        with pytest.raises(ConfigurationError, match="capital"):
            ensure_positive(float("inf"), "capital")

    def test_rejects_a_non_numeric_value(self) -> None:
        with pytest.raises(ConfigurationError, match="capital must be a real number"):
            ensure_positive("1000", "capital")  # type: ignore[arg-type]

    def test_rejects_a_bool(self) -> None:
        """``True`` is an ``int``, but never a meaningful volatility target."""
        with pytest.raises(ConfigurationError, match="annual_vol_target"):
            ensure_positive(True, "annual_vol_target")


class TestEnsureNonNegative:
    def test_accepts_zero(self) -> None:
        assert ensure_non_negative(0.0, "cost") == 0.0

    def test_rejects_negative(self) -> None:
        with pytest.raises(ConfigurationError, match="cost"):
            ensure_non_negative(-0.01, "cost")


class TestEnsureFraction:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_accepts_the_closed_unit_interval(self, value: float) -> None:
        assert ensure_fraction(value, "buffer_fraction") == value

    @pytest.mark.parametrize("value", [-0.0001, 1.0001])
    def test_rejects_values_outside_the_unit_interval(self, value: float) -> None:
        with pytest.raises(ConfigurationError, match="buffer_fraction"):
            ensure_fraction(value, "buffer_fraction")


class TestEnsurePositiveInt:
    def test_accepts_one(self) -> None:
        assert ensure_positive_int(1, "span") == 1

    def test_rejects_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="span"):
            ensure_positive_int(0, "span")

    def test_rejects_a_float(self) -> None:
        with pytest.raises(ConfigurationError, match="span"):
            ensure_positive_int(1.5, "span")  # type: ignore[arg-type]

    def test_rejects_a_bool(self) -> None:
        """``bool`` is a subclass of ``int``; accepting it would be a silent bug."""
        with pytest.raises(ConfigurationError, match="span"):
            ensure_positive_int(True, "span")


class TestEnsureSeries:
    def test_casts_an_integer_series_to_float64(self) -> None:
        raw = pd.Series([1, 2, 3], index=business_days(3), dtype="int64")

        result = ensure_series(raw, "price")

        assert result.dtype == np.dtype("float64")
        assert list(result) == [1.0, 2.0, 3.0]

    def test_preserves_the_index(self) -> None:
        raw = series([1.0, 2.0, 3.0])

        result = ensure_series(raw, "price")

        pd.testing.assert_index_equal(result.index, raw.index)

    def test_does_not_mutate_the_callers_series(self) -> None:
        raw = pd.Series([1, 2, 3], index=business_days(3), dtype="int64")

        ensure_series(raw, "price")

        assert raw.dtype == np.dtype("int64")

    def test_allows_missing_values(self) -> None:
        raw = series([1.0, float("nan"), 3.0])

        result = ensure_series(raw, "price")

        assert bool(result.isna().iloc[1])

    def test_allows_an_empty_series(self) -> None:
        raw = pd.Series([], index=business_days(0), dtype="float64")

        assert ensure_series(raw, "price").empty

    def test_rejects_a_list(self) -> None:
        with pytest.raises(DataValidationError, match="price"):
            ensure_series([1.0, 2.0], "price")

    def test_rejects_a_dataframe(self) -> None:
        frame = pd.DataFrame({"a": [1.0]}, index=business_days(1))
        with pytest.raises(DataValidationError, match="price"):
            ensure_series(frame, "price")

    def test_rejects_a_non_datetime_index(self) -> None:
        raw = pd.Series([1.0, 2.0], index=[0, 1], dtype="float64")
        with pytest.raises(DataValidationError, match="DatetimeIndex"):
            ensure_series(raw, "price")

    def test_rejects_an_unsorted_index(self) -> None:
        index = business_days(3)[::-1]
        raw = pd.Series([1.0, 2.0, 3.0], index=index, dtype="float64")
        with pytest.raises(DataValidationError, match="monotonic"):
            ensure_series(raw, "price")

    def test_rejects_a_duplicated_index(self) -> None:
        stamp = pd.Timestamp("2020-01-01")
        raw = pd.Series([1.0, 2.0], index=pd.DatetimeIndex([stamp, stamp]), dtype="float64")
        with pytest.raises(DataValidationError, match="duplicate"):
            ensure_series(raw, "price")

    def test_rejects_a_non_numeric_series(self) -> None:
        raw = pd.Series(["a", "b"], index=business_days(2))
        with pytest.raises(DataValidationError, match="numeric"):
            ensure_series(raw, "price")


class TestEnsureAligned:
    def test_accepts_identical_indexes(self) -> None:
        ensure_aligned(price=series([1.0, 2.0]), volatility=series([0.1, 0.2]))

    def test_rejects_differing_indexes(self) -> None:
        with pytest.raises(DataValidationError, match="volatility"):
            ensure_aligned(price=series([1.0, 2.0]), volatility=series([0.1, 0.2, 0.3]))

    def test_accepts_a_single_series(self) -> None:
        ensure_aligned(price=series([1.0, 2.0]))


class TestExceptionHierarchy:
    def test_configuration_error_is_a_value_error(self) -> None:
        """Callers that already catch ``ValueError`` keep working."""
        assert issubclass(ConfigurationError, ValueError)

    def test_data_validation_error_is_a_value_error(self) -> None:
        assert issubclass(DataValidationError, ValueError)

    @pytest.mark.parametrize("error", [ConfigurationError, DataValidationError])
    def test_every_error_shares_one_base(self, error: type[Exception]) -> None:
        assert issubclass(error, QuantRulesError)
