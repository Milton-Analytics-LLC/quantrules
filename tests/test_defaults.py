# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the library's default constants.

These guard the invariants that hold *between* the constants. A constant that
can be changed freely is not tested here; a constant whose value other code
silently depends on is.
"""

from __future__ import annotations

import math

from quantrules import defaults


class TestForecastConstants:
    def test_target_average_absolute_forecast_is_ten(self) -> None:
        assert defaults.TARGET_AVG_ABS_FORECAST == 10.0

    def test_forecast_cap_is_twenty(self) -> None:
        assert defaults.FORECAST_CAP == 20.0

    def test_the_cap_is_above_the_target(self) -> None:
        """A cap at or below the target average would clip most of the signal."""
        assert defaults.FORECAST_CAP > defaults.TARGET_AVG_ABS_FORECAST


class TestAnnualisationConstants:
    def test_business_days_per_year_is_256(self) -> None:
        assert defaults.BUSINESS_DAYS_PER_YEAR == 256

    def test_the_root_matches_the_day_count_exactly(self) -> None:
        """256 is used precisely because its square root is the integer 16."""
        root = defaults.ROOT_BUSINESS_DAYS_PER_YEAR
        assert root == 16.0
        assert math.isclose(root**2, defaults.BUSINESS_DAYS_PER_YEAR)

    def test_weeks_and_months_per_year(self) -> None:
        assert defaults.WEEKS_PER_YEAR == 52
        assert defaults.MONTHS_PER_YEAR == 12


class TestVolatilityConstants:
    def test_annual_volatility_target_is_a_positive_fraction(self) -> None:
        assert 0.0 < defaults.ANNUAL_VOL_TARGET < 1.0

    def test_volatility_span_is_a_positive_int(self) -> None:
        assert isinstance(defaults.VOLATILITY_EWMA_SPAN, int)
        assert defaults.VOLATILITY_EWMA_SPAN > 0

    def test_minimum_periods_does_not_exceed_the_span(self) -> None:
        assert defaults.VOLATILITY_MIN_PERIODS <= defaults.VOLATILITY_EWMA_SPAN

    def test_the_volatility_floor_is_a_low_percentile(self) -> None:
        assert 0.0 < defaults.VOLATILITY_FLOOR_PERCENTILE < 0.5


class TestPortfolioConstants:
    def test_the_diversification_multiplier_cap_is_at_least_one(self) -> None:
        """Diversification can only ever increase risk-adjusted size."""
        assert defaults.MAX_DIVERSIFICATION_MULTIPLIER >= 1.0

    def test_the_buffer_fraction_is_a_fraction(self) -> None:
        assert 0.0 < defaults.BUFFER_FRACTION < 1.0

    def test_the_correlation_floor_is_not_negative(self) -> None:
        """Negative correlations are floored before diversification maths."""
        assert defaults.CORRELATION_FLOOR >= 0.0


class TestEwmacSpeedPairs:
    def test_pairs_are_an_immutable_tuple(self) -> None:
        assert isinstance(defaults.EWMAC_SPEED_PAIRS, tuple)

    def test_every_slow_span_is_four_times_its_fast_span(self) -> None:
        for fast, slow in defaults.EWMAC_SPEED_PAIRS:
            assert slow == 4 * fast, f"({fast}, {slow}) breaks the 1:4 ratio"

    def test_pairs_are_ordered_from_fastest_to_slowest(self) -> None:
        fast_spans = [fast for fast, _ in defaults.EWMAC_SPEED_PAIRS]
        assert fast_spans == sorted(fast_spans)

    def test_the_standard_set_is_present(self) -> None:
        assert defaults.EWMAC_SPEED_PAIRS == (
            (2, 8),
            (4, 16),
            (8, 32),
            (16, 64),
            (32, 128),
            (64, 256),
        )


class TestModuleSurface:
    def test_every_public_constant_is_exported(self) -> None:
        """A new constant that is not added to __all__ is a documentation bug."""
        public = {name for name in vars(defaults) if name.isupper()}
        assert public == set(defaults.__all__)

    def test_exports_are_sorted(self) -> None:
        assert defaults.__all__ == sorted(defaults.__all__)
