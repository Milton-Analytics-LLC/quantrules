# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed configuration objects."""

from __future__ import annotations

import dataclasses
import math

import pytest

from quantrules import defaults
from quantrules.config import (
    BufferingConfig,
    CostConfig,
    ForecastScalingConfig,
    SystemConfig,
    VolatilityEstimationConfig,
    VolatilityTargetConfig,
)
from quantrules.exceptions import ConfigurationError


class TestVolatilityTargetConfig:
    def test_stores_the_supplied_values(self) -> None:
        config = VolatilityTargetConfig(annual_vol_target=0.20, capital=500_000.0)

        assert config.annual_vol_target == 0.20
        assert config.capital == 500_000.0
        assert config.periods_per_year == defaults.BUSINESS_DAYS_PER_YEAR

    def test_annual_cash_volatility_target(self) -> None:
        """1,000,000 of capital at a 25% target risks 250,000 a year."""
        config = VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000_000.0)

        assert config.annual_cash_vol_target == 250_000.0

    def test_daily_cash_volatility_target(self) -> None:
        """250,000 a year over 256 business days is 250,000 / 16 = 15,625 a day."""
        config = VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000_000.0)

        assert config.periodic_cash_vol_target == 15_625.0

    def test_weekly_cash_volatility_target(self) -> None:
        config = VolatilityTargetConfig(
            annual_vol_target=0.25, capital=1_000_000.0, periods_per_year=52
        )

        assert math.isclose(config.periodic_cash_vol_target, 250_000.0 / math.sqrt(52))

    def test_coerces_an_integer_capital_to_float(self) -> None:
        config = VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000_000)

        assert isinstance(config.capital, float)

    def test_is_frozen(self) -> None:
        config = VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000.0)

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.capital = 2_000.0  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        """Frozen configs are cache keys for estimated scalars."""
        config = VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000.0)

        assert hash(config) == hash(VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000.0))

    @pytest.mark.parametrize("target", [0.0, -0.25])
    def test_rejects_a_non_positive_volatility_target(self, target: float) -> None:
        with pytest.raises(ConfigurationError, match="annual_vol_target"):
            VolatilityTargetConfig(annual_vol_target=target, capital=1_000.0)

    def test_rejects_non_positive_capital(self) -> None:
        with pytest.raises(ConfigurationError, match="capital"):
            VolatilityTargetConfig(annual_vol_target=0.25, capital=0.0)

    def test_rejects_non_positive_periods_per_year(self) -> None:
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000.0, periods_per_year=0)


class TestVolatilityEstimationConfig:
    def test_defaults_come_from_the_defaults_module(self) -> None:
        config = VolatilityEstimationConfig()

        assert config.span == defaults.VOLATILITY_EWMA_SPAN
        assert config.min_periods == defaults.VOLATILITY_MIN_PERIODS
        assert config.floor_percentile == defaults.VOLATILITY_FLOOR_PERCENTILE
        assert config.floor_window == defaults.VOLATILITY_FLOOR_WINDOW

    def test_rejects_a_non_positive_span(self) -> None:
        with pytest.raises(ConfigurationError, match="span"):
            VolatilityEstimationConfig(span=0)

    def test_rejects_a_floor_percentile_outside_the_unit_interval(self) -> None:
        with pytest.raises(ConfigurationError, match="floor_percentile"):
            VolatilityEstimationConfig(floor_percentile=1.5)

    def test_allows_disabling_the_floor_with_zero(self) -> None:
        assert VolatilityEstimationConfig(floor_percentile=0.0).floor_percentile == 0.0


class TestForecastScalingConfig:
    def test_defaults_to_a_target_of_ten_capped_at_twenty(self) -> None:
        config = ForecastScalingConfig()

        assert config.target_avg_abs_forecast == defaults.TARGET_AVG_ABS_FORECAST
        assert config.cap == defaults.FORECAST_CAP

    def test_window_defaults_to_none_meaning_expanding(self) -> None:
        assert ForecastScalingConfig().window is None

    def test_accepts_a_rolling_window(self) -> None:
        assert ForecastScalingConfig(window=500).window == 500

    def test_rejects_a_non_positive_window(self) -> None:
        with pytest.raises(ConfigurationError, match="window"):
            ForecastScalingConfig(window=0)

    def test_rejects_a_cap_below_the_target(self) -> None:
        """A cap under the average absolute forecast would clip most observations."""
        with pytest.raises(ConfigurationError, match="cap"):
            ForecastScalingConfig(target_avg_abs_forecast=10.0, cap=5.0)

    def test_allows_a_cap_equal_to_the_target(self) -> None:
        assert ForecastScalingConfig(target_avg_abs_forecast=10.0, cap=10.0).cap == 10.0


class TestBufferingConfig:
    def test_defaults_to_the_standard_buffer(self) -> None:
        assert BufferingConfig().fraction == defaults.BUFFER_FRACTION

    def test_allows_zero_meaning_no_buffering(self) -> None:
        assert BufferingConfig(fraction=0.0).fraction == 0.0

    def test_rejects_a_fraction_above_one(self) -> None:
        with pytest.raises(ConfigurationError, match="fraction"):
            BufferingConfig(fraction=1.5)


class TestCostConfig:
    def test_defaults_to_free_trading(self) -> None:
        config = CostConfig()

        assert config.cost_per_trade == 0.0
        assert config.periods_per_year == defaults.BUSINESS_DAYS_PER_YEAR

    def test_stores_a_cost(self) -> None:
        assert CostConfig(cost_per_trade=0.002).cost_per_trade == 0.002

    def test_rejects_a_negative_cost(self) -> None:
        with pytest.raises(ConfigurationError, match="cost_per_trade"):
            CostConfig(cost_per_trade=-0.1)

    def test_rejects_a_non_positive_periods_per_year(self) -> None:
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            CostConfig(periods_per_year=0)


class TestSystemConfig:
    def test_requires_a_volatility_target_and_defaults_the_rest(self) -> None:
        target = VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000_000.0)

        config = SystemConfig(volatility_target=target)

        assert config.volatility_target is target
        assert config.volatility_estimation == VolatilityEstimationConfig()
        assert config.forecast_scaling == ForecastScalingConfig()
        assert config.buffering == BufferingConfig()
        assert config.costs == CostConfig()

    def test_is_frozen(self) -> None:
        config = SystemConfig(
            volatility_target=VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000.0)
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.buffering = BufferingConfig()  # type: ignore[misc]


class TestMappingRoundTrip:
    @pytest.mark.parametrize(
        "config",
        [
            VolatilityTargetConfig(annual_vol_target=0.30, capital=250_000.0),
            VolatilityEstimationConfig(span=20, min_periods=5),
            ForecastScalingConfig(window=750),
            BufferingConfig(fraction=0.2),
            CostConfig(cost_per_trade=0.001),
        ],
    )
    def test_flat_configs_round_trip(self, config: object) -> None:
        restored = type(config).from_mapping(config.to_mapping())  # type: ignore[attr-defined]

        assert restored == config

    def test_system_config_round_trips_through_nested_mappings(self) -> None:
        """`dataclasses.asdict` nests, so `from_mapping` must rebuild children."""
        config = SystemConfig(
            volatility_target=VolatilityTargetConfig(annual_vol_target=0.30, capital=250_000.0),
            buffering=BufferingConfig(fraction=0.05),
            costs=CostConfig(cost_per_trade=0.0015),
        )

        restored = SystemConfig.from_mapping(config.to_mapping())

        assert restored == config

    def test_system_config_from_mapping_defaults_absent_sections(self) -> None:
        target = {"annual_vol_target": 0.25, "capital": 1_000_000.0}

        restored = SystemConfig.from_mapping({"volatility_target": target})

        assert restored.buffering == BufferingConfig()

    def test_system_config_from_mapping_requires_a_volatility_target(self) -> None:
        """Capital has no default, so the section carrying it cannot be optional."""
        with pytest.raises(ConfigurationError, match="volatility_target"):
            SystemConfig.from_mapping({"buffering": {"fraction": 0.1}})

    def test_to_mapping_is_plain_data(self) -> None:
        """The mapping must be serialisable, so no config objects may survive in it."""
        mapping = SystemConfig(
            volatility_target=VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000.0)
        ).to_mapping()

        assert isinstance(mapping["volatility_target"], dict)
