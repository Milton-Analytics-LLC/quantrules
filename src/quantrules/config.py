# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration objects.

Configuration is plain frozen dataclasses rather than a validation framework, so
that quantrules adds nothing to your dependency tree beyond `pandas` and
`numpy`. Every object validates itself on construction and names the offending
parameter when it refuses.

Being frozen, configs are hashable and safe to share between threads, and can be
used as cache keys.

Each config converts to and from a plain mapping, so a system can be described
in YAML or JSON and loaded without any extra machinery:

```python
config = SystemConfig.from_mapping(yaml.safe_load(text))
```
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from quantrules import defaults
from quantrules._validation import (
    ensure_fraction,
    ensure_positive,
    ensure_positive_int,
)
from quantrules.exceptions import ConfigurationError

__all__ = [
    "BufferingConfig",
    "ForecastScalingConfig",
    "SystemConfig",
    "VolatilityEstimationConfig",
    "VolatilityTargetConfig",
]


@dataclass(frozen=True, slots=True)
class VolatilityTargetConfig:
    """How much risk the system as a whole is asked to take.

    Attributes:
        annual_vol_target: Target annualised standard deviation of returns, as a
            fraction. `0.25` means 25% a year.
        capital: Capital the target applies to, in account currency.
        periods_per_year: Observations in a year, used to de-annualise the
            target. Defaults to business days.
    """

    annual_vol_target: float = defaults.ANNUAL_VOL_TARGET
    capital: float = 1.0
    periods_per_year: int = defaults.BUSINESS_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        """Validate and normalise the supplied values."""
        set_ = object.__setattr__
        set_(
            self, "annual_vol_target", ensure_positive(self.annual_vol_target, "annual_vol_target")
        )
        set_(self, "capital", ensure_positive(self.capital, "capital"))
        set_(
            self, "periods_per_year", ensure_positive_int(self.periods_per_year, "periods_per_year")
        )

    @property
    def annual_cash_vol_target(self) -> float:
        """Cash standard deviation of returns targeted over a year."""
        return self.capital * self.annual_vol_target

    @property
    def periodic_cash_vol_target(self) -> float:
        """Cash standard deviation of returns targeted over one period.

        Volatility scales with the square root of time, so the annual target is
        divided by the square root of the number of periods in a year.
        """
        return self.annual_cash_vol_target / math.sqrt(self.periods_per_year)

    def to_mapping(self) -> dict[str, Any]:
        """Return this config as a plain, serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> VolatilityTargetConfig:
        """Build a config from a plain mapping, as produced by `to_mapping`."""
        return cls(**mapping)


@dataclass(frozen=True, slots=True)
class VolatilityEstimationConfig:
    """How return volatility is estimated.

    Attributes:
        span: Span of the exponentially weighted estimate.
        min_periods: Observations required before an estimate is emitted.
        floor_percentile: Percentile of past volatility used as a floor on the
            current estimate. `0.0` disables the floor.
        floor_window: Observations over which that percentile is measured.
        periods_per_year: Observations in a year, used to annualise.
    """

    span: int = defaults.VOLATILITY_EWMA_SPAN
    min_periods: int = defaults.VOLATILITY_MIN_PERIODS
    floor_percentile: float = defaults.VOLATILITY_FLOOR_PERCENTILE
    floor_window: int = defaults.VOLATILITY_FLOOR_WINDOW
    periods_per_year: int = defaults.BUSINESS_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        """Validate and normalise the supplied values."""
        set_ = object.__setattr__
        set_(self, "span", ensure_positive_int(self.span, "span"))
        set_(self, "min_periods", ensure_positive_int(self.min_periods, "min_periods"))
        set_(self, "floor_percentile", ensure_fraction(self.floor_percentile, "floor_percentile"))
        set_(self, "floor_window", ensure_positive_int(self.floor_window, "floor_window"))
        set_(
            self, "periods_per_year", ensure_positive_int(self.periods_per_year, "periods_per_year")
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return this config as a plain, serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> VolatilityEstimationConfig:
        """Build a config from a plain mapping, as produced by `to_mapping`."""
        return cls(**mapping)


@dataclass(frozen=True, slots=True)
class ForecastScalingConfig:
    """How raw forecasts are scaled to a common size and capped.

    Attributes:
        target_avg_abs_forecast: Average absolute value a scaled forecast should
            have over the long run.
        cap: Absolute value at which the scaled forecast is clipped. Must be at
            least `target_avg_abs_forecast`.
        min_periods: Observations required before an estimated scalar is emitted.
        window: Observations in the estimation window, or `None` to use an
            expanding window that grows with the history.
    """

    target_avg_abs_forecast: float = defaults.TARGET_AVG_ABS_FORECAST
    cap: float = defaults.FORECAST_CAP
    min_periods: int = defaults.SCALAR_ESTIMATION_MIN_PERIODS
    window: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the supplied values."""
        set_ = object.__setattr__
        set_(
            self,
            "target_avg_abs_forecast",
            ensure_positive(self.target_avg_abs_forecast, "target_avg_abs_forecast"),
        )
        set_(self, "cap", ensure_positive(self.cap, "cap"))
        set_(self, "min_periods", ensure_positive_int(self.min_periods, "min_periods"))
        if self.window is not None:
            set_(self, "window", ensure_positive_int(self.window, "window"))

        if self.cap < self.target_avg_abs_forecast:
            message = (
                f"cap must be at least target_avg_abs_forecast, got cap={self.cap} "
                f"and target_avg_abs_forecast={self.target_avg_abs_forecast}"
            )
            raise ConfigurationError(message)

    def to_mapping(self) -> dict[str, Any]:
        """Return this config as a plain, serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ForecastScalingConfig:
        """Build a config from a plain mapping, as produced by `to_mapping`."""
        return cls(**mapping)


@dataclass(frozen=True, slots=True)
class BufferingConfig:
    """How far a position may drift before it is worth trading.

    Attributes:
        fraction: Half-width of the no-trade buffer, as a fraction of the
            average position. `0.0` disables buffering and trades to the optimal
            position every period.
    """

    fraction: float = defaults.BUFFER_FRACTION

    def __post_init__(self) -> None:
        """Validate and normalise the supplied values."""
        object.__setattr__(self, "fraction", ensure_fraction(self.fraction, "fraction"))

    def to_mapping(self) -> dict[str, Any]:
        """Return this config as a plain, serialisable mapping."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> BufferingConfig:
        """Build a config from a plain mapping, as produced by `to_mapping`."""
        return cls(**mapping)


@dataclass(frozen=True, slots=True)
class SystemConfig:
    """Every configurable part of a system, in one object.

    Only `volatility_target` is required: it carries your capital, which has no
    sensible default. The rest fall back to the library defaults.

    Attributes:
        volatility_target: Risk target and capital.
        volatility_estimation: How return volatility is estimated.
        forecast_scaling: How forecasts are scaled and capped.
        buffering: How far positions may drift before trading.
    """

    volatility_target: VolatilityTargetConfig
    volatility_estimation: VolatilityEstimationConfig = field(
        default_factory=VolatilityEstimationConfig
    )
    forecast_scaling: ForecastScalingConfig = field(default_factory=ForecastScalingConfig)
    buffering: BufferingConfig = field(default_factory=BufferingConfig)

    def to_mapping(self) -> dict[str, Any]:
        """Return this config, and every config nested in it, as plain mappings."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SystemConfig:
        """Build a system config from nested plain mappings.

        Sections absent from ``mapping`` fall back to their defaults.

        Args:
            mapping: Nested mapping, as produced by `to_mapping`. Must contain a
                ``volatility_target`` section.

        Returns:
            The reconstructed config.

        Raises:
            ConfigurationError: If the ``volatility_target`` section is missing.
        """
        if "volatility_target" not in mapping:
            message = "mapping must contain a 'volatility_target' section"
            raise ConfigurationError(message)

        sections: dict[str, Any] = {
            "volatility_target": VolatilityTargetConfig.from_mapping(mapping["volatility_target"])
        }
        if "volatility_estimation" in mapping:
            sections["volatility_estimation"] = VolatilityEstimationConfig.from_mapping(
                mapping["volatility_estimation"]
            )
        if "forecast_scaling" in mapping:
            sections["forecast_scaling"] = ForecastScalingConfig.from_mapping(
                mapping["forecast_scaling"]
            )
        if "buffering" in mapping:
            sections["buffering"] = BufferingConfig.from_mapping(mapping["buffering"])
        return cls(**sections)
