# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cash volatility-target helpers."""

from __future__ import annotations

import pytest

from quantrules.config import VolatilityTargetConfig
from quantrules.exceptions import ConfigurationError
from quantrules.sizing.volatility_target import annual_cash_vol_target, periodic_cash_vol_target


class TestAnnualCashVolTarget:
    def test_matches_the_hand_computed_value(self) -> None:
        assert annual_cash_vol_target(capital=1_000_000.0, annual_vol_target=0.25) == 250_000.0

    def test_matches_the_config_property(self) -> None:
        config = VolatilityTargetConfig(capital=750_000.0, annual_vol_target=0.20)
        assert annual_cash_vol_target(capital=750_000.0, annual_vol_target=0.20) == pytest.approx(
            config.annual_cash_vol_target
        )

    def test_rejects_non_positive_capital(self) -> None:
        with pytest.raises(ConfigurationError, match="capital"):
            annual_cash_vol_target(capital=0.0)

    def test_rejects_non_positive_target(self) -> None:
        with pytest.raises(ConfigurationError, match="annual_vol_target"):
            annual_cash_vol_target(capital=1.0, annual_vol_target=-0.1)


class TestPeriodicCashVolTarget:
    def test_matches_the_hand_computed_value(self) -> None:
        # 1_000_000 * 0.25 / sqrt(256) = 250_000 / 16 = 15_625.
        result = periodic_cash_vol_target(
            capital=1_000_000.0, annual_vol_target=0.25, periods_per_year=256
        )
        assert result == pytest.approx(15_625.0)

    def test_matches_the_config_property(self) -> None:
        config = VolatilityTargetConfig(capital=1_000_000.0, annual_vol_target=0.25)
        assert periodic_cash_vol_target(capital=1_000_000.0) == pytest.approx(
            config.periodic_cash_vol_target
        )

    def test_rejects_non_positive_periods_per_year(self) -> None:
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            periodic_cash_vol_target(capital=1.0, periods_per_year=0)
