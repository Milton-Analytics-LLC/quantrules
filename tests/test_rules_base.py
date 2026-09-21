# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the rule adapter, registry and entry-point discovery."""

from __future__ import annotations

import pandas as pd
import pytest

import quantrules as qr
from quantrules.data import MarketData
from quantrules.exceptions import ConfigurationError, DataValidationError
from quantrules.rules import base
from tests.support import business_days, series


class FakeEntryPoint:
    """A stand-in for `importlib.metadata.EntryPoint` that loads a fixed object."""

    def __init__(self, name: str, obj: object) -> None:
        self.name = name
        self._obj = obj

    def load(self) -> object:
        return self._obj


def _price_md() -> MarketData:
    return MarketData(pd.DataFrame({"price": [1.0, 2.0, 3.0]}, index=business_days(3)))


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the module registry for an empty one and skip entry-point loading."""
    monkeypatch.setattr(base, "_REGISTRY", {})
    monkeypatch.setattr(base, "_ENTRY_POINT_STATE", {"loaded": True})


def _entry_points_returning(*points: FakeEntryPoint) -> object:
    """A fake ``entry_points`` that returns ``points`` for the rules group only."""

    def fake(*, group: str) -> list[FakeEntryPoint]:
        return list(points) if group == base.RULE_ENTRY_POINT_GROUP else []

    return fake


@pytest.mark.usefixtures("isolated_registry")
class TestRule:
    def test_calls_the_wrapped_function_with_market_data(self) -> None:
        @base.register_rule("double_price", required_fields=("price",))
        def _double(data: MarketData) -> pd.Series:
            return data["price"] * 2.0

        result = base.get_rule("double_price")(_price_md())
        assert list(result) == [2.0, 4.0, 6.0]

    def test_missing_required_field_raises_when_called(self) -> None:
        @base.register_rule("needs_vol", required_fields=("volatility",))
        def _needs_vol(data: MarketData) -> pd.Series:
            return data["volatility"]

        with pytest.raises(DataValidationError, match="volatility"):
            base.get_rule("needs_vol")(_price_md())

    def test_exposes_name_and_required_fields(self) -> None:
        @base.register_rule("labelled", required_fields=("price", "carry"))
        def _labelled(data: MarketData) -> pd.Series:
            return data["price"]

        rule = base.get_rule("labelled")
        assert rule.name == "labelled"
        assert rule.required_fields == ("price", "carry")


@pytest.mark.usefixtures("isolated_registry")
class TestRegistry:
    def test_register_duplicate_name_raises(self) -> None:
        @base.register_rule("dup")
        def _one(data: MarketData) -> pd.Series:
            return data["price"]

        with pytest.raises(ConfigurationError, match="dup"):

            @base.register_rule("dup")
            def _two(data: MarketData) -> pd.Series:
                return data["price"]

    def test_get_unknown_rule_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown_rule"):
            base.get_rule("unknown_rule")

    def test_available_rules_lists_sorted_names(self) -> None:
        @base.register_rule("zulu")
        def _z(data: MarketData) -> pd.Series:
            return data["price"]

        @base.register_rule("alpha")
        def _a(data: MarketData) -> pd.Series:
            return data["price"]

        assert base.available_rules() == ("alpha", "zulu")


class TestEntryPoints:
    def test_discovers_a_plugin_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base, "_REGISTRY", {})
        monkeypatch.setattr(base, "_ENTRY_POINT_STATE", {"loaded": False})
        plugin = base.Rule(name="plugin_rule", function=lambda data: data["price"])
        monkeypatch.setattr(
            base, "entry_points", _entry_points_returning(FakeEntryPoint("plugin_rule", plugin))
        )
        assert "plugin_rule" in base.available_rules()

    def test_entry_points_are_loaded_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base, "_REGISTRY", {})
        monkeypatch.setattr(base, "_ENTRY_POINT_STATE", {"loaded": False})
        calls = {"n": 0}

        def fake(*, group: str) -> list[FakeEntryPoint]:
            assert group == base.RULE_ENTRY_POINT_GROUP
            calls["n"] += 1
            return []

        monkeypatch.setattr(base, "entry_points", fake)
        base.available_rules()
        base.available_rules()
        assert calls["n"] == 1

    def test_plugin_colliding_with_a_builtin_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base, "_REGISTRY", {"taken": base.Rule("taken", lambda d: d["price"])})
        monkeypatch.setattr(base, "_ENTRY_POINT_STATE", {"loaded": False})
        plugin = base.Rule(name="taken", function=lambda data: data["price"])
        monkeypatch.setattr(
            base, "entry_points", _entry_points_returning(FakeEntryPoint("taken", plugin))
        )
        with pytest.raises(ConfigurationError, match="taken"):
            base.available_rules()

    def test_plugin_loading_a_non_rule_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base, "_REGISTRY", {})
        monkeypatch.setattr(base, "_ENTRY_POINT_STATE", {"loaded": False})
        monkeypatch.setattr(
            base, "entry_points", _entry_points_returning(FakeEntryPoint("bad", object()))
        )
        with pytest.raises(ConfigurationError, match="Rule"):
            base.available_rules()


class TestBuiltins:
    def test_standard_ewmac_speeds_are_registered(self) -> None:
        available = base.available_rules()
        assert "ewmac16_64" in available
        assert "ewmac2_8" in available

    def test_a_builtin_ewmac_rule_runs_on_market_data(self) -> None:
        idx = business_days(6)
        frame = pd.DataFrame(
            {"price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "volatility": [2.0] * 6}, index=idx
        )
        result = base.get_rule("ewmac2_8")(MarketData(frame))
        assert result.index.equals(idx)
        assert result.iloc[-1] > 0.0

    def test_the_builtin_carry_rule_runs_on_market_data(self) -> None:
        idx = business_days(4)
        frame = pd.DataFrame(
            {"raw_carry": [2.0, 4.0, 6.0, 8.0], "volatility": [2.0] * 4}, index=idx
        )
        result = base.get_rule("carry")(MarketData(frame))
        assert result.index.equals(idx)
        assert result.iloc[0] == pytest.approx(1.0)


class TestPackageApi:
    def test_market_data_is_exposed_at_top_level(self) -> None:
        assert qr.MarketData is MarketData

    def test_rule_functions_are_exposed_under_rules(self) -> None:
        price = series([1.0, 2.0, 3.0])
        vol = pd.Series(2.0, index=price.index, dtype="float64")
        direct = qr.rules.ewmac(price, fast_span=1, slow_span=2, volatility=vol)
        assert direct.iloc[2] == pytest.approx(2.0 / 9.0)
