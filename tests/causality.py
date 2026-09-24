# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Registry of the causality checks that cover the public API.

`CAUSAL_CASES` lists one case per public time-series function. A meta-test in
`test_causality_registry.py` compares this list against the functions the
package actually exports, so a new indicator or rule cannot ship without a
causality check.

To register a function, append a `CausalCase` naming it exactly as
`discover_series_functions` reports it: the package's own name, the module path
beneath it, then the function - for example
``"quantrules.indicators.trend.macd"``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from quantrules.correlation import rolling_correlation
from quantrules.forecasts.capping import cap
from quantrules.forecasts.combine import combine
from quantrules.forecasts.diversification import forecast_diversification_multiplier
from quantrules.forecasts.scaling import forecast_scalar, scale
from quantrules.indicators import (
    averages,
    breadth,
    channels,
    events,
    normalize,
    oscillators,
    trend,
    volatility,
    volume,
)
from quantrules.rules.breakout import breakout as breakout_rule
from quantrules.rules.carry import carry as carry_rule
from quantrules.rules.ewmac import ewmac as ewmac_rule
from quantrules.rules.mean_reversion import mean_reversion as mean_reversion_rule
from quantrules.sizing.buffering import buffer_position
from quantrules.sizing.instrument_vol import instrument_value_volatility, instrument_volatility
from quantrules.sizing.position import subsystem_position
from quantrules.testing import Panel
from tests.support import (
    breadth_data,
    buffering_inputs,
    carry_data,
    crossing_pair,
    events_series,
    forecast_panel,
    instrument_data,
    ohlcv,
    position_inputs,
    random_walk,
)

_EQUAL_WEIGHTS = {"rule_0": 1.0 / 3.0, "rule_1": 1.0 / 3.0, "rule_2": 1.0 / 3.0}

_TIME_SERIES_RETURN_MARKERS = ("Series", "DataFrame", "Frame")


@dataclass(frozen=True)
class CausalCase:
    """One registered causality check.

    Attributes:
        name: Dotted name of the function, as `discover_series_functions`
            reports it.
        function: The function under test, adapted to take a single `Series` or
            `DataFrame` of every input it needs.
        build_data: Callable returning deterministic input data for the check.
    """

    name: str
    function: Callable[[Any], Panel]
    build_data: Callable[[], Panel]


# Populated from phase 2 onward, one entry per public time-series function.
CAUSAL_CASES: list[CausalCase] = [
    CausalCase(
        "quantrules.indicators.averages.sma",
        lambda data: averages.sma(data, window=10),
        lambda: random_walk(120, seed=101),
    ),
    CausalCase(
        "quantrules.indicators.averages.ewma",
        lambda data: averages.ewma(data, span=10),
        lambda: random_walk(120, seed=102),
    ),
    CausalCase(
        "quantrules.indicators.volatility.rolling_std",
        lambda data: volatility.rolling_std(data, window=10),
        lambda: random_walk(120, seed=203),
    ),
    CausalCase(
        "quantrules.indicators.volatility.ewma_volatility",
        lambda data: volatility.ewma_volatility(data, span=36, min_periods=10),
        lambda: random_walk(150, seed=204),
    ),
    CausalCase(
        "quantrules.indicators.volatility.realized_volatility",
        lambda data: volatility.realized_volatility(data, window=21),
        lambda: random_walk(150, seed=205),
    ),
    CausalCase(
        "quantrules.indicators.volatility.true_range",
        lambda df: volatility.true_range(df["high"], df["low"], df["close"]),
        lambda: ohlcv(150, seed=206),
    ),
    CausalCase(
        "quantrules.indicators.volatility.atr",
        lambda df: volatility.atr(df["high"], df["low"], df["close"], window=14),
        lambda: ohlcv(150, seed=207),
    ),
    CausalCase(
        "quantrules.indicators.volatility.atr_percent",
        lambda df: volatility.atr_percent(df["high"], df["low"], df["close"], window=14),
        lambda: ohlcv(150, seed=208),
    ),
    CausalCase(
        "quantrules.indicators.trend.macd",
        lambda data: trend.macd(data, fast=12, slow=26, signal=9),
        lambda: random_walk(120, seed=301),
    ),
    CausalCase(
        "quantrules.indicators.trend.ewma_crossover",
        lambda data: trend.ewma_crossover(data, fast=8, slow=32),
        lambda: random_walk(120, seed=302),
    ),
    CausalCase(
        "quantrules.indicators.trend.rolling_slope",
        lambda data: trend.rolling_slope(data, window=10),
        lambda: random_walk(120, seed=303),
    ),
    CausalCase(
        "quantrules.indicators.channels.rolling_max",
        lambda data: channels.rolling_max(data, window=10),
        lambda: random_walk(120, seed=401),
    ),
    CausalCase(
        "quantrules.indicators.channels.rolling_min",
        lambda data: channels.rolling_min(data, window=10),
        lambda: random_walk(120, seed=402),
    ),
    CausalCase(
        "quantrules.indicators.channels.donchian_channels",
        lambda data: channels.donchian_channels(data, window=10),
        lambda: random_walk(120, seed=403),
    ),
    CausalCase(
        "quantrules.indicators.channels.bollinger_bands",
        lambda data: channels.bollinger_bands(data, window=20),
        lambda: random_walk(120, seed=404),
    ),
    CausalCase(
        "quantrules.indicators.channels.keltner_channels",
        lambda df: channels.keltner_channels(df["high"], df["low"], df["close"], window=20),
        lambda: ohlcv(150, seed=405),
    ),
    CausalCase(
        "quantrules.indicators.channels.keltner_position",
        lambda df: channels.keltner_position(df["high"], df["low"], df["close"], window=20),
        lambda: ohlcv(150, seed=406),
    ),
    CausalCase(
        "quantrules.indicators.oscillators.rsi",
        lambda data: oscillators.rsi(data, window=14),
        lambda: random_walk(150, seed=501),
    ),
    CausalCase(
        "quantrules.indicators.normalize.zscore",
        lambda data: normalize.zscore(data, window=20),
        lambda: random_walk(120, seed=601),
    ),
    CausalCase(
        "quantrules.indicators.normalize.clip",
        lambda data: normalize.clip(data, lower=90.0, upper=110.0),
        lambda: random_walk(120, seed=602),
    ),
    CausalCase(
        "quantrules.indicators.normalize.rolling_rank",
        lambda data: normalize.rolling_rank(data, window=20),
        lambda: random_walk(120, seed=603),
    ),
    CausalCase(
        "quantrules.indicators.normalize.rolling_percentile",
        lambda data: normalize.rolling_percentile(data, window=20),
        lambda: random_walk(120, seed=604),
    ),
    CausalCase(
        "quantrules.indicators.volume.obv",
        lambda df: volume.obv(df["close"], df["volume"]),
        lambda: ohlcv(150, seed=801),
    ),
    CausalCase(
        "quantrules.indicators.volume.accumulation_distribution",
        lambda df: volume.accumulation_distribution(
            df["high"], df["low"], df["close"], df["volume"]
        ),
        lambda: ohlcv(150, seed=802),
    ),
    CausalCase(
        "quantrules.indicators.volume.chaikin_money_flow",
        lambda df: volume.chaikin_money_flow(
            df["high"], df["low"], df["close"], df["volume"], window=20
        ),
        lambda: ohlcv(150, seed=803),
    ),
    CausalCase(
        "quantrules.indicators.breadth.advance_decline_line",
        lambda df: breadth.advance_decline_line(df["advances"], df["declines"]),
        lambda: breadth_data(150, seed=901),
    ),
    CausalCase(
        "quantrules.indicators.breadth.mcclellan_oscillator",
        lambda df: breadth.mcclellan_oscillator(df["advances"], df["declines"]),
        lambda: breadth_data(150, seed=902),
    ),
    CausalCase(
        "quantrules.indicators.breadth.new_high_new_low_index",
        lambda df: breadth.new_high_new_low_index(df["new_highs"], df["new_lows"]),
        lambda: breadth_data(150, seed=903),
    ),
    CausalCase(
        "quantrules.indicators.breadth.trin",
        lambda df: breadth.trin(
            df["advances"], df["declines"], df["advancing_volume"], df["declining_volume"]
        ),
        lambda: breadth_data(150, seed=904),
    ),
    CausalCase(
        "quantrules.indicators.events.cross_up",
        lambda df: events.cross_up(df["fast"], df["slow"]),
        lambda: crossing_pair(150, seed=1001),
    ),
    CausalCase(
        "quantrules.indicators.events.cross_down",
        lambda df: events.cross_down(df["fast"], df["slow"]),
        lambda: crossing_pair(150, seed=1002),
    ),
    CausalCase(
        "quantrules.indicators.events.bars_since",
        events.bars_since,
        lambda: events_series(150, seed=1003),
    ),
    CausalCase(
        "quantrules.rules.ewmac.ewmac",
        lambda df: ewmac_rule(
            df["price"],
            fast_span=8,
            slow_span=32,
            volatility=volatility.rolling_std(df["price"], window=32),
        ),
        lambda: random_walk(200, seed=1101).to_frame("price"),
    ),
    CausalCase(
        "quantrules.rules.carry.carry",
        lambda df: carry_rule(df["raw_carry"], volatility=df["volatility"], smoothing_span=90),
        lambda: carry_data(200, seed=1102),
    ),
    CausalCase(
        "quantrules.rules.breakout.breakout",
        lambda data: breakout_rule(data, span=20, smoothing_span=10),
        lambda: random_walk(150, seed=1103),
    ),
    CausalCase(
        "quantrules.rules.mean_reversion.mean_reversion",
        lambda data: mean_reversion_rule(data, window=20),
        lambda: random_walk(150, seed=1104),
    ),
    CausalCase(
        "quantrules.correlation.rolling_correlation",
        lambda df: rolling_correlation(df["fast"], df["slow"], min_periods=5),
        lambda: crossing_pair(150, seed=1201),
    ),
    CausalCase(
        "quantrules.forecasts.scaling.forecast_scalar",
        lambda data: forecast_scalar(data, target=10.0, min_periods=20),
        lambda: random_walk(200, seed=1202),
    ),
    CausalCase(
        "quantrules.forecasts.scaling.scale",
        lambda data: scale(data, target=10.0, min_periods=20),
        lambda: random_walk(200, seed=1203),
    ),
    CausalCase(
        "quantrules.forecasts.capping.cap",
        lambda data: cap(data, cap=120.0),
        lambda: random_walk(150, seed=1204),
    ),
    CausalCase(
        "quantrules.forecasts.diversification.forecast_diversification_multiplier",
        lambda df: forecast_diversification_multiplier(df, _EQUAL_WEIGHTS, min_periods=20),
        lambda: forecast_panel(200, seed=1205),
    ),
    CausalCase(
        "quantrules.forecasts.combine.combine",
        lambda df: combine(df, _EQUAL_WEIGHTS, min_periods=20),
        lambda: forecast_panel(200, seed=1206),
    ),
    CausalCase(
        "quantrules.sizing.instrument_vol.instrument_volatility",
        lambda data: instrument_volatility(
            data, span=36, min_periods=10, floor_window=20, floor_percentile=0.1
        ),
        lambda: random_walk(200, seed=1301),
    ),
    CausalCase(
        "quantrules.sizing.instrument_vol.instrument_value_volatility",
        lambda df: instrument_value_volatility(
            df["price"], df["volatility"], block_size=10.0, exchange_rate=1.5
        ),
        lambda: instrument_data(200, seed=1302),
    ),
    CausalCase(
        "quantrules.sizing.position.subsystem_position",
        lambda df: subsystem_position(
            df["forecast"], df["instrument_value_volatility"], cash_vol_target=1000.0
        ),
        lambda: position_inputs(200, seed=1303),
    ),
    CausalCase(
        "quantrules.sizing.buffering.buffer_position",
        lambda df: buffer_position(df["optimal"], df["average_position"], fraction=0.1),
        lambda: buffering_inputs(200, seed=1304),
    ),
]


def returns_time_series(candidate: object) -> bool:
    """Return whether ``candidate`` is a function annotated to return a Series or Frame.

    The check reads the annotation as written rather than resolving it, so a
    module using `from __future__ import annotations` needs no special handling
    and an unresolvable forward reference cannot break discovery.

    Args:
        candidate: Any object; non-callables return `False`.

    Returns:
        `True` if the object is a callable whose return annotation names a
        pandas `Series` or `DataFrame`.
    """
    if not callable(candidate):
        return False
    try:
        annotation = inspect.signature(candidate).return_annotation
    except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
        return False
    if annotation is inspect.Signature.empty:
        return False
    return any(marker in str(annotation) for marker in _TIME_SERIES_RETURN_MARKERS)


def discover_series_functions(package: ModuleType) -> set[str]:
    """Return the dotted names of every public time-series function in ``package``.

    Walks the package and its subpackages, skipping modules whose name begins
    with an underscore, and inspects each module's ``__all__``. Names are
    reported as the package's own name followed by the module path beneath it,
    so ``quantrules.indicators.trend`` contributes
    ``"quantrules.indicators.trend.macd"``.

    Args:
        package: The imported package to walk.

    Returns:
        A set of dotted function names.
    """
    root = package.__name__
    base_name = root.rsplit(".", maxsplit=1)[-1]
    found: set[str] = set()

    for module in _walk_public_modules(package):
        relative = module.__name__[len(root) + 1 :]
        for attribute in getattr(module, "__all__", []):
            value = getattr(module, attribute, None)
            if not returns_time_series(value):
                continue
            qualified = f"{relative}.{attribute}" if relative else attribute
            found.add(f"{base_name}.{qualified}")

    return found


def _walk_public_modules(package: ModuleType) -> list[ModuleType]:
    """Return ``package`` and every non-private module beneath it."""
    modules = [package]
    search_paths = getattr(package, "__path__", None)
    if search_paths is None:  # pragma: no cover - a plain module, not a package
        return modules

    for info in pkgutil.walk_packages(search_paths, prefix=f"{package.__name__}."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        modules.append(importlib.import_module(info.name))
    return modules
