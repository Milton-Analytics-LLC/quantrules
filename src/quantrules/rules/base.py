# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""The rule adapter, in-process registry and entry-point discovery.

The rule maths in this package are container-free functions of `pandas` objects.
A [`Rule`][quantrules.rules.base.Rule] is the thin adapter that binds such a
function to a [`MarketData`][quantrules.data.MarketData] container, declaring the
fields it consumes so a system can run any rule over any instrument uniformly.

Rules are registered by name, either in-process with
[`register_rule`][quantrules.rules.base.register_rule] or, from a separate
distribution, through the ``quantrules.rules`` entry-point group. Entry-point
discovery is *lazy* - it runs on the first registry lookup, not at import - so
``import quantrules`` stays fast and a broken third-party plugin cannot break the
import. A third-party name that collides with an existing rule raises rather than
silently shadowing it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, TypeAlias

from quantrules._typing import FloatSeries
from quantrules.exceptions import ConfigurationError

if TYPE_CHECKING:
    from quantrules.data import MarketData

__all__ = ["Rule", "available_rules", "get_rule", "register_rule"]

RuleFunction: TypeAlias = Callable[["MarketData"], FloatSeries]

RULE_ENTRY_POINT_GROUP = "quantrules.rules"

_REGISTRY: dict[str, Rule] = {}
# A mutable holder rather than a bare bool so the lazy load can flip it without a
# module-level ``global`` statement.
_ENTRY_POINT_STATE: dict[str, bool] = {"loaded": False}


@dataclass(frozen=True)
class Rule:
    """A named trading rule: a `MarketData`-consuming function and its fields.

    Attributes:
        name: The name the rule is registered under.
        function: A callable mapping a `MarketData` to a forecast series.
        required_fields: The `MarketData` fields the rule reads. Checked before
            the function runs, so a missing field fails with a clear message.
    """

    name: str
    function: RuleFunction
    required_fields: tuple[str, ...] = field(default=())

    def __call__(self, data: MarketData) -> FloatSeries:
        """Run the rule, first asserting every required field is present."""
        data.require(self.required_fields)
        return self.function(data)


def register_rule(
    name: str,
    *,
    required_fields: Iterable[str] = (),
) -> Callable[[RuleFunction], RuleFunction]:
    """Register a `MarketData`-consuming function as a named rule.

    Use as a decorator::

        @register_rule("my_rule", required_fields=("price",))
        def my_rule(data: MarketData) -> pd.Series: ...

    The decorated function is returned unchanged, so it stays directly callable;
    the wrapping [`Rule`][quantrules.rules.base.Rule] is retrievable with
    [`get_rule`][quantrules.rules.base.get_rule].

    Args:
        name: The name to register under. Must be unique.
        required_fields: The `MarketData` fields the rule reads.

    Returns:
        A decorator that registers its function and returns it unchanged.

    Raises:
        ConfigurationError: If ``name`` is already registered.
    """

    def decorator(function: RuleFunction) -> RuleFunction:
        _register(Rule(name=name, function=function, required_fields=tuple(required_fields)))
        return function

    return decorator


def get_rule(name: str) -> Rule:
    """Return the registered rule named ``name``.

    Triggers lazy entry-point discovery on first use.

    Args:
        name: The registered name.

    Returns:
        The rule.

    Raises:
        ConfigurationError: If no rule is registered under ``name``.
    """
    _load_entry_points()
    try:
        return _REGISTRY[name]
    except KeyError:
        message = f"no rule registered under {name!r}; available: {sorted(_REGISTRY)}"
        raise ConfigurationError(message) from None


def available_rules() -> tuple[str, ...]:
    """Return the names of every registered rule, sorted.

    Triggers lazy entry-point discovery on first use.
    """
    _load_entry_points()
    return tuple(sorted(_REGISTRY))


def _register(rule: Rule) -> None:
    """Add ``rule`` to the registry, refusing to shadow an existing name."""
    if rule.name in _REGISTRY:
        message = f"a rule named {rule.name!r} is already registered; names must be unique"
        raise ConfigurationError(message)
    _REGISTRY[rule.name] = rule


def _load_entry_points() -> None:
    """Discover and register rules advertised on the ``quantrules.rules`` group."""
    if _ENTRY_POINT_STATE["loaded"]:
        return
    _ENTRY_POINT_STATE["loaded"] = True
    for entry_point in entry_points(group=RULE_ENTRY_POINT_GROUP):
        loaded = entry_point.load()
        if not isinstance(loaded, Rule):
            message = (
                f"entry point {entry_point.name!r} in group {RULE_ENTRY_POINT_GROUP!r} "
                f"must load a Rule, got {type(loaded).__name__}"
            )
            raise ConfigurationError(message)
        _register(loaded)
