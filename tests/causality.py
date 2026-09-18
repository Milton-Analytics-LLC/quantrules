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

from quantrules.testing import Panel

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
CAUSAL_CASES: list[CausalCase] = []


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
