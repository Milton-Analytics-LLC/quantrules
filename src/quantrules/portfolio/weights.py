# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Instrument weights.

Two ways to build the instrument weights the portfolio layer consumes: equal
weights, or normalising a set of hand-chosen weights. Both return a plain ``dict``
summing to 1 (cross-sectional, so there is nothing to prove causal). The
correlation-driven alternative, handcrafting, arrives in a later release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantrules._weights import positive_weight_array
from quantrules.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["equal_weights", "normalize_weights"]


def equal_weights(instruments: Sequence[str]) -> dict[str, float]:
    r"""Equal weights across ``instruments``.

    Each of the ``N`` instruments gets :math:`1/N`.

    Worked example: ``["a", "b", "c", "d"]`` gives ``0.25`` each.

    Args:
        instruments: The instrument names.

    Returns:
        A mapping of each instrument to ``1/N``.

    Raises:
        ConfigurationError: If ``instruments`` is empty or contains duplicates.
    """
    names = list(instruments)
    if not names:
        message = "instruments must contain at least one instrument"
        raise ConfigurationError(message)
    if len(set(names)) != len(names):
        message = f"instruments must not contain duplicates, got {names}"
        raise ConfigurationError(message)
    return dict.fromkeys(names, 1.0 / len(names))


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    r"""Scale ``weights`` to sum to 1, preserving their proportions.

    $$w_i' = \frac{w_i}{\sum_j w_j}$$

    Worked example: ``{"a": 2, "b": 1, "c": 1}`` gives
    ``{"a": 0.5, "b": 0.25, "c": 0.25}``.

    Args:
        weights: Non-negative weights keyed by name, with a positive sum.

    Returns:
        The weights scaled to sum to 1.

    Raises:
        ConfigurationError: If any weight is negative or the sum is not positive.
    """
    keys, values = positive_weight_array(weights)
    total = float(values.sum())
    return {key: float(value) / total for key, value in zip(keys, values, strict=True)}
