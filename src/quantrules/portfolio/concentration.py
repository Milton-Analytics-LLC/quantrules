# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
r"""Portfolio concentration measures.

Scalar summaries of how concentrated a set of weights is. They return plain
floats, so there is no time series and nothing to prove causal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantrules._weights import weight_array

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["effective_number_of_instruments", "herfindahl_hirschman_index"]


def herfindahl_hirschman_index(weights: Mapping[str, float]) -> float:
    r"""The Herfindahl-Hirschman Index of a set of weights.

    The sum of squared weights:

    $$\mathrm{HHI} = \sum_i w_i^2$$

    For weights summing to 1 it lies in :math:`[1/N, 1]`: ``1/N`` for ``N`` equal
    weights (maximally diversified) and ``1`` for a single instrument.

    Worked example: equal weights over four instruments give
    :math:`4 \cdot 0.25^2 = 0.25`; ``{0.5, 0.3, 0.2}`` gives
    :math:`0.25 + 0.09 + 0.04 = 0.38`.

    Reference: O. C. Herfindahl (1950) and A. O. Hirschman (1945), the
    concentration index.

    Args:
        weights: Weights keyed by name; non-negative and summing to 1.

    Returns:
        The concentration index.

    Raises:
        ConfigurationError: If any weight is negative or they do not sum to 1.
    """
    _, values = weight_array(weights)
    return float(np.sum(values**2))


def effective_number_of_instruments(weights: Mapping[str, float]) -> float:
    r"""The effective number of instruments, the reciprocal of the concentration index.

    The reciprocal of the
    [`herfindahl_hirschman_index`][quantrules.portfolio.concentration.herfindahl_hirschman_index]:

    $$N_{\text{eff}} = \frac{1}{\mathrm{HHI}} = \frac{1}{\sum_i w_i^2}$$

    It lies in :math:`[1, N]`: ``N`` for ``N`` equal weights and ``1`` for a single
    instrument, reading as "how many equally weighted instruments would give this
    much diversification."

    Worked example: ``{0.5, 0.3, 0.2}`` gives :math:`1 / 0.38 \approx 2.63`.

    Reference: the reciprocal of the Herfindahl-Hirschman index, its
    "numbers-equivalent" reading.

    Args:
        weights: Weights keyed by name; non-negative and summing to 1.

    Returns:
        The effective number of instruments.

    Raises:
        ConfigurationError: If any weight is negative or they do not sum to 1.
    """
    return 1.0 / herfindahl_hirschman_index(weights)
