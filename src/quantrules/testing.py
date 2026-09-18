# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Testing utilities for quantrules and for libraries that extend it.

The central utility is [`assert_causal`][quantrules.testing.assert_causal],
which proves that a time-series function uses no information from the future.

It works by comparison rather than inspection. For a function `f` with no
look-ahead, the value computed at time *t* cannot depend on anything after *t*,
so running `f` on history truncated at *t* must give the same answer at *t* as
running it on the full history:

```text
f(x[:t])[-1] == f(x)[t]      for every t
```

Any violation - a full-sample mean, a centred window, a negative shift, a
backward fill - breaks that identity.

If you ship your own rules through the `quantrules.rules` entry point, run this
against them too:

```python
from quantrules.testing import assert_causal

def test_my_rule_has_no_look_ahead(prices):
    assert_causal(my_rule, prices)
```
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

__all__ = ["assert_causal", "default_checkpoints"]

Panel: TypeAlias = "pd.Series[float] | pd.DataFrame"
CausalFunction: TypeAlias = Callable[[Any], Panel]

_MIN_OBSERVATIONS = 2
_DEFAULT_CHECKPOINT_COUNT = 12
_DEFAULT_RTOL = 1e-9
_DEFAULT_ATOL = 1e-12


def default_checkpoints(periods: int, count: int = _DEFAULT_CHECKPOINT_COUNT) -> list[int]:
    """Return evenly spaced positions to test, over the back two thirds of a series.

    The early part of a series is skipped because most functions are still in
    their warmup there and compare NaN to NaN, which proves nothing.

    Args:
        periods: Length of the series.
        count: Maximum number of positions to return.

    Returns:
        Sorted, unique integer positions.

    Raises:
        ValueError: If ``periods`` is less than two.
    """
    if periods < _MIN_OBSERVATIONS:
        message = (
            f"a causality check needs at least {_MIN_OBSERVATIONS} observations, got {periods}"
        )
        raise ValueError(message)
    start = periods // 3
    positions = np.linspace(start, periods - 1, num=min(count, periods - start), dtype=int)
    return sorted(set(positions.tolist()))


def assert_causal(
    function: CausalFunction,
    data: Panel,
    *,
    checkpoints: Sequence[int] | None = None,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    name: str | None = None,
) -> None:
    """Assert that ``function`` uses no information from after each timestamp.

    Pass every input the function needs as columns of one `DataFrame`, so that
    truncating the history truncates all of them together. Testing a multi-input
    function against pre-computed full-history inputs would hide look-ahead in
    those inputs.

    Args:
        function: The function under test. Takes the same type as ``data`` and
            returns a `Series` or `DataFrame` sharing its input's index.
        data: Input history, as a `Series` or a `DataFrame` of several inputs.
        checkpoints: Integer positions to test. Defaults to
            [`default_checkpoints`][quantrules.testing.default_checkpoints].
        rtol: Relative tolerance when comparing values.
        atol: Absolute tolerance when comparing values.
        name: Name used in failure messages. Defaults to the function's own.

    Raises:
        AssertionError: If the function looks ahead, or if its output does not
            share its input's index.
        ValueError: If ``data`` is empty or a checkpoint lies outside it.
    """
    label = name or getattr(function, "__name__", None) or repr(function)
    periods = len(data)
    if periods == 0:
        message = f"{label}: cannot check causality on an empty series"
        raise ValueError(message)

    positions = list(checkpoints) if checkpoints is not None else default_checkpoints(periods)
    for position in positions:
        if not 0 <= position < periods:
            message = (
                f"{label}: checkpoint {position} lies outside a series of {periods} observations"
            )
            raise ValueError(message)

    full = function(data)
    _assert_index_preserved(full, data, label, "the full history")

    for position in positions:
        window = data.iloc[: position + 1]
        partial = function(window)
        _assert_index_preserved(partial, window, label, f"history truncated at {position}")
        _assert_last_value_matches(full, partial, position, label, rtol, atol)


def _assert_index_preserved(output: Panel, source: Panel, label: str, described: str) -> None:
    """Require that ``output`` carries exactly the index of ``source``."""
    if not output.index.equals(source.index):
        message = (
            f"{label}: output index does not match the input index on {described}. "
            f"quantrules functions must return the index they were given, padding "
            f"warmup with NaN rather than dropping or reindexing rows "
            f"(got {len(output)} rows for {len(source)} inputs)."
        )
        raise AssertionError(message)


def _assert_last_value_matches(
    full: Panel,
    partial: Panel,
    position: int,
    label: str,
    rtol: float,
    atol: float,
) -> None:
    """Compare the value at ``position`` computed from full and truncated history."""
    timestamp = full.index[position]

    if isinstance(full, pd.DataFrame):
        partial_frame = pd.DataFrame(partial)
        for column in full.columns:
            _assert_scalar_matches(
                float(full[column].iloc[position]),
                float(partial_frame[column].iloc[-1]),
                label=f"{label}[{column}]",
                timestamp=timestamp,
                position=position,
                rtol=rtol,
                atol=atol,
            )
        return

    _assert_scalar_matches(
        float(full.iloc[position]),
        float(pd.Series(partial).iloc[-1]),
        label=label,
        timestamp=timestamp,
        position=position,
        rtol=rtol,
        atol=atol,
    )


def _assert_scalar_matches(
    from_full_history: float,
    from_truncated_history: float,
    *,
    label: str,
    timestamp: object,
    position: int,
    rtol: float,
    atol: float,
) -> None:
    """Raise unless two values agree, treating NaN as equal to NaN."""
    full_is_nan = np.isnan(from_full_history)
    partial_is_nan = np.isnan(from_truncated_history)
    if full_is_nan and partial_is_nan:
        return
    both_are_numbers = not full_is_nan and not partial_is_nan
    if both_are_numbers and np.isclose(
        from_full_history, from_truncated_history, rtol=rtol, atol=atol
    ):
        return

    message = (
        f"{label}: look-ahead detected at {timestamp} (position {position}). "
        f"Computed from the full history the value is {from_full_history!r}, but "
        f"from history truncated at that timestamp it is {from_truncated_history!r}. "
        f"A function with no look-ahead gives the same answer either way."
    )
    raise AssertionError(message)
