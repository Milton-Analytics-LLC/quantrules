# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""The `MarketData` container.

A `MarketData` bundles one instrument's data as named fields (``price``,
``high``, ``low``, ``close``, ``carry`` and so on) on a single shared, validated
`DatetimeIndex`. It is the uniform input a registered
[`Rule`][quantrules.rules.base.Rule] consumes, so a system can apply any rule to
any instrument without per-rule glue. The container-free rule maths in
`quantrules.rules` never see it: only the thin `Rule` adapter does.

The container validates once, at construction, and never mutates the caller's
data. Field access returns a copy, so a rule cannot reach back and corrupt the
container or the original frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from quantrules._typing import FloatSeries, Frame
from quantrules._validation import ensure_aligned, ensure_frame, ensure_series
from quantrules.exceptions import DataValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["MarketData"]


class MarketData:
    """An instrument's named data fields on one shared `DatetimeIndex`.

    Each field is validated against the quantrules data contract (a unique,
    monotonic `DatetimeIndex` and a numeric dtype) and stored as `float64`. All
    fields must share one index.

    Args:
        frame: A `DataFrame` whose columns are the fields, on a unique, monotonic
            `DatetimeIndex`. Must have at least one column.

    Raises:
        DataValidationError: If ``frame`` is not a `DataFrame`, has no columns, or
            any column violates the data contract.
    """

    def __init__(self, frame: object) -> None:
        """Validate ``frame`` and store its columns as `float64` fields.

        ``frame`` is deliberately typed ``object``: this is the boundary at which
        an unknown value becomes a validated container.
        """
        self._frame: Frame = ensure_frame(frame, "MarketData")

    @classmethod
    def from_fields(cls, **fields: FloatSeries) -> MarketData:
        """Build a `MarketData` from keyword field series.

        Args:
            **fields: Field series keyed by name; all must share one index.

        Returns:
            The container.

        Raises:
            DataValidationError: If no fields are given, or the series are not
                indexed identically.
        """
        if not fields:
            message = "MarketData requires at least one field"
            raise DataValidationError(message)
        validated = {name: ensure_series(value, name) for name, value in fields.items()}
        ensure_aligned(**validated)
        return cls(pd.DataFrame(validated))

    @property
    def fields(self) -> tuple[str, ...]:
        """The field names, in column order."""
        return tuple(str(name) for name in self._frame.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        """The shared `DatetimeIndex` of every field."""
        return cast("pd.DatetimeIndex", self._frame.index)

    def __contains__(self, field: object) -> bool:
        """Return whether ``field`` is one of the container's fields."""
        return field in self._frame.columns

    def __getitem__(self, field: str) -> FloatSeries:
        """Return the named field as a copy, so a caller cannot mutate the container."""
        if field not in self._frame.columns:
            message = f"MarketData has no field {field!r}; available fields: {list(self.fields)}"
            raise DataValidationError(message)
        return cast("FloatSeries", self._frame[field].copy())

    def require(self, fields: Iterable[str]) -> None:
        """Assert that every field in ``fields`` is present.

        Args:
            fields: Field names a rule declares it needs.

        Raises:
            DataValidationError: Naming every missing field.
        """
        missing = [field for field in fields if field not in self._frame.columns]
        if missing:
            message = f"MarketData is missing required field(s): {missing}"
            raise DataValidationError(message)
