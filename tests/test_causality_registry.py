# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""The meta-test that keeps the causality guarantee honest.

Every public time-series function quantrules exports must be registered in
`tests.causality.CAUSAL_CASES`. Adding a function without registering it fails
the build, so the no-look-ahead guarantee cannot quietly stop being true as the
library grows.
"""

from __future__ import annotations

import pandas as pd
import pytest

import quantrules
import tests.samplepkg
from quantrules.testing import assert_causal
from tests.causality import CAUSAL_CASES, CausalCase, discover_series_functions, returns_time_series


class TestReturnsTimeSeries:
    def test_detects_a_series_return(self) -> None:
        def indicator(values: pd.Series) -> pd.Series:
            return values

        assert returns_time_series(indicator)

    def test_detects_a_frame_return(self) -> None:
        def channels(values: pd.Series) -> pd.DataFrame:
            return pd.DataFrame({"value": values})

        assert returns_time_series(channels)

    def test_ignores_a_scalar_return(self) -> None:
        def sharpe(values: pd.Series) -> float:
            return float(values.mean())

        assert not returns_time_series(sharpe)

    def test_ignores_a_none_return(self) -> None:
        def check(values: pd.Series) -> None:
            del values

        assert not returns_time_series(check)

    def test_ignores_a_non_callable(self) -> None:
        assert not returns_time_series(3.0)


class TestDiscovery:
    """Proven against a fixture package, so the walker is not merely vacuous."""

    def test_finds_series_and_frame_functions(self) -> None:
        found = discover_series_functions(tests.samplepkg)

        assert "samplepkg.exported_series_function" in found
        assert "samplepkg.frames.channel" in found

    def test_recurses_into_subpackages(self) -> None:
        found = discover_series_functions(tests.samplepkg)

        assert "samplepkg.nested.nested_series_function" in found

    def test_ignores_functions_returning_scalars(self) -> None:
        found = discover_series_functions(tests.samplepkg)

        assert not any(name.endswith("exported_scalar_function") for name in found)

    def test_ignores_names_absent_from_dunder_all(self) -> None:
        found = discover_series_functions(tests.samplepkg)

        assert not any(name.endswith("unexported_series_function") for name in found)

    def test_ignores_private_modules(self) -> None:
        found = discover_series_functions(tests.samplepkg)

        assert not any("_private" in name for name in found)


class TestEveryExportedFunctionIsRegistered:
    def test_no_time_series_function_escapes_the_causality_check(self) -> None:
        exported = discover_series_functions(quantrules)
        registered = {case.name for case in CAUSAL_CASES}

        unregistered = exported - registered

        assert not unregistered, (
            "These public time-series functions are not covered by a causality "
            f"check: {sorted(unregistered)}. Add a CausalCase for each in "
            "tests/causality.py, or the no-look-ahead guarantee is unproven."
        )

    def test_registered_cases_name_functions_that_still_exist(self) -> None:
        """A renamed or deleted function must not leave a stale registration behind."""
        exported = discover_series_functions(quantrules)
        stale = {case.name for case in CAUSAL_CASES} - exported

        assert not stale, f"Stale causality registrations: {sorted(stale)}"

    def test_case_names_are_unique(self) -> None:
        names = [case.name for case in CAUSAL_CASES]

        assert len(names) == len(set(names))


@pytest.mark.parametrize("case", CAUSAL_CASES, ids=lambda case: case.name)
def test_registered_function_has_no_look_ahead(case: CausalCase) -> None:
    assert_causal(case.function, case.build_data(), name=case.name)
