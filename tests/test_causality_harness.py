# SPDX-FileCopyrightText: 2026 Milton Analytics, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the causality harness itself.

The harness is what proves the rest of the library has no look-ahead, so it
needs its own proof: it must accept functions that are genuinely causal and
reject every way a function can peek at the future.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantrules.testing import assert_causal, default_checkpoints
from tests.support import random_walk, series


@pytest.fixture
def prices() -> pd.Series:
    return random_walk(120, seed=7)


class TestAcceptsCausalFunctions:
    def test_expanding_mean(self, prices: pd.Series) -> None:
        assert_causal(lambda s: s.expanding().mean(), prices)

    def test_exponentially_weighted_mean(self, prices: pd.Series) -> None:
        assert_causal(lambda s: s.ewm(span=16).mean(), prices)

    def test_trailing_rolling_mean(self, prices: pd.Series) -> None:
        assert_causal(lambda s: s.rolling(10).mean(), prices)

    def test_a_lagged_series(self, prices: pd.Series) -> None:
        """Looking backwards is always safe."""
        assert_causal(lambda s: s.shift(1), prices)

    def test_warmup_nans_compare_equal(self, prices: pd.Series) -> None:
        """A long warmup produces NaN on both sides; NaN must match NaN."""
        assert_causal(lambda s: s.rolling(100).mean(), prices)

    def test_a_function_returning_a_frame(self, prices: pd.Series) -> None:
        def channels(s: pd.Series) -> pd.DataFrame:
            return pd.DataFrame({"high": s.rolling(10).max(), "low": s.rolling(10).min()})

        assert_causal(channels, prices)

    def test_a_function_taking_a_frame(self, prices: pd.Series) -> None:
        """Multi-input rules are tested by passing every input in one frame."""
        frame = pd.DataFrame({"price": prices, "volatility": prices.diff().ewm(span=36).std()})

        assert_causal(lambda f: f["price"].ewm(span=8).mean() / f["volatility"], frame)


class TestRejectsLookAhead:
    def test_a_whole_series_statistic(self, prices: pd.Series) -> None:
        """Using the full-sample mean is the classic accidental look-ahead."""
        with pytest.raises(AssertionError, match="look-ahead"):
            assert_causal(lambda s: pd.Series(s.mean(), index=s.index), prices)

    def test_a_centred_rolling_window(self, prices: pd.Series) -> None:
        with pytest.raises(AssertionError, match="look-ahead"):
            assert_causal(lambda s: s.rolling(11, center=True).mean(), prices)

    def test_an_explicit_negative_shift(self, prices: pd.Series) -> None:
        with pytest.raises(AssertionError, match="look-ahead"):
            assert_causal(lambda s: s.shift(-1), prices)

    def test_a_backward_filled_series(self, prices: pd.Series) -> None:
        """Back-filling pulls a future observation into the present."""
        gapped = prices.copy()
        gapped.iloc[::5] = float("nan")

        with pytest.raises(AssertionError, match="look-ahead"):
            assert_causal(lambda s: s.bfill(), gapped)

    def test_a_reversed_cumulative_sum(self, prices: pd.Series) -> None:
        with pytest.raises(AssertionError, match="look-ahead"):
            assert_causal(lambda s: s[::-1].cumsum()[::-1], prices)

    def test_leakage_only_in_a_frame_column(self, prices: pd.Series) -> None:
        def leaky(s: pd.Series) -> pd.DataFrame:
            return pd.DataFrame({"safe": s.rolling(5).mean(), "leaky": s.shift(-1)})

        with pytest.raises(AssertionError, match="leaky"):
            assert_causal(leaky, prices)


class TestErrorReporting:
    def test_names_the_function_under_test(self, prices: pd.Series) -> None:
        with pytest.raises(AssertionError, match="peeking_rule"):
            assert_causal(lambda s: s.shift(-1), prices, name="peeking_rule")

    def test_reports_the_offending_timestamp(self, prices: pd.Series) -> None:
        with pytest.raises(AssertionError, match=r"\d{4}-\d{2}-\d{2}"):
            assert_causal(lambda s: s.shift(-1), prices)


class TestOutputContract:
    def test_rejects_an_output_shorter_than_its_input(self, prices: pd.Series) -> None:
        """Dropping warmup rows breaks composition, so the harness rejects it."""
        with pytest.raises(AssertionError, match="index"):
            assert_causal(lambda s: s.rolling(10).mean().dropna(), prices)

    def test_rejects_a_reindexed_output(self, prices: pd.Series) -> None:
        with pytest.raises(AssertionError, match="index"):
            assert_causal(lambda s: s.reset_index(drop=True), prices)


class TestCheckpoints:
    def test_explicit_checkpoints_are_honoured(self, prices: pd.Series) -> None:
        """A leak confined to one timestamp is caught when that point is checked."""
        leak_at = 100

        def leaks_once(s: pd.Series) -> pd.Series:
            out = s.rolling(5).mean()
            if len(s) > leak_at + 1:
                out.iloc[leak_at] = s.iloc[leak_at + 1]
            return out

        assert_causal(leaks_once, prices, checkpoints=[50, 60])

        with pytest.raises(AssertionError, match="look-ahead"):
            assert_causal(leaks_once, prices, checkpoints=[leak_at])

    def test_rejects_a_checkpoint_outside_the_series(self, prices: pd.Series) -> None:
        with pytest.raises(ValueError, match="checkpoint"):
            assert_causal(lambda s: s, prices, checkpoints=[len(prices)])


class TestDefaultCheckpoints:
    def test_covers_the_back_of_the_series(self) -> None:
        checkpoints = default_checkpoints(120)

        assert checkpoints == sorted(set(checkpoints))
        assert min(checkpoints) >= 120 // 3
        assert max(checkpoints) == 119

    def test_never_exceeds_the_requested_count(self) -> None:
        assert len(default_checkpoints(500, count=5)) <= 5

    @pytest.mark.parametrize("periods", [2, 3, 6, 40])
    def test_short_series_yield_valid_in_range_checkpoints(self, periods: int) -> None:
        """Short histories check more of the series, never fewer valid positions."""
        checkpoints = default_checkpoints(periods)

        assert checkpoints, "a checkable series must yield at least one checkpoint"
        assert all(0 <= position < periods for position in checkpoints)
        assert checkpoints[-1] == periods - 1, "the last observation is always checked"

    @pytest.mark.parametrize("periods", [0, 1])
    def test_rejects_a_series_too_short_to_prove_anything(self, periods: int) -> None:
        with pytest.raises(ValueError, match="at least 2 observations"):
            default_checkpoints(periods)


class TestEmptyInput:
    def test_rejects_an_empty_series(self) -> None:
        empty = series([])

        with pytest.raises(ValueError, match="empty"):
            assert_causal(lambda s: s, empty)
