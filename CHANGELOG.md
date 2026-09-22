# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, the public API may change in a minor release.

## [Unreleased]

### Added

- `quantrules.forecasts` - the forecast-processing layer. Every function carries the input
  index, pads warmup with `NaN`, and is covered by a registered no-look-ahead check:
    - `scaling`: `forecast_scalar` and `scale`, which rescale a raw forecast to a target
      average absolute forecast either from an explicit scalar or by causal estimation.
    - `capping`: `cap`, which clips a scaled forecast to a symmetric bound (default +/-20).
    - `diversification`: `forecast_diversification_multiplier`, the causal
      `1 / sqrt(wᵀ C w)` multiplier, floored and capped, that restores a combined
      forecast's average absolute value.
    - `combine`: `combine`, a weighted average of forecasts scaled by the diversification
      multiplier and capped.
- `quantrules.correlation.rolling_correlation` - a standalone, causal Pearson correlation
  over an expanding or trailing window, shared by the diversification multiplier.
- `quantrules._validation.ensure_frame` - validates a `DataFrame` of named series against
  the data contract (now shared by `MarketData` and the forecast-combination functions).
- API-reference pages for `quantrules.correlation` and every forecast module.
- `quantrules.rules` - the trading-rule layer. Container-free rule maths, each built on the
  phase-2 indicators, scaled and capped later by `quantrules.forecasts`, and covered by a
  registered no-look-ahead check:
    - `ewmac`: a volatility-normalised fast-minus-slow EWMA crossover.
    - `carry`: a volatility-normalised, EWMA-smoothed carry measure.
    - `breakout`: price's position within its rolling high-low channel, optionally smoothed.
    - `mean_reversion`: the negated rolling z-score of the price.
- `quantrules.data.MarketData` - an immutable container bundling an instrument's named
  fields on one validated `DatetimeIndex`, the uniform input a registered rule consumes.
- A rule registry: `Rule`, the `register_rule` decorator, `get_rule` and `available_rules`,
  with lazy `importlib.metadata` discovery on the `quantrules.rules` entry-point group so a
  separate distribution can ship rules that are found without changes here. A third-party
  name colliding with a built-in raises rather than shadowing it. The six standard EWMAC
  speeds and carry are registered as built-ins.
- API-reference pages for `quantrules.data` and every rule module.
- `quantrules.indicators` - the non-options indicator layer. Every function takes and
  returns `pandas` objects on the shared `DatetimeIndex`, pads warmup with `NaN`,
  annualises only through an explicit `periods_per_year`, and is covered by a registered
  no-look-ahead check:
    - `averages`: `sma`, `ewma` (recursive, `adjust=False`).
    - `volatility`: `rolling_std`, `ewma_volatility` (RiskMetrics), `realized_volatility`,
      `true_range`, `atr`, `atr_percent` (Wilder or simple).
    - `trend`: `macd`, `ewma_crossover`, `rolling_slope`.
    - `channels`: `rolling_max`, `rolling_min`, `donchian_channels`, `bollinger_bands`,
      `keltner_channels`, `keltner_position`.
    - `oscillators`: `rsi` (Wilder or simple).
    - `normalize`: `zscore`, `clip`, `rolling_rank`, `rolling_percentile`.
    - `volume`: `obv`, `accumulation_distribution`, `chaikin_money_flow`.
    - `breadth`: `advance_decline_line`, `mcclellan_oscillator`, `new_high_new_low_index`,
      `trin`.
    - `events`: `cross_up`, `cross_down`, `bars_since`.
- A "Using quantrules from a NumPy codebase" guide and API-reference pages for every
  indicator module.
- Project scaffolding: packaging with `hatchling`, dependency management with `uv`,
  linting and formatting with `ruff`, strict type checking with `mypy`, documentation
  with `mkdocs-material` and `mkdocstrings`.
- `quantrules.defaults` - every default constant in one place, documented with its
  provenance.
- `quantrules.config` - frozen, self-validating configuration dataclasses
  (`VolatilityTargetConfig`, `VolatilityEstimationConfig`, `ForecastScalingConfig`,
  `BufferingConfig`, `SystemConfig`), convertible to and from plain mappings.
- `quantrules.exceptions` - `QuantRulesError` and its `ConfigurationError` and
  `DataValidationError` subclasses.
- `quantrules.testing.assert_causal` - proves a time-series function uses no information
  from the future, for this library and for anyone extending it.
- A causality registry meta-test: a public time-series function that is not covered by a
  causality check fails the build.
- CI across CPython 3.10 to 3.13 on Linux, macOS and Windows, exercising both pandas 2.x
  and pandas 3.x.

[Unreleased]: https://github.com/Milton-Analytics-LLC/quantrules/commits/main
