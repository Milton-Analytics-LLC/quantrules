# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, the public API may change in a minor release.

## [Unreleased]

### Added

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
