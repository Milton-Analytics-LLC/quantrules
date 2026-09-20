# quantrules - design

Date: 2026-09-18
Status: approved, phase 2 implemented

## Purpose

A data-provider-agnostic, composable Python library for building systematic trading
systems in the style of Robert Carver's *Systematic Trading*. Inputs and outputs are
`pandas` objects. Four layers, each usable independently: indicators, trading rules,
forecast processing, and sizing/portfolio construction, plus a minimal vectorised
evaluation module.

## Non-negotiable constraints

1. **Apache-2.0**, copyright Milton Analytics, LLC (the Delaware entity; Milton Labs is
   the California foreign-entity name and is not used in licensing).
2. **No GPL-derived code.** Nothing read, copied or adapted from `pysystemtrade` or any
   GPL/AGPL project. Everything from first principles or published formulas. Carver's
   book and blog are cited as references only; no book text or tables are reproduced.
3. **No look-ahead**, proven by tests rather than asserted.
4. **No network access** in the library or its tests. No broker or exchange code.
5. **Python 3.10+**, fully typed, `mypy --strict` clean.
6. Public API explicit via `__all__`; everything else private.

## Decisions

### Configuration: frozen dataclasses, not pydantic

Chosen for dependency hygiene above all. `quantrules` is meant to be embedded in other
systems, some of which already pin a pydantic major version; forcing one on them is a
liability. The configs are small bags of floats, so pydantic's coercion and schema
machinery buys little. Frozen dataclasses are `mypy --strict` clean without a plugin and
are hashable, which matters for memoising scalar estimation. The cost - hand-written
validation - is paid once in `_validation.py`.

### The data contract

Not in the original brief, and everything depends on it:

- Inputs are numeric `Series`/`DataFrame` on a unique, monotonic `DatetimeIndex`.
  Missing values allowed.
- **Outputs carry the input's index**, NaN-padded through warmup. Never dropped, never
  reindexed. This is what makes composition safe and makes causality testing mechanical.
- No resampling inside any function. Annualisation is an explicit `periods_per_year`.
- Everything is `float64`. Inputs are never mutated.

### Causality as infrastructure

`assert_causal` tests the identity `f(x[:t])[-1] == f(x)[t]`, which cannot be fooled by
any leak mechanism. A meta-test walks the package for public functions annotated to
return a `Series`/`DataFrame` and fails if any is unregistered, so the guarantee cannot
lapse as the library grows. Shipped as `quantrules.testing` rather than kept in `tests/`
so that downstream packages shipping proprietary rules can prove the same property.

### `src/` layout

Guarantees tests and coverage run against the installed artefact, catching packaging
bugs before release rather than after.

### Rule registration

Math functions stay container-free. The registered `Rule` is a thin adapter over a
`MarketData` container declaring `required_fields`. Registration by decorator, plus
lazy `importlib.metadata` entry-point discovery on group `quantrules.rules`, so a
private downstream package can ship proprietary rules with no change here. Lazy so that
`import quantrules` stays fast and a broken third-party plugin cannot break the import.
Third-party names colliding with built-ins raise rather than silently shadow.

### Shared correlation module

FDM, IDM and handcrafting all need a causal, optionally shrunk correlation matrix.
Without one module it gets written three times, three slightly different ways.

### Scope trims (deliberate)

- Mean reversion ships as one concrete, tested rule, not an abstract skeleton.
- Pooled forecast-scalar estimation deferred; per-instrument with a documented hook.
- `evaluate.backtest` stays a vectorised positions-to-returns calculator.
- `pytest-benchmark` deferred past v0.1.0.

## Module layout

```
quantrules/
  __init__.py  defaults.py  config.py  exceptions.py
  data.py            # MarketData container
  correlation.py     # shared causal correlation + shrinkage
  testing.py         # assert_causal, public
  _typing.py  _validation.py
  indicators/  averages volatility trend channels oscillators normalize volume breadth events options/
  rules/       base ewmac carry breakout mean_reversion
  forecasts/   scaling capping diversification combine
  sizing/      volatility_target instrument_vol position buffering
  portfolio/   weights handcrafting idm costs
  evaluate/    returns metrics backtest
```

## Toolchain

`uv` + `hatchling`; `ruff` (lint + format); `mypy --strict` with `pandas-stubs`;
`pytest` + `pytest-cov` (95% floor) + `hypothesis` (derandomised); `mkdocs-material` +
`mkdocstrings`; `pre-commit`.

CI is 8 test jobs - Ubuntu on 3.10-3.13, macOS and Windows on 3.10 and 3.13 - plus
lint, typecheck, docs and build jobs, gated behind one `ci-ok` status check. Python 3.10
resolves to pandas 2.3.x and 3.13 to pandas 3.0.x, so the matrix also proves the
`pandas>=2.0` floor. Releases publish to PyPI via trusted publishing on a `v*` tag, with
a tag-versus-`__version__` consistency check.

`[options]` pulls `scipy` and will ship a Black-Scholes/Black-76 implied-volatility
solver, so the options indicators work from a raw chain rather than requiring the user
to source IV elsewhere.

## Phases

1. **Tooling, repo, CI, licensing, `defaults`/`config`/`_validation`, causality
   harness.** (done)
2. `indicators` (non-options). Beyond Carver's own set (averages, volatility, trend,
   channels, oscillators, normalize) the catalog also ships `volume`, `breadth` and
   `events` modules and rolling rank/percentile. The framework and data contract are
   unchanged.
3. `rules` + `forecasts`.
4. `sizing` + `portfolio`.
5. `evaluate` + end-to-end example on a bundled synthetic dataset.
6. `indicators.options` behind the `[options]` extra.
7. Docs polish, changelog, `v0.1.0` to PyPI.

## Open items requiring the book

Exact published tables are not encoded from memory. They will be requested when their
phase arrives, and used as test fixtures rather than reproduced in documentation:

- forecast scalars per EWMAC speed (phase 3),
- forecast and instrument diversification multiplier tables (phases 3 and 4),
- handcrafting candidate-weight tables (phase 4),
- cost and turnover figures in Sharpe-ratio units (phase 4).

Defaults encoded in phase 1 that are worth confirming against the source:
`VOLATILITY_EWMA_SPAN` (36), `CARRY_SMOOTHING_SPAN` (90), `BUFFER_FRACTION` (0.10),
`MAX_DIVERSIFICATION_MULTIPLIER` (2.5).
