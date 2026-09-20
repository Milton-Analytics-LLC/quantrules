# quantrules — Claude Code Context

`quantrules` is a data-provider-agnostic Python library of composable building blocks for
systematic trading systems, in the style of Robert Carver's *Systematic Trading*. Inputs
and outputs are `pandas` objects. Apache-2.0, © Milton Analytics, LLC.

Full design: `docs/superpowers/specs/2026-09-18-quantrules-design.md`. Read it before
starting a new phase.

---

## Non-negotiables

These are not style preferences. A change that breaks one does not ship.

1. **No GPL-derived code.** Never read, copy, adapt or transcribe from `pysystemtrade`
   or any GPL/AGPL project. Implement from first principles or a published formula, and
   cite the formula in the docstring. The library must stay usable inside closed source.
2. **No look-ahead**, and it is *enforced*. Every public function annotated to return a
   `Series`/`DataFrame` must have a `CausalCase` registered in `tests/causality.py`, or
   the meta-test in `tests/test_causality_registry.py` fails the build. Multi-input
   functions are tested by passing all inputs as columns of one `DataFrame`, so
   truncation applies to every input together.
3. **No network access** in the library or its tests. Ever. Tests use the deterministic
   generators in `tests/support.py`.
4. **No new runtime dependencies** without asking. Runtime surface is `pandas` + `numpy`,
   plus `scipy` behind the `[options]` extra.
5. **TDD.** Write the failing test, watch it fail, then implement. New numerical code
   needs a golden test with hand-computed values (arithmetic shown in the docstring), a
   registered causality check, and a property test where a property exists.
6. **Coverage stays at 100%** (the gate is 95%; it has never been below 100).

## The data contract

Every public function obeys this. It is why composition works.

- Inputs: numeric `Series`/`DataFrame` on a **unique, monotonic `DatetimeIndex`**.
  Missing values allowed. Validated at the boundary by `quantrules._validation`.
- **Output carries the input's index**, NaN-padded through warmup — never dropped,
  never reindexed.
- Inputs are never mutated. Everything is `float64`.
- **No resampling inside any function.** Annualisation is an explicit `periods_per_year`
  (default 256 business days).

## Commands

```bash
uv sync --group dev --all-extras
uv run pytest                                      # 95% coverage gate
uv run ruff check . && uv run ruff format --check .
uv run mypy                                        # strict
uv run mkdocs build --strict
```

CI runs exactly these. Green locally means green there.

## Layout

`src/` layout, so tests run against the installed artefact.

```
src/quantrules/
  defaults.py config.py exceptions.py testing.py _typing.py _validation.py
  indicators/ rules/ forecasts/ sizing/ portfolio/ evaluate/
```

`quantrules.testing.assert_causal` is **public API**, not test-only: downstream packages
shipping proprietary rules through the `quantrules.rules` entry point use it to prove the
same property.

## Phases

1. ~~Tooling, repo, CI, licensing, `defaults`/`config`/`_validation`, causality harness.~~ **Done.**
2. ~~`indicators` (non-options) — averages, volatility, trend, channels, oscillators, normalize, volume, breadth, events.~~ **Done.** The catalog intentionally extends past Carver's own set; the framework and data contract are unchanged.
3. `rules` + `forecasts` — EWMAC, carry, breakout, mean reversion; scaling, capping, weights, FDM, combination.
4. `sizing` + `portfolio` — vol targeting, instrument vol, subsystem position, buffering, IDM, handcrafting, costs.
5. `evaluate` + end-to-end example on a bundled synthetic dataset.
6. `indicators.options` behind the `[options]` extra (includes a Black-Scholes/Black-76 IV solver).
7. Docs polish, CHANGELOG, tag `v0.1.0`, release to PyPI.

## Workflow

- **`main` is protected**: `ci-ok` is a required status check, linear history, no force
  push. `enforce_admins` is false, so an org admin retains an override. Land phases as
  **pull requests**.
- Conventional-commit titles (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Update `CHANGELOG.md` under `Unreleased` for anything user-visible.
- Link the issue with `Closes #123` when a PR fully resolves it.

## Open items

Ask before guessing on these.

- **Published tables are deliberately not encoded from memory.** Request them when the
  phase needs them, and use them as test fixtures, never as text in the docs: forecast
  scalars per EWMAC speed (phase 3); FDM/IDM tables (phases 3–4); handcrafting
  candidate-weight tables (phase 4); cost and turnover figures in SR units (phase 4).
- **Constants encoded in phase 1 that are worth confirming against the book**:
  `VOLATILITY_EWMA_SPAN` (36), `CARRY_SMOOTHING_SPAN` (90), `BUFFER_FRACTION` (0.10),
  `MAX_DIVERSIFICATION_MULTIPLIER` (2.5).
- **Two email addresses were invented and may not exist**: `conduct@withmilton.ai`
  (CODE_OF_CONDUCT.md) and `security@withmilton.ai` (SECURITY.md).
- **Deferred to phase 7**: README badges, and a `CODECOV_TOKEN` repo secret. The coverage
  upload step is `continue-on-error`, so its absence cannot redden CI.
- **Python 3.10 reaches EOL in October 2026.** It currently pins the floor to pandas
  2.3.x; dropping it would let the floor rise. The matrix deliberately spans both pandas
  majors today.
