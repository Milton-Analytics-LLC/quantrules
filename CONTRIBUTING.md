# Contributing to quantrules

Thanks for your interest. This document covers how to get set up, what the project's
non-negotiables are, and what a good pull request looks like.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

`quantrules` uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Milton-Analytics-LLC/quantrules
cd quantrules
uv sync --group dev --all-extras
uv run pre-commit install
```

Then:

```bash
uv run pytest                    # tests, with the coverage gate
uv run ruff check . && uv run ruff format --check .
uv run mypy                      # strict
uv run mkdocs serve              # docs at http://127.0.0.1:8000
```

CI runs exactly these. If they pass locally they will pass there.

## Non-negotiables

These are the constraints that make the library what it is. A change that breaks one
will not be merged, however good it is otherwise.

### 1. No GPL-derived code

`quantrules` is Apache-2.0 and must stay usable inside closed-source systems. **Do not
read, copy, adapt, or transcribe code from `pysystemtrade` or any other GPL or AGPL
project.** Implement from first principles or from a published formula, and cite the
formula's source in the docstring.

If you are unsure whether something you have seen counts, say so in the pull request
rather than guessing.

### 2. No look-ahead

Every time-series function must compute the value at time *t* from information available
at or before *t*. This is enforced, not trusted: a meta-test finds every public function
annotated to return a `Series` or `DataFrame` and fails if it is not covered by a
causality check.

Adding an indicator or rule therefore means adding a `CausalCase` in
`tests/causality.py`. See [the causality docs](docs/concepts/causality.md).

### 3. No network access

Not in the library, not in the tests. `quantrules` takes data you already have. Tests
use the deterministic generators in `tests/support.py`.

### 4. No new runtime dependencies without discussion

The runtime surface is `pandas` and `numpy`, plus `scipy` behind the `options` extra.
Open an issue before adding to it. Dev and docs dependencies are less sensitive, but
still worth mentioning.

### 5. Tests come first

Write the failing test, watch it fail, then make it pass. New numerical code needs:

- a **golden test** with hand-computed expected values on a small fixture, showing the
  arithmetic in the test docstring;
- a **causality check**, registered as above;
- a **property test** where a property exists worth stating (a scaled forecast averages
  10; a capped forecast never exceeds the cap; buffering never increases turnover).

Coverage must stay at or above 95%.

## Pull requests

- Branch from `main`.
- Keep the change focused. Unrelated refactoring belongs in its own PR.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for the title:
  `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Link the issue the PR addresses, with `Closes #123` when it fully resolves it.
- Add a `CHANGELOG.md` entry under `Unreleased` for anything user-visible.
- All CI checks must pass.

## Style

Ruff decides formatting and lint; do not argue with it, run it. Beyond that:

- Public functions are typed, documented with Google-style docstrings, and listed in
  the module's `__all__`.
- Prefer a clear name over a comment. Where a comment is needed, explain *why*, not
  *what*.
- Vectorise. A Python loop over a price series is almost always avoidable and always
  slower.

## Reporting a security issue

Please do not open a public issue. See [SECURITY.md](SECURITY.md).
