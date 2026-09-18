## What this changes

<!-- One or two sentences. What is different after this PR that was not before? -->

## Why

<!-- The problem being solved. Link the issue: "Closes #123" if it fully resolves it,
     "Refs #123" if it only moves it along. -->

## Checklist

- [ ] Tests were written before the code, and I watched them fail first
- [ ] `uv run pytest` passes, with coverage at or above 95%
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy` passes
- [ ] New time-series functions are registered in `tests/causality.py`
- [ ] New numerical code has a golden test showing the arithmetic in its docstring
- [ ] `CHANGELOG.md` updated under `Unreleased`, if this is user-visible
- [ ] No code was copied or adapted from `pysystemtrade` or any GPL/AGPL project
- [ ] No new runtime dependency, or one discussed in an issue first
