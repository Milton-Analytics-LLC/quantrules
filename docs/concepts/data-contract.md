# The data contract

Every public function in `quantrules` obeys the same contract. Once you know it, you
know how to compose any two functions in the library, and how to write your own that
slot in alongside them.

## What a function expects

An input series must be:

- a `pandas.Series` (or a `DataFrame` of them),
- of **numeric dtype**,
- indexed by a **`DatetimeIndex`** that is **monotonically increasing** and contains
  **no duplicates**.

Missing values are allowed. Real price histories have holidays, halts and gaps, and a
library that refused `NaN` would be useless on real data.

Violations raise
[`DataValidationError`][quantrules.exceptions.DataValidationError], naming the offending
parameter, at the boundary - not several layers deeper as a silent `NaN`.

## What a function promises

- **The output carries the same index as the input.** Always. A function that needs 36
  observations of warmup returns `NaN` for the first 35 rows rather than dropping them.
  This is what makes composition safe: you never have to re-align anything.
- **The input is not mutated.** Functions return new objects, or the caller's object
  unchanged when no conversion was needed.
- **Values are `float64`.** Integer inputs are converted on the way in.

!!! note "Why warmup is padded, not dropped"

    If a 36-day volatility estimate returned a series 35 rows shorter than the price, and
    a 256-day estimate returned one 255 rows shorter, combining them would require the
    caller to work out the alignment. Padding with `NaN` makes every intermediate series
    the same length, so arithmetic between them simply works, and the `NaN`s propagate to
    exactly the rows where the answer genuinely is unknown.

## Frequency and annualisation

`quantrules` never resamples your data. A function given daily bars returns daily
results; a function given weekly bars returns weekly results.

Where a calculation has to convert between periodic and annual terms - a volatility
target, a Sharpe ratio - it takes an explicit `periods_per_year` argument. The default is
[`BUSINESS_DAYS_PER_YEAR`][quantrules.defaults.BUSINESS_DAYS_PER_YEAR], which is 256
rather than 252 because its square root is exactly 16.

If you pass weekly data and leave `periods_per_year` at its default, your results will
be wrong by a factor of about 2.2. The library cannot detect this for you, because
inferring frequency from an index is unreliable on real calendars.

## Multiple inputs

Functions taking several series - a price and a volatility, say - require them to share
an index exactly. They do not align, reindex, or forward-fill on your behalf, because
silently aligning mismatched data is a good way to introduce look-ahead.

Align your inputs once, deliberately, before the pipeline.

## Configuration

Parameters travel in frozen dataclasses from [`quantrules.config`][quantrules.config].
They validate on construction, are hashable, and convert to and from plain mappings so a
system can be described in YAML:

```python
from quantrules.config import SystemConfig

config = SystemConfig.from_mapping(
    {
        "volatility_target": {"annual_vol_target": 0.25, "capital": 1_000_000.0},
        "buffering": {"fraction": 0.1},
    }
)
```
