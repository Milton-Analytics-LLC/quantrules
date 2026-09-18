# Using quantrules from a NumPy codebase

`quantrules` takes and returns `pandas` objects on a `DatetimeIndex`. Plenty of systems
do their numerical work in bare NumPy arrays. Bridging the two is lossless and takes one
line in each direction, because quantrules' output obeys a contract those systems already
rely on: **the result is the same length as the input, with warmup padded by `NaN`** —
never dropped, never reindexed.

## The round trip

You hold values and their timestamps as arrays. Wrap them in a `Series`, call the
indicator, and drop back to an array:

```python
import pandas as pd
from quantrules.indicators.volatility import realized_volatility

prices = pd.Series(close, index=dates)  # you already hold both
vol = realized_volatility(prices, window=21, periods_per_year=252)
vol_array = vol.to_numpy()  # same length as `close`
```

`vol_array[i]` lines up with `close[i]`; the warmup rows are `NaN`, exactly where the
estimate is genuinely undefined. There is nothing to realign.

For a multi-input indicator, build one aligned frame and pass its columns:

```python
from quantrules.indicators.volatility import atr

bars = pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)
atr14 = atr(bars["high"], bars["low"], bars["close"], window=14).to_numpy()
```

## Choose the annualisation convention explicitly

quantrules never guesses your sampling frequency. Anything that annualises takes an
explicit `periods_per_year`, defaulting to
[`BUSINESS_DAYS_PER_YEAR`][quantrules.defaults.BUSINESS_DAYS_PER_YEAR] (256, whose square
root is exactly 16). A codebase that annualises with 252 trading days simply passes it:

```python
realized_volatility(prices, window=21, periods_per_year=252)
```

Pass `periods_per_year=1` to keep an estimate per-period.

## Conventions worth matching

When cross-checking against an existing implementation, line up the conventions or the
numbers will disagree for good reasons:

- **Standard deviation degrees of freedom.** `rolling_std` and `zscore` default to
  `ddof=0` (population); `realized_volatility` uses `ddof=1` (sample). Set `ddof` to match
  the other side.
- **EWMA seeding.** [`ewma`][quantrules.indicators.averages.ewma] is the recursive form
  seeded from the first observation (pandas `adjust=False`), which is Carver's convention.
  An EWMA seeded from a simple moving average of the first *n* points will differ during
  and after warmup, so compare EWMA-based measures (`ewma`, `macd`, Keltner) structurally
  rather than value-for-value.
- **Wilder versus simple** for `atr` and `rsi`: pass `method="wilder"` or
  `method="simple"` to match the other implementation.

## If you only have values, not timestamps

These indicators are indifferent to the actual index values — only annualisation cares
about frequency, and that is handled by `periods_per_year`. When you have no real
calendar, a synthetic monotonic index is a fine carrier and keeps you inside the data
contract:

```python
import pandas as pd
from quantrules.indicators.averages import sma

index = pd.date_range("2000-01-01", periods=len(values), freq="B")
result = sma(pd.Series(values, index=index), window=10).to_numpy()
```
