# No look-ahead

A backtest that peeks at the future is worse than no backtest, because it is
confidently wrong. Look-ahead is also easy to introduce by accident: a full-sample mean,
a centred window, a `bfill` over a gap, a forgotten `shift`. Each produces a plausible,
profitable, meaningless equity curve.

`quantrules` treats this as a property to be proven, not a convention to be remembered.

## The guarantee

Every public time-series function in `quantrules` computes the value at time *t* using
only observations at or before *t*.

## How it is proven

For a function `f` with no look-ahead, the answer at time *t* cannot depend on anything
after *t*. So running `f` on history truncated at *t* must agree with running it on the
full history:

```text
f(x[:t])[-1] == f(x)[t]      for every t
```

That identity is the whole test. It needs no knowledge of what `f` does internally,
which is exactly why it cannot be fooled - any leak, by any mechanism, breaks it.

[`assert_causal`][quantrules.testing.assert_causal] checks that identity at a spread of
positions:

```python
from quantrules.testing import assert_causal

assert_causal(lambda prices: prices.ewm(span=32).mean(), prices)  # passes
assert_causal(lambda prices: prices.rolling(11, center=True).mean(), prices)  # fails
```

## It cannot be skipped

A meta-test walks the package, finds every public function annotated to return a
`Series` or `DataFrame`, and compares that set against the registry of causality checks.
A new indicator or rule that is not registered fails the build.

The guarantee therefore cannot quietly lapse as the library grows: it is not possible to
add a time-series function without either proving it causal or breaking CI.

## Use it on your own rules

If you write your own rules - especially if you ship them through the
`quantrules.rules` entry point - run the same check against them:

```python
import pandas as pd
from quantrules.testing import assert_causal


def test_my_rule_does_not_peek() -> None:
    frame = pd.DataFrame({"price": prices, "volatility": volatility})

    assert_causal(lambda data: my_rule(data["price"], data["volatility"]), frame)
```

!!! warning "Pass every input together"

    Give the function all of its inputs as columns of one `DataFrame`, so that
    truncating the history truncates all of them at once. Testing a rule against a
    pre-computed, full-history volatility series would hide any look-ahead living in
    that series.
