# quantrules

Composable, data-provider-agnostic building blocks for systematic trading systems, in
the style of Robert Carver's *Systematic Trading*.

!!! danger "Not investment advice"

    This library is for research and education only. Nothing it produces is a
    recommendation to buy or sell any security. Trading involves substantial risk of
    loss, including the total loss of capital. See the
    [full disclaimer](https://github.com/Milton-Analytics-LLC/quantrules/blob/main/DISCLAIMER.md).

!!! info "Pre-release"

    `quantrules` is being built layer by layer toward `v0.1.0`. Until then the public
    API may change without deprecation.

## What it is

Four layers, each usable on its own, all speaking `pandas`:

| Layer | What it does |
| --- | --- |
| **Indicators** | Moving averages, volatility, trend, channels, oscillators, normalisation |
| **Rules** | Turn indicators into raw forecasts: EWMAC, carry, breakout, mean reversion |
| **Forecasts** | Scale to a common size, cap, weight, and combine |
| **Sizing & portfolio** | Volatility targeting, position sizing, buffering, instrument weights, costs |

## What it is not

- **Not a data source.** `quantrules` never downloads anything. You bring the prices.
- **Not a broker integration.** There is no order routing and no exchange code.
- **Not an event-driven backtester.** `quantrules.evaluate` is a vectorised
  positions-to-returns calculator for validating a rule, nothing more.
- **Not a strategy.** It is arithmetic. The decisions remain yours.

## Install

```bash
pip install quantrules              # core: pandas and numpy only
pip install "quantrules[options]"   # adds implied-volatility indicators (pulls scipy)
```

Requires Python 3.10 or newer. Tested on CPython 3.10 through 3.13, against both
pandas 2.x and pandas 3.x.

## Where to start

- [Concepts overview](concepts/index.md) - how the four layers fit together.
- [The data contract](concepts/data-contract.md) - what every function expects and
  returns. Read this first; everything else assumes it.
- [No look-ahead](concepts/causality.md) - the guarantee, and how to verify your own
  rules against it.

## Licence

Apache-2.0. Copyright 2026 Milton Analytics, LLC.

`quantrules` was written from first principles and from publicly published formulas.
Carver's book and blog are cited as references only; no code was copied or adapted
from `pysystemtrade` or any other GPL-licensed project.
