# quantrules

Composable, data-provider-agnostic building blocks for systematic trading systems,
in the style of Robert Carver's *Systematic Trading*.

> **Not investment advice.** This library is for research and education only. Nothing it
> produces is a recommendation to buy or sell any security. Trading involves substantial
> risk of loss. See [DISCLAIMER.md](DISCLAIMER.md).

> **Status: pre-release.** `quantrules` is being built layer by layer toward `v0.1.0`.
> The quickstart below is the target API; see [CHANGELOG.md](CHANGELOG.md) for what has
> actually shipped. Until `v0.1.0` the public API may change without deprecation.

## Quickstart

```python
import pandas as pd
import quantrules as qr

price = pd.Series(...)  # any float series on a DatetimeIndex — bring your own data

# 1. Estimate volatility, then build a raw forecast from a trading rule.
vol = qr.indicators.ewma_volatility(price)
raw = qr.rules.ewmac(price, fast_span=16, slow_span=64, volatility=vol)

# 2. Scale to an average absolute forecast of 10, then cap at +/-20.
scaled = qr.forecasts.scale(raw, target=qr.defaults.TARGET_AVG_ABS_FORECAST)
capped = qr.forecasts.cap(scaled)

# 3. Turn the forecast into a position at a 25% annualised volatility target.
config = qr.VolatilityTargetConfig(annual_vol_target=0.25, capital=1_000_000.0)
position = qr.sizing.subsystem_position(capped, price, vol, config)

# 4. Reduce turnover with a buffer, then evaluate.
buffered = qr.sizing.buffer_positions(position, config=qr.BufferingConfig())
print(qr.evaluate.sharpe_ratio(qr.evaluate.strategy_returns(buffered, price)))
```

## Design principles

- **Bring your own data.** `quantrules` never downloads anything and has no broker or
  exchange code. Every function takes `pandas` objects you supply.
- **No look-ahead, enforced by tests.** Every rolling computation uses only information
  available at or before time *t*. A causality harness proves it for every exported
  time-series function; forgetting to register a new one fails the build.
- **Pure functions.** No hidden state, no global config, no mutation of your inputs.
- **Layers are independent.** Use the indicators without the rules, the forecast maths
  without the sizing, or the whole stack.
- **Small dependency surface.** `pandas` and `numpy`. `scipy` only if you install the
  `options` extra.

## Installation

```bash
pip install quantrules              # core
pip install "quantrules[options]"   # adds implied-volatility indicators (pulls scipy)
```

Requires Python 3.10+. Tested against CPython 3.10, 3.11, 3.12 and 3.13.

## The four layers

| Layer | Package | What it does |
| --- | --- | --- |
| Indicators | `quantrules.indicators` | Moving averages, volatility, trend, channels, oscillators, normalisation |
| Rules | `quantrules.rules` | EWMAC, carry, breakout, mean reversion — and your own, via a registry |
| Forecasts | `quantrules.forecasts` | Scaling, capping, weighting, diversification multiplier, combination |
| Sizing & portfolio | `quantrules.sizing`, `quantrules.portfolio` | Volatility targeting, position sizing, buffering, instrument weights, costs |

`quantrules.evaluate` adds a minimal vectorised backtest for validating rules. It is
deliberately **not** an event-driven backtester.

## Extending with your own rules

Register a rule in-process with a decorator, or ship it from a separate distribution via
the `quantrules.rules` entry-point group — `quantrules` discovers it with no changes here.

```python
@qr.rules.register_rule("my_rule")
def my_rule(data: qr.MarketData) -> pd.Series: ...
```

## Documentation

Full documentation, including a Concepts section explaining the mathematics of each
layer, lives at <https://Milton-Analytics-LLC.github.io/quantrules/>.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache-2.0. Copyright 2026 Milton Analytics, LLC. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

`quantrules` was written from first principles and from publicly published formulas.
Robert Carver's *Systematic Trading* and his blog are cited as references only; no code
was copied or adapted from `pysystemtrade` or any other GPL-licensed project.
