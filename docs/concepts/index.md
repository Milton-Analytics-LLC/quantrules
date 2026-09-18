# Concepts overview

A systematic trading system answers one question repeatedly: *given what I know now,
how much of this instrument should I hold?* `quantrules` splits that question into four
steps, and gives you each step separately.

## The pipeline

```text
prices ─► indicators ─► rules ─► raw forecast
                                      │
                                      ▼
                          scale ─► cap ─► combine ─► combined forecast
                                                           │
                                                           ▼
                                   volatility target ─► position ─► buffer
```

Each arrow is a pure function on `pandas` objects. Nothing is hidden in an object graph,
so you can stop at any stage, inspect it, plot it, or replace it with your own.

## 1. Indicators

Descriptive statistics of a price series: a moving average, a volatility estimate, a
channel, an oscillator. They make no claim about what to do. They are the raw material
for everything above them, and the volatility estimate in particular is used twice -
once to normalise the forecast, and again to size the position.

## 2. Rules and raw forecasts

A trading rule turns indicators into a **raw forecast**: a number whose *sign* says
long or short and whose *magnitude* says how strongly. Raw forecasts are risk-adjusted -
divided by the instrument's own volatility - so the same rule gives comparable numbers
on a quiet government bond and a violent commodity.

Raw forecasts are not yet comparable *between rules*, because a fast crossover and a
carry signal naturally have different scales.

## 3. Forecast processing

This is where forecasts become comparable and combinable:

- **Scaling** multiplies a raw forecast by a constant chosen so its long-run average
  absolute value is 10. Two rules scaled this way can be averaged meaningfully.
- **Capping** clips the scaled forecast at ±20, so one extreme reading cannot dominate.
- **Combining** takes a weighted average of several scaled forecasts, then multiplies by
  a **forecast diversification multiplier** - because averaging imperfectly correlated
  signals shrinks their combined variance, and the multiplier restores the intended
  average size.

The output is one combined forecast per instrument, still on the ±20 scale.

## 4. Sizing and portfolio

The combined forecast says how strongly to hold. Turning that into a quantity needs:

- **A volatility target.** How much risk the whole system may take, expressed as an
  annualised standard deviation of returns and converted into a cash amount per period.
- **Instrument value volatility.** What one unit of the instrument does to your account
  in cash terms, combining price volatility, contract size, and any currency conversion.
- **Instrument weights and a diversification multiplier**, for the same reason forecasts
  need them.
- **Buffering.** A position is left alone while it sits inside a no-trade band around the
  optimal position. This gives up a little accuracy and buys a large reduction in
  turnover, which is what you actually pay for.

## Two rules that hold everywhere

Two guarantees run through every layer, and everything else depends on them:

1. **[The data contract](data-contract.md)** - what a function may assume about its
   input, and what it promises about its output.
2. **[No look-ahead](causality.md)** - a value computed for time *t* uses only
   information available at *t*, proven by tests rather than asserted in a docstring.
