# MACD indicator

MACD uses the difference between a fast and a slow EMA, then smooths that
difference with a signal EMA. Its line, signal, and histogram are price-valued
differences before normalization.

## Relative Feature Convention

Whenever MACD outputs are represented as relative features, divide each output
by the closing price:

- MACD line: `(fast EMA - slow EMA) / close`
- MACD signal: `signal EMA / close`
- MACD histogram: `(MACD line - MACD signal) / close`, where the line and signal
  in the numerator are their price-valued forms

This keeps the outputs dimensionless and comparable across assets with
different price levels. Do not normalize these values by an EMA or by the
signal line.
