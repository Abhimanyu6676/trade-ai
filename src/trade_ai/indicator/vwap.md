# VWAP indicator

The volume Weighted Average Price (VWAP) is an intraday technical indicator that shows the average price a security has traded at throughout the session, weighted by volume. Unlike simple moving averages, it reveals where the most money has actually changed hands, serving as a benchmark for fair value that resets at the start of each trading day.

## What is VWAP?

VWAP measures true intraday average price. It combines price and volume into a single line plotted on a chart.
- Bullish/Bearish Bias: Price above VWAP signals buyer control (bullish bias); price below VWAP signals seller control (bearish bias).
- Session Reset: The line starts calculating from the first trade of the session and resets every morning, making it primarily an intraday tool. (Note: Anchored VWAP allows traders to pin the start date to specific news or swings).


## Key Uses of VWAP

- Trend Identification: Prices trading above VWAP signal bullish momentum (buying strength), while prices trading below VWAP signal bearish momentum (selling pressure).
- Dynamic Support and Resistance: The VWAP line often acts as a pivot or bounce zone where price action reverses.
- Execution Benchmark: Institutional traders use VWAP to fill large orders quietly over time without causing massive price spikes or slippage.

## How it is Calculated

VWAP divides the cumulative sum of price multiplied by volume by the cumulative volume for the session:
    ```VWAP = (Typical Price * volume) / volume```

1. Calculate Typical Price: For each period (e.g., 1-minute or 5-minute candle), find the average of `high`, `low`, and `close`:
    ```Typical Price = (high + low + close) / 3```
1. Multiply by volume: Multiply the Typical Price by that period's volume:
    ```(Typical Price × volume).```
2. Accumulate: Add these values up continuously from the market open.
3. Divide: Divide that running total by the total cumulative volume traded since the open.

## How Professional & Institutional Traders Use It

Institutions (like mutual funds, hedge funds, and pension managers) move massive amounts of capital that can accidentally spike prices if executed all at once. They rely on VWAP in specific ways:
- Execution Benchmarking: Algorithms slice huge institutional orders into smaller chunks throughout the day. They judge their execution quality against VWAP—buying below VWAP or selling above VWAP is considered a successful fill.
- Minimizing Market Impact: Instead of using aggressive market orders, institutional execution algos attempt to match or trade around the VWAP curve to stay invisible and reduce slippage.
- Dynamic Support and Resistance: During strong trends, institutional algorithms often treat pullbacks back to the VWAP line as fair-value accumulation zones, causing price to bounce off the line.
- Standard Deviation Bands: Pros plot standard deviation bands (± 1σ, ± 2σ, etc.) around VWAP to measure statistical extremes (overbought or oversold conditions relative to volume-weighted fair value).


## Relative Feature Convention

When VWAP distance is used as a relative feature, normalize the signed distance
by VWAP:

`(close - VWAP) / VWAP`

The implementation uses this convention when `relative=True` (the default);
`relative=False` returns the unnormalized price-unit distance. VWAP rate of
change remains a ratio:
`(VWAP - VWAP.shift(roc_period)) / VWAP.shift(roc_period)`.

## Implementation Notes

- Implement this indicator in the ratio as other indicators are in ratio

- VWAP should also be converted into a relative ratio or percentage distance

- Keeping VWAP expressed as a relative distance maintains total consistency with your relative EMA distances, ATR percentages, and your new ORB feature distances!

- Also,  Add `VWAP Rate of Change (The Slope)`
  Instead of looking at the absolute value of VWAP, you measure how fast VWAP is rising or falling over a specific lookback window (e.g., 5 candles ago):

  ```VWAP_Slope_Pct= (VWAP − VWAP.shift(5)) / VWAP```

  - Why it helps: A positive percentage tells the model that institutional volume is actively driving the average price upward (an uptrend), while a negative percentage indicates a declining volume-weighted trend.

#  Best Usage & How to Combine It with Other Features

VWAP is exceptionally powerful when paired with other features to capture two primary market behaviors: Trend Continuation and Mean Reversion.

    A. Trend Confirmation (Institutional Backing)

        - Rule: When price is consistently above VWAP and VWAP is sloping upward (VWAP_ROC_5 > 0), institutional buyers are aggressively supporting the stock.

        - Combo with EMAs: Pair VWAP_Dist with your macro trend filter (EMA50_Dist). If a stock is trading above both VWAP and the 50 EMA, your LightGBM and GRU models will recognize a high-conviction bullish session.

    B. Mean Reversion (Overextended Spreads)
        - Rule: VWAP acts like a rubber band. If VWAP_Dist stretches too far away from zero (e.g., price is 3% or 4% away from VWAP in a short window), the price often pulls back to retest the VWAP line.

        - Combo with ATR: Combine VWAP_Dist with your ATR (Volatility) feature. A large VWAP distance combined with high ATR indicates an overextended move prone to mean-reversion, whereas a large VWAP distance with low ATR indicates a steady, persistent trend breakout.
