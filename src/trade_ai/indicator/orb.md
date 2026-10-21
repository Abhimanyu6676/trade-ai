# ORB indicator

An Opening Range Breakout (ORB Indicator by Taran13) calculates a session's high and low boundaries during a set opening window (commonly the first 5, 15, or 30 minutes), freezes those levels for the day, and projects target and stop-loss levels based on the range size.

## Core Calculation Steps

- Session Start (t₀): Marks the official market open (e.g., 09:15 AM IST for NSE or 09:30 AM EST for US markets).
- Time Window (m): Defines the duration of the opening range, typically 5, 15, 30, or 60 minutes.
- Opening Range High (ORH): The maximum high price (H) recorded across all bars formed inside the opening window:
```
ORH = max(H)
```
- Opening Range Low (ORL): The minimum low price (L) recorded across all bars formed inside the opening window:
```
ORL = min(L)
```
- Opening Range Midpoint (ORM): The exact center of the range:
```
ORM = (ORH + ORL)/2
```
- Range Size (Range): The total distance of the opening volatility:
```
Range = ORH - ORL
```

## Important Notes when applying ORB indicator

Given how your pipeline is built—using percentage distances for your EMAs ((Close - EMA) / EMA), percentage-based ATR, and volatility—introducing absolute price levels for an Open Range Breakout (ORB) would break your multi-stock training architecture.   

Here is why relative ORB is essential for your setup, along with how you should structure it:

- ### Why Absolute ORB Fails in Multi-Stock Models

  If you pass absolute price values (e.g., an ORB High of ₹2,450.50 for Reliance and ₹540.20 for SBIN), the neural network's weights will become hardcoded to specific share prices. A model trained on high-priced stocks will misinterpret low-priced stocks, and vice versa.

- ### How to Format ORB to Fit Your Relative Pipeline
  
  o keep your features scale-independent and clean, you should translate the Opening Range into percentages or volatility multiples. Here are the three best ways to compute ORB for your feature engine:

  1. Distance to ORB High/Low (Percentage): 
    Measure how far the current price is from the session's opening range boundaries as a percentage:
    - Why: This tells the model precisely how close price is to breaking out (e.g., +0.002 means price is 0.2% away from breaking the ORB high), regardless of whether the stock trades at ₹100 or ₹5,000.
  
  2. ORB Range Height (Normalized by ATR or Price):
   Measure the total size of the opening range (High minus Low):
   - Why: This normalizes volatility. A wide opening range means high initial volatility; a tight range indicates compression and a potential explosive breakout.
  
  3. Breakout State Flags:
   You can also include binary or continuous state indicators that fit cleanly into your feature schema:
   - ORB_Above: 1.0 if price is currently above the ORB high, else 0.0.
   - ORB_Below: 1.0 if price is currently below the ORB low, else 0.0.

By keeping your ORB features expressed as relative percentages or ATR multiples, you maintain complete scale-independence, allowing your GRU model to learn universal breakout patterns across any stock in your watchlist

---

## Breakout and Price Target Projections

Once the opening window closes, ORH and ORL lock in for the session and project potential expansion levels using a multiplier (k, commonly 1x or 2x):

| Signal Type | Breakout Trigger | Target 1 (100% Range) | Target 2 (200% Range) |
| --- | --- | --- | --- |
| Bullish / Buy | Price closes or trades above ORH	| T1 = ORH + (1 × Range) | T2 = ORH + (2 × Range) |
| Bearish / Sell | Price closes or trades below ORL | T1 = ORL - (1 × Range) | T2 = ORL - (2 × Range) |

## Calculation and Nomenclature Used in `trade-ai`

The feature pipeline calculates opening ranges of 15, 30, and 60 minutes from the NSE open at 09:15 Asia/Kolkata, using one-minute candles timestamped at candle open. `ORB_1h` means the first 60 minutes after 09:15; it is not the regular resampled 1-hour bar. The opening range uses timestamps from the session open up to, but not including, the window end. Its levels are available only on rows at or after the window end. The window is considered valid only when all expected one-minute opening candles are present. If they are missing, that session's ORB feature values remain zero.

For each window, the pipeline returns these five features:

| Feature suffix | Calculation | Meaning |
| --- | --- | --- |
| `Range_Pct` | `(ORH - ORL) / ORM` | Opening range width relative to its midpoint. |
| `High_Distance_Pct` | `(ORH - Close) / Close` | Positive below ORH; negative above ORH. |
| `Low_Distance_Pct` | `(ORL - Close) / Close` | Positive above ORL; negative below ORL. |
| `Midpoint_Distance_Pct` | `(Close - ORM) / Close` | Positive above ORM; negative below ORM. |
| `Breakout_Direction` | `1` if `Close > ORH`; `-1` if `Close < ORL`; otherwise `0` | Close-based breakout state; a wick alone does not count. |

For example, `ORB_15m_High_Distance_Pct` is the 15-minute opening-range high distance. `_Pct` values are ratios stored as decimals, not values multiplied by 100; `0.01` represents 1%. The model does not currently use the readiness flag, absolute ORH/ORL values, or the T1/T2 target projections described above. There is no 5-minute ORB feature in the active model schema.

