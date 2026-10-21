# Strategies Notes

When combining 2 indicators we keep the relative position of the indicators to each other in consideration

How one interacts with other technical indicators using pure ratios or percentage spreads, ensuring absolute prices are never used.
    - For Example: measure the gap between VWAP & Moving Average Spread on same timeframe
        ```VWAP_EMA9_Spread = (VWAP - EMA9) / EMA9```

    - Why it helps:
        - When VWAP is significantly above the 9 EMA (+ spread), short-term momentum and volume are stretching apart, often signaling strong trend acceleration or an overextended move.
        - When VWAP converges tightly with the 9 EMA (~0.0 spread), it signals compression, equilibrium, or a potential consolidation breakout.

By keeping these relationships formatted as percentage ratios, they plug straight into your FEATURE_COLUMNS schema alongside your other normalized indicators without breaking multi-stock scaling.