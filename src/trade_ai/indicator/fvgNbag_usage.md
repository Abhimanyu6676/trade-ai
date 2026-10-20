
Institutional Context & Execution Playbook

Institutional algorithms do not view FVGs or BAGs as simple "chart patterns." They treat them as liquidity-clearing mechanisms and premium/discount execution matrices.
True institutional execution relies on a strict order of operations: HTF (Higher Timeframe) Context → Liquidity Purge → LTF (Lower Timeframe) Footprint Structural Shift.

1. Intraday Execution Framework: FVG vs. BAG

Institutional traders deploy these two delivery footprints under entirely different volatility regimes.
[Normal Delivery] ---> Sweep ---> Displaced FVG ---> Deep Premium/Discount Retest ---> Entry
[High-Vol Delivery] --> Sweep ---> Opposing Overlap (BAG) ---> Instant Boundary Rejection --> Entry

2. The FVG (Fair Value Gap) Playbook

Institutions trade the FVG as an Inefficient Price Void. The algorithm expects a measured return to the area of unfair delivery to mitigate remaining exposure.

Buy Side (Long) Institutional Setup

• The Catalyst: Price sweeps Sell-Side Liquidity (SSL) (e.g., Asian Session Low or Previous Day Low) on the 15M/5M chart.
• The Footprint: An aggressive displacement candle closes above a swing high, leaving a clear 3-candle Bullish FVG inside a New York or London Killzone.
• Algorithm Behavior: The algorithm pulls back into the discount array to re-balance the one-sided buy delivery.
• Institutional Execution:
	• Entry: Buy limit orders stacked exactly at Consequent Encroachment (50%) or the Premium Entry (Top Edge) of the FVG.
	• Stop Placement: Deployed strictly below the low of Candle 1 (the anchor candle of the FVG).
	• Take Profit: Target Unmitigated Buy-Side Liquidity (BSL) at session highs.

Sell Side (Short) Institutional Setup

• The Catalyst: Price sweeps Buy-Side Liquidity (BSL) (e.g., London Session High) on the 15M/5M chart.
• The Footprint: A massive downward displacement candle breaks structure, creating a clear Bearish FVG.
• Algorithm Behavior: The algorithm engineered a relief rally back up into premium pricing to fill short order books.
• Institutional Execution:
	• Entry: Sell limit orders placed between the Bottom Edge and Consequent Encroachment (50%) of the Bearish FVG.
	• Stop Placement: Hard stop strictly above the high of Candle 1.
	• Take Profit: Target clean pools of Sell-Side Liquidity (SSL) or an unmitigated HTF Order Block below.

3. The BAG (Balanced Price Gap / BPR) Playbook

Institutions trade the BAG as an Efficient Institutional Roadblock. Because the zone has already seen immediate two-sided delivery, the algorithm treats it as an absolute barrier.

Buy Side (Long) Institutional Setup

• The Catalyst: Price prints a Bearish FVG down into an HTF demand zone, trapping late breakout shorts.
• The Footprint: Instantly, a massive bullish candle engulfs and closes entirely above that Bearish FVG, creating a Bullish BAG/BPR where the two zones overlap.
• Algorithm Behavior: The market has already achieved equilibrium over these specific ticks. The algorithm protects this price level aggressively.
• Institutional Execution:
	• Entry: Market execution or tight Buy Limits placed strictly at the top edge of the BAG. Price must not draw deep into this zone.
	• Stop Placement: Tightly placed below the lowest point of the overlapping BAG zone.
	• Take Profit: Target major daily trendline liquidity or psychological round numbers.

Sell Side (Short) Institutional Setup

• The Catalyst: Price drives up via a Bullish FVG to trap retail breakout longs above key structural highs.
• The Footprint: A violent bearish displacement leg immediately breaks lower, printing a Bearish FVG directly on top of the prior Bullish FVG. The intersection forms the Bearish BAG/BPR.
• Algorithm Behavior: Total reversal of sentiment. The algorithm recognizes perfect balance has been met and will deny price access back above this ceiling.
• Institutional Execution:
	• Entry: Sell Limit stacked at the bottom edge of the BAG zone.
	• Stop Placement: Tightly placed above the highest point of the overlapping BAG zone.
	• Take Profit: Target internal range liquidity or structural lows.

4. Institutional Direct Comparison Matrix


Algorithmic Metric	The FVG Execution Profile	The BAG (BPR) Execution Profile
Order Book Intent	Rebalancing a one-sided order book vacuum.	Defending a heavily repriced, fully efficient zone.
Retest Depth	Expected deep draw to Consequent Encroachment (50%).	Shallow, rapid validation. Must respect the outer edge.
Execution Speed	Calculated, slower limit fills during the pullback phase.	Immediate market entry or tight limit fills; high momentum.
Risk-to-Reward (R:R)	Standard, defined by wider structure invalidation.	Highly asymmetric. Stops are exceptionally tight.

## Implemented Model Features

The feature pipeline now encodes measurable parts of this playbook on 1m, 3m, 5m, 15m, 30m, and 1h bars:

- FVG candidates still require the opposing liquidity sweep and three-candle gap. `FVG_Setup_Qualified` requires a displacement body of at least 1.0 rolling 14-bar ATR, a close beyond the 20-bar structure that existed before the sweep candle, and formation inside a London or New York killzone. The component flags are also exposed separately.
- London and New York killzones are defined as 07:00-10:00 in each location's local time. Naive timestamps are treated as Asia/Kolkata; timezone-aware timestamps use their supplied timezone. Session is a feature rather than a hard filter so it remains usable for instruments that do not trade during both windows.
- FVG state includes CE touch, maximum retest depth, preferred entry-edge distance, candle-1 stop distance, and nearest directional liquidity target distance.
- BAG requires two qualified opposing FVGs within two bars. The newer FVG must close entirely beyond the previous opposing gap while the zones overlap. Its state includes edge rejection, retest depth, far-edge stop distance, reversal displacement, and nearest directional liquidity target distance.
- `FVG_HTF_Alignment` and `BAG_HTF_Alignment` compare the active 1m direction with completed 15m and 1h FVG/BAG directions. These are directional context proxies, not detected order blocks or discretionary demand/supply zones.

These fields are model inputs, not order instructions. The project still predicts direction and does not place entries, stops, or take-profit orders. Retrain the model before inference because the feature schema has changed.
