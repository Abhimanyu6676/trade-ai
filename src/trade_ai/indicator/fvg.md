
Q. does buy/sell side liquidity and sweep imply on FVG

Yes, buy-side/sell-side liquidity and sweeps have a massive structural implication on Fair Value Gaps (FVGs). In Smart Money Concepts (SMC), a liquidity sweep and an FVG do not exist in isolation; they represent the strict cause-and-effect relationship behind institutional market delivery.
A liquidity sweep acts as the fuel, while the resulting FVG acts as the footprint of institutional displacement.

1. The Cause and Effect Relationship

• The Sweep (The Cause): Institutions need immense volume to enter large positions. To do this, they engineer price to run past key structural levels—Buy-Side Liquidity (BSL) above old highs or Sell-Side Liquidity (SSL) below old lows—to trigger retail stop-losses and pending breakout orders.
• The FVG (The Effect): Once that liquidity is swept ("grabbed"), smart money aggressively floods the market in the opposite direction. Because this institutional volume is so massive and sudden, it creates a severe buy/sell imbalance. The market moves too fast to fill orders symmetrically, leaving behind a Fair Value Gap.

2. How Sweeps Dictate FVG Validity & Probability

A random FVG formed in the middle of a trading range has very low probability and is often ignored by the market algorithm. However, when an FVG forms immediately after a liquidity sweep, its probability skyrockets.

Market Event	Rationale & Structural Implication on FVG
Sell-Side Liquidity (SSL) Sweep	Price spikes below an old swing low to trigger long stop-losses. It then aggressively reverses upward, leaving a Bullish FVG. This FVG is highly valid because it represents institutional buying after purging shorts.
Buy-Side Liquidity (BSL) Sweep	Price spikes above an old swing high to trigger short stop-losses. It then aggressively displacements downward, leaving a Bearish FVG. This FVG acts as strong institutional supply.


3. Structural Mechanics: The FVG as a Magnet

When a sweep occurs, the FVG left behind in the displacement leg serves two purposes:
1. The Rebalancing Target: Because the FVG represents an inefficiently delivered price void, the market treats it like a pricing magnet. The algorithm is incentivised to return to this FVG to "mitigate" or rebalance the spread before continuing its true direction.
2. The Entry Matrix: Traders use the FVG formed after a sweep as their high-probability entry frame (often looking for a retest of the FVG's 50% equilibrium, known as consequent encroachment).

4. The Inverse FVG (IFVG) Implication

Liquidity sweeps also dictate what happens when an FVG fails. If price sweeps liquidity, pulls back, and then closes completely through an opposing FVG (invalidating it), that FVG flips into an Inverse Fair Value Gap (IFVG).
For example, if price sweeps BSL and then aggressively crashes right through a previously formed bullish FVG, that failed bullish FVG now implies strong bearish resistance on a retest.
Would you like me to map out a step-by-step mechanical trading checklist combining a higher-timeframe liquidity sweep with a lower-timeframe FVG entry?

## Feature implementation in this project

- A bullish FVG is a three-candle gap where `low[t] > high[t-2]`; it is detected only when candle `t-1` swept a previously confirmed sell-side liquidity level. A bearish FVG uses `high[t] < low[t-2]` after a buy-side sweep.
- Each detected gap records the third candle's body-to-14-bar-ATR ratio and whether its close breaks structure from before the sweep candle. A structure-qualified event requires at least a 1.0 ATR body and a close beyond the prior 20 bars' highs/lows. `FVG_Setup_Qualified` additionally requires formation during a London or New York killzone. The numeric thresholds are configurable in `detect_fvg_events`.
- `FVG_In_Killzone` records whether the gap formed from 07:00-10:00 London or New York local time. Naive timestamps are interpreted as `Asia/Kolkata` by default; timezone-aware timestamps retain their timezone. Raw FVG events remain available outside those windows, but `FVG_Setup_Qualified` requires the event to be in a killzone.
- Liquidity levels are confirmed swing highs/lows with two closed bars to the right. This keeps the sweep-to-FVG sequence causal and avoids using future bars.
- The FVG midpoint is consequent encroachment. `FVG_Distance_Pct` measures the signed midpoint distance relative to close, while `FVG_Size_Pct` measures zone width relative to close. `FVG_CE_Touched` and `FVG_Retest_Depth` describe whether price reached the midpoint and the deepest zone penetration so far.
- `FVG_Entry_Edge_Distance_Pct` measures distance to the playbook's preferred edge (upper edge for bullish gaps, lower edge for bearish gaps). `FVG_Stop_Distance_Pct` uses the first candle's low for bullish gaps or high for bearish gaps. `FVG_Target_Liquidity_Distance_Pct` uses the nearest remaining BSL/SSL level in the trade direction; zero means no such level is currently available, consistent with the liquidity features.
- Partial mitigation keeps the FVG active. A full wick fill without a close through the far edge retires it. A close through the far edge flips its direction and sets `FVG_Is_Inverse`; the inverse state remains until price closes back through the zone.
- The feature pipeline calculates liquidity once per timeframe and passes its sweep series into the FVG calculation, avoiding duplicate pivot/sweep work.
