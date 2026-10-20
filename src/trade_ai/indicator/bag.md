Q. does buy/sell side liquidity and sweep imply on BAG

Yes, the exact same mechanics apply to Balanced Price Gaps (BAGs/BPRs), but with a structural twist.
While a classic Fair Value Gap (FVG) represents a one-way imbalance, a Balanced Price Gap (BAG) represents a highly aggressive double-sided rebalancing act directly triggered by a liquidity sweep.
In the inner circle trading lexicon, a BAG is often referred to as a Balanced Price Range (BPR).

1. How a Liquidity Sweep Forms a BAG

A BAG occurs when two opposing FVGs overlap during a highly violent shift in market structure. A liquidity sweep is almost always the catalyst for this:
1. The Trapping Leg: Price aggressively pushes in one direction to chase liquidity, leaving behind FVG #1 (e.g., a bullish FVG pushing up to sweep Buy-Side Liquidity).
2. The Sweep & Trap: Price grabs the liquidity (BSL) and instantly traps those breakout traders.
3. The Displacement Leg: Smart money aggressively hammers price in the opposite direction. This violent reversal creates FVG #2 (a bearish FVG) that cuts right back through FVG #1.
4. The Overlap (The BAG): The specific price zone where FVG #1 and FVG #2 overlap is the Balanced Price Gap.

2. The Implication: Why BAGs are Stronger than FVGs

When a liquidity sweep creates a BAG, its structural implication is vastly different from a standard FVG:
• Instant Balance: Because price aggressively moved up and down through the exact same price zone within a few candles, the market algorithm considers this zone perfectly balanced.
• Zero Magnet Status: Unlike a normal FVG, which acts as a magnet drawing price deep inside to fill an imbalance, a BAG does not need to be filled. Price has already been delivered efficiently in both directions.
• Brick-Wall Support/Resistance: Because it is already balanced, a BAG acts as a massive institutional roadblock. When price pulls back to a BAG, it should reject instantly from the very edge of the zone without drawing deep into it.

3. FVG vs. BAG After a Liquidity Sweep


Feature	FVG (Fair Value Gap) after a Sweep	BAG (Balanced Price Gap) after a Sweep
Market Delivery	One-sided. Price moved aggressively in one direction only.	Two-sided. Price quickly moved up and down over the same zone.
Price Behavior	Magnet. Price tends to draw deep into the FVG (often 50%) to rebalance it.	Roadblock. Price should strictly respect the boundaries and reject rapidly.
Order Speed	Fast, but allows for a measured pullback and mitigation.	Extremely violent. Represents an immediate, aggressive institutional shift.

Would you like me to illustrate this with a text-based chart matrix showing exactly how the bullish and bearish candles overlap to create a BAG after a sweep?

## Feature implementation in this project

- A BAG/BPR is the positive-width intersection of two opposing FVG zones. Both input FVGs must meet the sweep qualification documented in `fvg.md`.
- The opposing FVGs must form within five bars on the timeframe being calculated. The newer FVG is the reversal leg, so its direction supplies `BAG_Direction`.
- `BAG_Size_Pct` is the overlap width divided by close. `BAG_Distance_Pct` measures the signed distance to the directionally relevant rejection edge: the upper edge for bullish support and the lower edge for bearish resistance. It deliberately does not measure distance to the midpoint, because BAGs are treated as rejection zones rather than fill magnets.
- `BAG_Rejection` is a one-bar signal when price touches or penetrates the directional edge and closes back outside the BAG in its expected direction.
- The latest BAG remains active until a candle closes beyond its far edge. A newer overlapping pair replaces its active zone.
- The feature pipeline detects sweep-qualified FVG events once per timeframe and shares those events with both the FVG and BAG calculations.
