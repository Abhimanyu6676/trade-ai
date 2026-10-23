# Session-Safe GRU Sequences

Implementation: `sequence_loader.py`, used by `train_gru.py` and `../prediction_engine.py`.

The shared `DEFAULT_SEQUENCE_LENGTH` is 30 candles and is used by both GRU training and app inference.

## Training

`valid_window_starts()` finds rolling windows whose rows all share a calendar date. `SlidingWindowDataset` uses those starts and keeps only windows whose final row has a valid Triple Barrier target. Earlier feature rows on that same date can still provide context even if they do not themselves have a label. Windows cannot cross from one calendar date into another.

## Inference

`latest_session_sequence()` finds the newest date present in the feature frame and returns its last `sequence_length` rows. If that date has too few valid feature rows, it returns `None`; prediction can then use LightGBM alone if available.

## Feature-history distinction

Features are calculated on the full chronological history before sequence selection. Prior sessions therefore warm up higher-timeframe indicators such as the 1h EMA 50, but do not become GRU sequence steps. This avoids overnight sequence bridging without resetting indicator history each morning.

## Boundaries

“Session” currently means calendar date from the candle timestamp. The sequence loader does not check exchange market hours, missing minutes, or whether all rows are exactly one minute apart. Triple Barrier labels separately use a 15:15 market-close cutoff by default; candle-open time 15:14 is eligible and 15:15 is excluded.
