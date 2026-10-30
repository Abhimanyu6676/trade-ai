# Ensemble Prediction Engine

Implementation: `prediction_engine.py`. Four directional specialists predict
triple-barrier outcome probabilities (`-1` stop, `0` timeout, `1` target); the
PPO policy combines them into a flat, long, or short decision with a position
size and entry bracket width. It does not place trades.

## Model ownership and loading

- `models.py` owns the active model instances and loads four directional base
  checkpoints plus the PPO checkpoint at application startup.
- The GRU input width is read from `gru.weight_ih_l0` in the saved state dict, so loading does not wait for a prediction or read cached candle data.
- Every base checkpoint and a schema-compatible PPO checkpoint are required for
  inference; missing or stale artifacts make the prediction service unavailable.
- `GRUTradingBrain` is defined in `models.py`: two GRU layers, hidden size 64, and a three-class output layer.
- The manager also owns unloading, timestamped backups, in-memory instance access, and checkpoint reloads. Training orchestration requests those operations through the manager.

## Inference flow

1. The route calls `predict_with_history()`. It determines the target date from the latest supplied candle, or today when candles are omitted.
2. It requests at least 30 prior trading sessions from `dataStore`, which uses the exchange's previous-trading-day calendar. If no caller candles were supplied, it fetches the target day's candles from OpenAlgo.
3. Target-day candles are merged transiently with the prior history. They are not added to the in-memory history store.
4. `predict_from_dataframe()` passes the merged frame to `predict()`. `predict()` requests currently loaded model instances from `models.py`, then calls `merge_ohlcv_history()` to normalize known OHLCV names, parse timestamps, stable-sort, and drop duplicate timestamps in favor of the later row.
5. The raw chronological candles are independently passed through feature
   calculation with `model_type="lgb"` and `model_type="gru"`. ATR defaults to
   relative-to-close values in feature construction; PPO consumes that relative
   value directly as volatility context.
6. Long and short LightGBM specialists predict from the latest valid feature
   row. Their GRU counterparts receive the last `sequence_length` valid rows
   from the newest calendar date only.
7. The four specialist probability vectors, relative ATR, volatility ratio,
   and optional portfolio state become the PPO observation. Its 19-action
   contract selects flat or a directional size and one of the 0.75, 1.0, or
   1.25 bracket multipliers. At entry, the multiplier scales both the baseline
   target and stop, preserving the 2:1 ratio; it remains fixed while the
   position is open. The API returns the selected-side take-profit probability
   as confidence plus specialist probabilities and bracket distances.

## Current limitations

- GRU training and app inference use the shared `DEFAULT_SEQUENCE_LENGTH` of 30 candles, defined in `training/gru_sequence_builder.py`.
- Session grouping uses calendar date. It does not enforce exchange hours or exact one-minute timestamp continuity.
- The history store requires every expected exchange trading date and logs observed intraday gaps over one minute. It does not synthesize candles for missing intervals because those intervals may have had no trades.
- Confidence is a specialist probability, not a calibrated guarantee.
- If the newest session lacks enough valid same-date GRU rows, inference returns
  an insufficient-data error because PPO requires all four specialist outputs.
- `prediction_engine.py` does not change model state. `train_master.py` coordinates backup, unload, training, and reload through `models.py`.


## Pro-Tip for Maximum Speed (Optional) TODO - do not remove this until its applied

    If you ever notice latency creeping up as your historical dataset grows larger, the best optimization is to compute your rolling indicators incrementally or cache the historical feature dataframe so you only append the newest live candle rather than recalculating indicators from scratch every single minute. But right now, your architecture is more than fast enough for intraday execution!
