# Ensemble Prediction Engine

Implementation: `prediction_engine.py`. The engine forecasts one of the Triple Barrier classes: `-1` (stop-loss), `0` (timeout), or `1` (take-profit). It does not place trades.

## Model ownership and loading

- `models.py` owns the active model instances and loads both checkpoints at application startup.
- The GRU input width is read from `gru.weight_ih_l0` in the saved state dict, so loading does not wait for a prediction or read cached candle data.
- A missing checkpoint produces a warning to train that model first. The other model can still load and serve predictions on its own.
- `GRUTradingBrain` is defined in `models.py`: two GRU layers, hidden size 64, and a three-class output layer.
- The manager also owns unloading, timestamped backups, in-memory instance access, and checkpoint reloads. Training orchestration requests those operations through the manager.

## Inference flow

1. `predict(raw_df, ticker, sequence_length)` requests currently loaded model instances from `models.py`, then calls `merge_ohlcv_history()` to normalize known OHLCV column names, parse timestamps, stable-sort, and drop duplicate timestamps in favor of the later row.
2. The raw chronological candles are independently passed through feature calculation with `model_type="lgb"` and `model_type="gru"`. The two variants scale RSI differently.
3. LightGBM predicts from the latest valid feature row. GRU receives the last `sequence_length` valid rows from the newest calendar date only.
4. If both models return class probabilities, the engine averages them equally. If only one returns probabilities, that model is used alone. If neither can return probabilities, the engine returns an error result.
5. The signal is the highest-probability class mapped back to `-1`, `0`, or `1`; confidence is that class probability.

## Current limitations

- GRU training and app inference use the shared `DEFAULT_SEQUENCE_LENGTH` of 30 candles, defined in `sequence_loader.py`.
- Session grouping uses calendar date. It does not enforce exchange hours or exact one-minute timestamp continuity.
- The 50/50 blend is fixed, and confidence values are not calibrated guarantees.
- If the GRU lacks enough valid same-date rows but LightGBM is available, the result falls back to LightGBM alone.
- `prediction_engine.py` does not change model state. `train_master.py` coordinates backup, unload, training, and reload through `models.py`.


## Pro-Tip for Maximum Speed (Optional) TODO - do not remove this until its applied

    If you ever notice latency creeping up as your historical dataset grows larger, the best optimization is to compute your rolling indicators incrementally or cache the historical feature dataframe so you only append the newest live candle rather than recalculating indicators from scratch every single minute. But right now, your architecture is more than fast enough for intraday execution!
