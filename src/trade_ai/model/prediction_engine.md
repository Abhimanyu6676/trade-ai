# Ensemble Prediction Engine

Implementation: `prediction_engine.py`. Four directional specialists predict
triple-barrier outcome probabilities (`-1` stop, `0` timeout, `1` target); the
PPO policy combines them into a flat, long, or short decision with a position
size and entry bracket width. It does not place trades.

## Model ownership and loading

- `models.py` owns the active model instances and loads four directional base
  checkpoints plus the PPO checkpoint at application startup.
- The GRU input width is read from `gru.weight_ih_l0` in the saved state dict, so loading does not wait for a prediction or read cached candle data.
- The default meta-ensemble requires all four base checkpoints and a
  schema-compatible PPO checkpoint. A specialist-only request requires only its
  selected compatible base checkpoint and does not invoke PPO.
- `GRUTradingBrain` is defined in `models.py`: two GRU layers, hidden size 64, and a three-class output layer.
- The manager also owns unloading, timestamped backups, in-memory instance access, and checkpoint reloads. Training orchestration requests those operations through the manager.

## Inference flow

1. The route calls `predict_with_history()`. It determines the target date from the latest supplied candle, or today when candles are omitted.
2. It requests at least 30 prior trading sessions from `dataStore`, which uses the exchange's previous-trading-day calendar. If no caller candles were supplied, it fetches the target day's candles from OpenAlgo.
3. Target-day candles are merged transiently with the prior history. They are not added to the in-memory history store.
4. `predict_from_dataframe()` passes the merged frame to `_predict_normalized()`;
   the direct `predict()` helper normalizes its input and calls the same function.
   The selected model instances come from `models.py`.
5. The requested model family determines which feature pipeline runs. The
   meta-ensemble computes both LightGBM and GRU features; a specialist request
   computes only its own family. ATR defaults to relative-to-close values in
   feature construction; PPO consumes that relative value directly as
   volatility context.
6. For `target_model=meta_ensemble`, long and short LightGBM specialists predict from the latest valid feature
   row. Their GRU counterparts receive the last `sequence_length` valid rows
   from the newest calendar date only.
7. The four specialist probability vectors, relative ATR, volatility ratio,
   and optional portfolio state become the PPO observation. Its 19-action
   contract selects flat or a directional size and one of the 0.75, 1.0, or
   1.25 bracket multipliers. At entry, the multiplier scales both the baseline
   target and stop, preserving the 2:1 ratio; it remains fixed while the
   position is open. The API returns a fixed `PredictionBreakdown` shape:
   `target_model`, root-level final `signal` and policy fields, and nested
   `long` and `short` objects. Each direction contains `lgb` and `gru`
   classifier objects with `probabilities` and `predicted_class`, plus its
   `ensemble_probabilities`. The final PPO signal/action, selected size, and
   bracket distances remain at the root. The root has no classifier
   `predicted_class` or probability array.
   The selected-side take-profit probability is returned as confidence.

For `target_model` set to a base specialist (`long_lgb`, `long_gru`,
`short_lgb`, or `short_gru`), only that model is required and run. The API
returns its highest-probability class and probability vector under that
specialist's direction/model path (for example, `breakdown.long.lgb`). The
opposite direction is null, and the unselected model and ensemble
probabilities in the selected direction are null. Classes are ordered as
stop, timeout, and take profit relative to that model's direction. It does not
run PPO; signal, position size, and bracket outputs are null. All fields
remain present in the fixed response schema, with null used for values that do
not apply.

## Current limitations

- GRU training and app inference use the shared `DEFAULT_SEQUENCE_LENGTH` of 30 candles, defined in `training/gru_sequence_builder.py`.
- Session grouping uses calendar date. It does not enforce exchange hours or exact one-minute timestamp continuity.
- The history store requires every expected exchange trading date and logs observed intraday gaps over one minute. It does not synthesize candles for missing intervals because those intervals may have had no trades.
- Confidence is a specialist probability, not a calibrated guarantee.
- If the newest session lacks enough valid same-date GRU rows, GRU specialist
  and meta-ensemble requests return an insufficient-data error.
- `prediction_engine.py` does not change model state. `train_master.py` coordinates backup, unload, training, and reload through `models.py`.


## Pro-Tip for Maximum Speed (Optional) TODO - do not remove this until its applied

    If you ever notice latency creeping up as your historical dataset grows larger, the best optimization is to compute your rolling indicators incrementally or cache the historical feature dataframe so you only append the newest live candle rather than recalculating indicators from scratch every single minute. But right now, your architecture is more than fast enough for intraday execution!
