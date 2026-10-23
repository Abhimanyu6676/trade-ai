# Market Data Engine

Implementation: `data_engine.py`. It is called by `app.py` for a market-analysis request.

## Request and history window

- The UI/parser supplies a ticker, relative `days_back`, source interval, and exchange. Only `1m` is accepted.
- The request ends on the machine's current local date. The display window begins `int(days_back * 1.5) + 5` calendar days before that date to include a weekend/holiday buffer.
- The OpenAlgo request begins another 30 calendar days before the display window to warm up multi-timeframe features, including the 1h EMAs.
- This code does not accept a separately supplied target date or intraday start/end time. It makes one OpenAlgo history request for the calculated range.

## Processing

1. `openalgo.get_history()` returns candles using the shared client in `openalgo_client.py`.
2. `merge_ohlcv_history()` lowercases OHLCV names, parses timestamps, stable-sorts candles, and keeps the last row on duplicate timestamps.
3. If the result exceeds 25,000 rows, the oldest rows are trimmed after sorting.
4. `compute_normalized_features(..., model_type="gru")` calculates display features on the complete fetched history. Feature calculation copies the source frame.
5. The raw OHLCV frame is passed to `run_ai_prediction_inference()`, which computes the separate LightGBM and GRU feature variants itself. App inference uses the shared 30-candle `DEFAULT_SEQUENCE_LENGTH`.
6. The processed display data is then sliced to the display-window start. It is not sliced to a specific target day or clock-time window before prediction.

On success, the function returns `(display_dataframe, signal, confidence)`. On failure, it returns `(None, message, 0.0)`. Display columns are selected from timestamp/OHLCV and optional named fields when those fields exist.

## Operational limits

- The 30-day warm-up is a calendar buffer, not a promise that an EMA has reached mathematical convergence.
- Row limits, lookback dates, and the prediction horizon are defined in this module; no 10-million-row batch loop is implemented.
- Exceptions are caught and returned as an execution-failure message.
