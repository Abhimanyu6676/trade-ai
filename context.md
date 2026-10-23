# trade-ai Current Context

Last reviewed: 2026-10-23.

## Purpose and Runtime Flow

`trade-ai` is a Python application for market-data forecasts. It trains a LightGBM classifier and a PyTorch GRU classifier on three Triple Barrier outcomes. It displays forecasts through Gradio and exposes a small FastAPI parser API. It does not submit broker orders.

The market-analysis path is:

1. `app.py` asks `ai_parser.py` to extract an action, ticker, and `days_back` from the prompt. The parser uses local Ollama (`qwen2.5:0.5b`) except for directly detected training requests.
2. `brain/data_engine.py` fetches one-minute OHLCV candles through the shared OpenAlgo client. It requests a 30-calendar-day feature warm-up before the display-history range, sorts/deduplicates candles, and caps input at 25,000 rows.
3. `brain/features.py` builds the multi-timeframe features. The data engine uses this result for the display grid and passes the raw OHLCV frame to inference.
4. `brain/prediction_engine.py` engineers model-specific features, gets probabilities from the available model(s), and blends the probabilities when both models are ready.

The FastAPI `/api/chat` routes only parse prompts; they do not fetch market data or run predictions. `GET /api/openalgo/analyser-status` queries the configured OpenAlgo client and returns a small status summary for the browser chat. Starting `app.py` starts Gradio on port 7860 and the FastAPI server on port 3010. The server allows browser requests from the Astro chat origins listed in `TRADE_AI_CHAT_ORIGINS`, a comma-separated environment variable that defaults to `http://localhost:3011,http://127.0.0.1:3011`. Set it to the deployed frontend origin or origins when hosting the chat elsewhere.

## Data and Feature Contract

- Source data is one-minute candles timestamped at candle open. Expected columns are `timestamp`, `open`, `high`, `low`, `close`, and `volume`.
- Candle columns use the lowercase schema `timestamp`, `open`, `high`, `low`, `close`, and `volume`. The shared OHLCV normalizer applies to OpenAlgo responses, cached CSVs, training downloads, direct feature inputs, and merged history. `util.merge_ohlcv_history()` also parses timestamps, stable-sorts chronologically, and keeps the later-appended row for duplicate timestamps. The data engine applies the 25,000-row cap after sorting, preserving the newest tail.
- For a request of `days_back`, the display start is calculated as `today - (int(days_back * 1.5) + 5)` calendar days. The data request starts another 30 calendar days earlier to warm up indicators. The displayed frame is sliced back to the display start after feature calculation. This is a relative lookback from today, not an arbitrary target-date/time-window API.
- `FEATURE_COLUMNS` in `brain/features.py` contains 55 engineered inputs: three return features; seven indicators for each of 1m, 3m, 5m, 15m, 30m, and 1h (RSI, ATR, EMA 9/21/50 distances, FVG, and liquidity sweep); plus ten stock metadata flags.
- The model training and inference code selects all computed frame columns except `timestamp` and `target`; with standard OHLCV input, this means the 55 engineered columns plus the five raw OHLCV values (60 inputs). Extra columns returned by a data source can also enter this dynamic selection, so use the same input schema for training and inference.
- `model_type="gru"` scales RSI to 0–1; `model_type="lgb"` keeps RSI on its regular scale. The other listed feature calculations are shared. Multi-timeframe bars are mapped back with backward `merge_asof` joins to use completed bars only.
- Feature calculation runs on the full chronological history, allowing prior sessions to warm up higher-timeframe indicators such as EMA 50 on 1h bars. GRU sequence windows are separately restricted to one calendar date.
- The 30-day buffer is not a mathematical guarantee that an EMA is fully converged. The user-facing market close is 15:15, with candle timestamps at candle open; the 15:14 candle is the last eligible candle and 15:15 is excluded.

## Labels and Training

- `brain/labels.py` creates three mutually exclusive targets: `-1` stop-loss first, `0` timeout, `1` take-profit first. It checks future candle closes, allows entries from 10:00 through 14:30 by default, scans at most 60 future bars, and stops at the entry date boundary or 15:15 market close. Inputs must already be sorted chronologically.
- `brain/train_lgb.py` loads each ticker's cached CSV or downloads history from 2020 in 30-calendar-day API chunks (three retries per chunk). It calculates LGB features, labels rows, maps classes to `0/1/2`, concatenates per-ticker data, and trains up to 500 LightGBM rounds with early stopping. Its 80/20 split is unshuffled after ticker-wise concatenation; it is not a globally interleaved time split.
- `brain/train_gru.py` trains a two-layer, 64-hidden-unit, three-class GRU. Defaults are 10 epochs, 30 candles per sequence, batch size 256, and Adam. Sliding windows must remain on one calendar date; unlabeled feature rows can serve as input context, but a window's final row must have a label. The DataLoader shuffles the valid windows and drops incomplete batches.
- Both trainers use `src/trade_ai/tickData/<TICKER>_history.csv` when it exists. Existing cache files are read as-is; they are not automatically refreshed. If missing, the trainers download and cache history.
- `brain/train_master.py` runs LightGBM then GRU. The UI's “start training” action runs the master pipeline on a daemon thread. A lock protects start/status state and prevents duplicate concurrent starts in that app process. The module can also run synchronously with `uv run python -m trade_ai.brain.train_master`.
- The master trainer does not currently call the prediction engine's reload method after it saves checkpoints. If an inference engine already holds older model objects when background training finishes, restart the app or explicitly reload the prediction engine before expecting it to use the new checkpoints.
- Model files are written under `src/trade_ai/.run_cache/`: `lgb_trading_model.txt` and `gru_trading_model.pth`.

## Prediction Behavior and Known Limits

- `brain/prediction_engine.py` eagerly loads an existing LightGBM model and lazily initializes/loads the GRU when prediction is requested.
- If both models return probabilities, the engine takes a 50/50 arithmetic mean. If only one returns probabilities, it uses that model alone. It returns the class with the largest probability (`-1`, `0`, or `1`) and that probability as confidence. Probabilities are not calibrated confidence guarantees.
- GRU training and app inference use the shared `DEFAULT_SEQUENCE_LENGTH` of 30 candles, defined in `brain/sequence_loader.py`.
- `sequence_loader.latest_session_sequence()` selects the last requested number of valid feature rows from the newest calendar date. It does not verify market hours or that timestamps are exactly one minute apart. Training windows likewise enforce calendar-date boundaries, not missing-candle continuity.
- `model.py` contains a separate `StockPredictionBrain` and schema-tagged checkpoint helpers; the active master training/inference flow uses `GRUTradingBrain` in `brain/train_gru.py` and does not use those helpers.
- There is no liquidity/spread filter, probability calibration, automatic order submission, or documented 10-million-row processing loop in the active pipeline.

## Source Map

- `src/trade_ai/app.py`: Gradio UI and action routing.
- `src/trade_ai/ai_parser.py`: local Ollama prompt parsing.
- `src/trade_ai/brain/data_engine.py`: OpenAlgo market-history fetch, warm-up buffer, feature display, inference call.
- `src/trade_ai/brain/features.py`, `labels.py`, `sequence_loader.py`: feature schema, Triple Barrier targets, same-session GRU windows.
- `src/trade_ai/brain/train_lgb.py`, `train_gru.py`, `train_master.py`: individual trainers and master orchestration.
- `src/trade_ai/brain/prediction_engine.py`: model loading and ensemble inference.
- `src/trade_ai/openalgo_client.py`, `util.py`, `stock_metadata.py`: shared client, candle utilities, metadata inputs.
- `src/trade_ai/indicator/`: indicator calculations and their explanatory Markdown files.
- `src/trade_ai/server/`: FastAPI application and route modules, including the OpenAlgo analyser status route under `openalgo/`.

## Environment and Commands

- Python 3.14 or newer and `uv` are required.
- LightGBM's native library needs the GNU OpenMP runtime (`libgomp1` on Debian/Ubuntu).
- `OPENALGO_API_KEY` is required. `OPENALGO_HOST` is optional and defaults to the host in `openalgo_client.py`.
- The app does not load `.env` automatically; export its values into the shell before starting.
- Install dependencies with `uv sync`.
- Start the app with `uv run python src/trade_ai/app.py` (Gradio 7860; FastAPI 3010).
- Run the full dual-model training pipeline in the foreground with `uv run python -m trade_ai.brain.train_master`, or request “start training” in the UI for background execution.
- No test command is currently configured in `pyproject.toml`.
