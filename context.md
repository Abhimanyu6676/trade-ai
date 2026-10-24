# trade-ai Current Context

Last reviewed: 2026-10-24.

## Purpose and Runtime Flow

`trade-ai` is a Python application for market-data forecasts. It trains a LightGBM classifier and a PyTorch GRU classifier on three Triple Barrier outcomes. FastAPI exposes structured prediction requests, OpenAlgo history, and analyser status. It does not submit broker orders.

The market-analysis path is:

1. `app.py` initializes the model manager, which loads available checkpoints, then starts Uvicorn in the main process. FastAPI prediction requests call the async data engine directly.
2. `model/data_engine.py` fetches one-minute OHLCV candles through the shared OpenAlgo client. It requests a 30-calendar-day feature warm-up before the display-history range, sorts/deduplicates candles, and caps input at 25,000 rows.
3. `model/features.py` builds the multi-timeframe features. The data engine uses this result for the display grid and passes the raw OHLCV frame to inference.
4. `model/prediction_engine.py` engineers model-specific features, gets probabilities from the loaded model instances, and blends the probabilities when both models are ready. `model/models.py` owns the in-memory model instances and all loading, unloading, backup, and reload operations.

`POST /api/brain/predict` accepts structured ticker, exchange, lookback, and interval fields; supplied candle arrays are validated by `OhlcvCandle` in `openalgo/openalgoTypes.py`. The API process fetches candles and runs inference directly. The current engine supports `1m` source candles only. `POST /api/openalgo/history` fetches broker history through the shared OpenAlgo client and returns symbol, exchange, interval, and candles together in `data.data`; `GET /api/openalgo/analyser-status` queries analyser availability. All endpoints return `UniversalServerResponse` with an intent and output `eventType`; success data is stored in `data.data` and errors in `data.error`. `server/serverTypes.py` owns the generic response envelope, module-related domain types belong in their module's `*Types.py` file, and route-specific response contracts are defined with their routes. `model/modelTypes.py` and `openalgo/openalgoTypes.py` contain domain types and request/event literals, while `universalTypes.py` contains global intent/event types. Starting `app.py` loads the prediction models and runs FastAPI on port 3010 in the main process. The server allows browser requests from origins listed in `TRADE_AI_CHAT_ORIGINS`, defaulting to `http://localhost:3011,http://127.0.0.1:3011`.

## Data and Feature Contract

- Source data is one-minute candles timestamped at candle open. Expected columns are `timestamp`, `open`, `high`, `low`, `close`, and `volume`.
- Candle columns use the lowercase schema `timestamp`, `open`, `high`, `low`, `close`, and `volume`. The shared OHLCV normalizer applies to OpenAlgo responses, cached CSVs, training downloads, direct feature inputs, and merged history. `util.merge_ohlcv_history()` also parses timestamps, stable-sorts chronologically, and keeps the later-appended row for duplicate timestamps. The data engine applies the 25,000-row cap after sorting, preserving the newest tail.
- For a request of `days_back`, the display start is calculated as `today - (int(days_back * 1.5) + 5)` calendar days. The data request starts another 30 calendar days earlier to warm up indicators. The displayed frame is sliced back to the display start after feature calculation. This is a relative lookback from today, not an arbitrary target-date/time-window API.
- `FEATURE_COLUMNS` in `model/features.py` contains 55 engineered inputs: three return features; seven indicators for each of 1m, 3m, 5m, 15m, 30m, and 1h (RSI, ATR, EMA 9/21/50 distances, FVG, and liquidity sweep); plus ten stock metadata flags.
- The model training and inference code selects all computed frame columns except `timestamp` and `target`; with standard OHLCV input, this means the 55 engineered columns plus the five raw OHLCV values (60 inputs). Extra columns returned by a data source can also enter this dynamic selection, so use the same input schema for training and inference.
- `model_type="gru"` scales RSI to 0–1; `model_type="lgb"` keeps RSI on its regular scale. The other listed feature calculations are shared. Multi-timeframe bars are mapped back with backward `merge_asof` joins to use completed bars only.
- Feature calculation runs on the full chronological history, allowing prior sessions to warm up higher-timeframe indicators such as EMA 50 on 1h bars. GRU sequence windows are separately restricted to one calendar date.
- The 30-day buffer is not a mathematical guarantee that an EMA is fully converged. The user-facing market close is 15:15, with candle timestamps at candle open; the 15:14 candle is the last eligible candle and 15:15 is excluded.

## Labels and Training

- `model/training/labels.py` creates three mutually exclusive targets: `-1` stop-loss first, `0` timeout, `1` take-profit first. It checks future candle closes, allows entries from 10:00 through 14:30 by default, scans at most 60 future bars, and stops at the entry date boundary or 15:15 market close. Inputs must already be sorted chronologically.
- `model/training/train_lgb.py` loads each ticker's cached CSV or downloads history from 2020 in 30-calendar-day API chunks (three retries per chunk). It calculates LGB features, labels rows, maps classes to `0/1/2`, concatenates per-ticker data, and trains up to 500 LightGBM rounds with early stopping. Its 80/20 split is unshuffled after ticker-wise concatenation; it is not a globally interleaved time split.
- `model/training/train_gru.py` trains a two-layer, 64-hidden-unit, three-class GRU. Defaults are 10 epochs, 30 candles per sequence, batch size 256, and Adam. Sliding windows must remain on one calendar date; unlabeled feature rows can serve as input context, but a window's final row must have a label. The DataLoader shuffles the valid windows and drops incomplete batches.
- Both trainers use `src/trade_ai/tickData/<TICKER>_history.csv` when it exists. Existing cache files are read as-is; they are not automatically refreshed. If missing, the trainers download and cache history.
- `model/training/train_master.py` accepts `gru`, `lgb`, or `both` (default). It backs up loaded checkpoint files to timestamped copies, unloads the selected models, runs their training functions, then reloads the selected checkpoints. Run it with `uv run python -m trade_ai.model.training.train_master [gru|lgb|both]`; see `model/training/training.md` for the step-by-step lifecycle.
- Model files are written under `src/trade_ai/model/.run_cache/`: `lgb_trading_model.txt` and `gru_trading_model.pth`.

## Prediction Behavior and Known Limits

- `model/models.py` loads each existing checkpoint when the application starts. It infers GRU input width from the checkpoint, so GRU loading does not depend on a first prediction or cached candle data. Missing files produce a warning to train that model first.
- `model/prediction_engine.py` only requests loaded instances from `models.py`, performs inference, and blends available model probabilities. It does not load, unload, back up, or reload models.
- If both models return probabilities, the engine takes a 50/50 arithmetic mean. If only one returns probabilities, it uses that model alone. It returns the class with the largest probability (`-1`, `0`, or `1`) and that probability as confidence. Probabilities are not calibrated confidence guarantees.
- GRU training and app inference use the shared `DEFAULT_SEQUENCE_LENGTH` of 30 candles, defined in `model/training/sequence_loader.py`.
- `sequence_loader.latest_session_sequence()` selects the last requested number of valid feature rows from the newest calendar date. It does not verify market hours or that timestamps are exactly one minute apart. Training windows likewise enforce calendar-date boundaries, not missing-candle continuity.
- `GRUTradingBrain` is defined in `model/models.py` and shared by the model manager and GRU training module.
- Training is orchestrated through `train_master.py`; it backs up current selected checkpoints, unloads selected models for training, then reloads the resulting checkpoint files when training ends.
- There is no liquidity/spread filter, probability calibration, automatic order submission, or documented 10-million-row processing loop in the active pipeline.

## Structured Brain API

- `POST /api/brain/predict` accepts explicit `ticker`, `exchange`, `days_back`, and `interval` fields. Exchange is `NSE` or `BSE`; the current inference pipeline accepts `1m` only.
- It can use either OpenAlgo history (`days_back`) or caller-supplied `candles` containing timestamp and lowercase OHLCV fields. The supplied array is limited to 25,000 rows and is validated before inference.
- The route returns a `MODEL/PREDICTION` envelope. Its response data contains the signal, confidence, and processed candle count; errors are strings in `data.error`.
- Invalid requests also return the same envelope with a string error and HTTP 422.

## Source Map

- `src/trade_ai/app.py`: initializes the model manager and runs the API in the main process.
- `src/trade_ai/model/data_engine.py`: OpenAlgo market-history fetch, warm-up buffer, feature display, inference call.
- `src/trade_ai/model/features.py` and `src/trade_ai/model/training/sequence_loader.py`: feature schema and same-session GRU windows.
- `src/trade_ai/model/training/labels.py`: Triple Barrier target generation used by both trainers.
- `src/trade_ai/model/training/`: LightGBM and GRU trainers, master orchestration, and `training.md` lifecycle notes.
- `src/trade_ai/model/models.py`: GRU architecture, in-memory model manager, loading, unloading, timestamped checkpoint backup, and reload.
- `src/trade_ai/model/prediction_engine.py`: read-only ensemble inference using model instances from `models.py`.
- `src/trade_ai/openalgo/openalgo_client.py`, `src/trade_ai/openalgo/helper.py`, `src/trade_ai/util.py`, `src/trade_ai/model/stock_metadata.py`: shared client, OpenAlgo response conversion, candle utilities, metadata inputs.
- The OpenAlgo SDK's `analyzerstatus()` method is synchronous; call it without `await` and normalize its dictionary response in `OpenAlgo.get_analyzer_status()`.
- `src/trade_ai/indicator/`: indicator calculations and their explanatory Markdown files.
- `src/trade_ai/server/`: FastAPI application, `predict_route.py` prediction route, and `openalgo_route.py` history and analyser status routes.
- `src/trade_ai/server/serverTypes.py`: generic `UniversalServerResponse` envelope and response-data wrapper.
- `src/trade_ai/model/modelTypes.py`, `src/trade_ai/openalgo/openalgoTypes.py`, and `src/trade_ai/universalTypes.py`: internal domain/request types, event literals, and global intent/event literals. `modelTypes.py` owns the `ModelName` literal and `MODEL_NAMES` tuple; `openalgoTypes.py` owns OHLCV candle data and the validated prediction candle request model. Routes and model code import these module types when needed.

## Environment and Commands

- Python 3.14 or newer and `uv` are required.
- LightGBM's native library needs the GNU OpenMP runtime (`libgomp1` on Debian/Ubuntu).
- `OPENALGO_API_KEY` is required. `OPENALGO_HOST` is optional and defaults to the host in `openalgo_client.py`.
- The app does not load `.env` automatically; export its values into the shell before starting.
- Install dependencies with `uv sync`.
- Start the app with `uv run python src/trade_ai/app.py` (FastAPI 3010; models load in the main process).
- Run selected model training in the foreground with `uv run python -m trade_ai.model.training.train_master [gru|lgb|both]` (default: both).
- No test command is currently configured in `pyproject.toml`.
