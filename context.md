# trade-ai Current Context

Last reviewed: 2026-10-30.

## Purpose and Runtime Flow

`trade-ai` forecasts trade-relative outcomes from one-minute OHLCV candles. It
has long and short LightGBM/GRU specialists plus one PPO meta-agent that chooses
flat, long, or short exposure and a size level. FastAPI exposes forecasts,
OpenAlgo history, and analyser status. The application does not place broker
orders; a prediction is a decision signal, not an execution request.

`app.py` loads the four base checkpoints and the PPO artifact through
`model/models.py`, then starts Uvicorn in the main process. The model manager
checks versioned sidecars, direction, ordered input columns, and the four base
checkpoint hashes recorded in the PPO sidecar. Missing or stale artifacts are
left unloaded. Prediction requires all four base specialists and a compatible
PPO policy; otherwise `/api/brain/predict` returns HTTP 503.

`model/prediction_engine.py` gets prior-session history from `model/dataStore.py`,
fetches target-day candles when callers did not supply them, and runs
feature/inference work on worker threads. It computes long/short model features,
gets four probability vectors, builds the versioned PPO state, and maps its
action to flat/long/short plus a size multiplier. Callers may supply current
position, unrealized return, and bars held; omitted context means flat. The
shared OpenAlgo history, holiday-calendar, and analyser-status client methods
offload synchronous SDK calls to worker threads. Non-empty successful
holiday calendars are cached by year; failed, invalid, and empty responses are
not cached.

`POST /api/brain/predict` accepts ticker, exchange, lookback, interval, optional
candles, and optional portfolio context. `POST /api/openalgo/history` fetches
market history; `GET /api/openalgo/analyser-status` queries analyser status.
All endpoints return `UniversalServerResponse` envelopes. `server/serverTypes.py`
owns the generic response envelope, module domain types live in their module's
`*Types.py`, and route-specific contracts live with their routes. FastAPI runs
on port 3010; browser origins come from `TRADE_AI_CHAT_ORIGINS`, defaulting to
`http://localhost:3011,http://127.0.0.1:3011`.

## Logging

`src/trade_ai/logger.py` configures the shared `trade_ai` logger at INFO level
with timestamp, severity, and message fields. Application modules use this
logger instead of creating module-specific loggers. Python's
`logging.basicConfig()` preserves an existing host application's logging setup.

## Data and Feature Contract

- Source candles use lowercase `timestamp`, `open`, `high`, `low`, `close`, and
  `volume` fields and timestamps at candle open. History, caches, and supplied
  candles pass through `util.merge_ohlcv_history()`: it normalizes columns,
  promotes a timestamp-like index when needed, converts timestamps to
  Asia/Kolkata local time, stable-sorts, and keeps the later-appended duplicate.
- The prediction frame uses the latest supplied candle date, or today when
  candles are omitted. It includes at least 30 prior exchange sessions. Current
  target-day candles are merged transiently and are not saved in `dataStore`.
- `model/features.py` builds the same multi-timeframe technical inputs for both
  directions. Raw OHLCV remains available for calculations but is never a model
  input. GRU excludes ticker metadata; LightGBM receives fixed integer-coded
  `Cap` and `Sector` values as categorical inputs. See
  `src/trade_ai/model/features.md` for the feature schema and full inventories.
- `indicator/atr.py:calculate_atr()` accepts `relative` (default `True`): the
  default returns ATR divided by close; `relative=False` returns price-unit
  ATR. Feature construction uses the relative default, and PPO consumes
  `ATR_14_1m` directly as volatility context without dividing it a second time.
- The current feature schema is version 7: 35 GRU inputs and 37 LightGBM inputs.
  Both directional specialists use the same schemas. Active intervals are 1m,
  5m, 15m, and 1h; 3m and 30m are excluded.
- `model/dataStore.py` owns the process-wide `CandleStore`, keyed by normalized
  `TICKER:EXCHANGE`. It verifies expected trading dates, logs intraday gaps
  greater than one minute, and does not synthesize missing candles. It is
  cleared when the process exits.
- Multi-timeframe bars map back with backward `merge_asof` joins so only
  completed bars are used. Feature history includes prior sessions to warm up
  indicators such as the 1h EMA 50. GRU windows are restricted to one calendar
  date, but do not enforce exchange hours or exact one-minute continuity.
- The 30-day buffer does not guarantee a fully converged EMA. The market close
  is 15:15; the 15:14 candle is the last eligible candle and 15:15 is excluded.

## Labels and Training

- `config/trading_config.py` centralizes `SCHEMA_VERSION`, direction-specific
  fixed barrier distances, holding period, entry/close times, artifact paths,
  and simulator assumptions. Increment
  `SCHEMA_VERSION` when features, labels, directional semantics, PPO
  observations, or actions change. Feature and label schema versions remain
  separately recorded for precise loader errors. The PPO action and observation
  contract is schema version 3; because the trading schema is also stored in
  base-model manifests, all version-2 base checkpoints and PPO artifacts require
  retraining before use.
- `training/long/labels_long.py` uses +1% take profit and -0.5% stop loss.
  `training/short/labels_short.py` mirrors those as -1% take profit and +0.5%
  stop loss. Both labelers use trade-relative classes: `1` target first, `-1`
  stop first, and `0` timeout. Entries run 10:00–14:30; look-ahead stops at 60
  bars, the session boundary, or 15:15. The label schema is version 4.
- Base trainers iterate over every ticker key in `STOCK_METADATA` and use each
  ticker with sufficient available history. They derive one pooled set of
  chronological session boundaries so all tickers share the same dates: the
  first 55% fits the initial selector, the next 15% selects LightGBM boosting
  rounds or the GRU epoch, the next 20% is divided into four expanding OOS
  forecast folds, and the newest 10% is a final holdout. Each fit boundary
  purges up to 60 candle rows. Accepted base runs then refit on all eligible
  data before the final holdout with the selected iteration/epoch count fixed.
- For each expanding fold, the base model is freshly trained only on rows before
  that fold (again purging up to 60 rows) and predicts the next block. The four
  resulting specialist streams carry ticker, timestamp, market state, fold ID,
  and training cutoff. Manifests validate schemas and final checkpoint hashes;
  PPO joins only shared ticker/timestamp rows and requires matching provenance.
  GRU predictions require a full valid same-session sequence.
- PPO fits on the earlier 80% of the walk-forward OOS dates and selects its
  deterministic policy on the later 20%. It then reports on forecasts from
  the newest 10% final holdout, evaluating every available ticker/session
  episode. The holdout result is recorded but does not gate saving; the
  save/reject decision uses the policy-selection period against a flat baseline.
- Date partitions are chronological, not random. GRU shuffles minibatches only
  within its fit rows; LightGBM's bagging and feature-fraction settings sample
  during fitting; PPO samples ticker/session episodes only within its fit
  partition. These optimization choices do not move rows across date splits.
  The automated boundaries, purging, frozen-base OOS inputs, provenance checks,
  and untouched final holdout reduce leakage risk but cannot detect look-ahead
  already embedded in source data/features or remove survivorship bias. Repeated
  manual tuning after viewing the final holdout can also contaminate it.
- The base and PPO validation/selection steps run inside the training modules:
  LightGBM compares validation logloss with a training class-prior baseline,
  GRU selects the validation-loss epoch, and PPO uses chronological selection
  reward. Checkpoint schemas and prediction manifests are written by the
  trainers and checked by the model manager. This automates the workflow; it is
  not a proof that every upstream input is leakage-free.
- The PPO environment starts from the same 1%/0.5% direction-specific baseline
  barriers as the labels. At entry the policy can scale both distances by 0.75,
  1.0, or 1.25; their 2:1 ratio stays fixed. It models close-to-close returns,
  a 60-bar maximum hold, estimated 5 bp one-way cost, model-disagreement entry
  cost, and incremental drawdown penalty. Costs are assumptions, not measured
  broker costs.
- The RL observation contains four three-class model outputs, long/short
  agreement, scaled relative 1m ATR and volatility ratio, portfolio position,
  unrealized return, bars held, and the active bracket multiplier. ATR is
  volatility context; it does not set the profit or stop distance. Its 19
  actions combine flat or long/short positions at 0.5x/1x/1.5x size with
  0.75x/1x/1.25x bracket width. The 20-value observation and 19-action
  contracts are stored in the policy sidecar. Probabilities/agreement are 0–1,
  ATR and volatility ratio are clipped/scaled, portfolio position and return
  are clipped/scaled, and bars are divided by the 60-bar limit.
- `training/util.py` loads cached ticker CSVs or downloads adjacent inclusive
  30-calendar-day OpenAlgo chunks with up to three attempts. The current default
  download window ends 2026-07-31; a ticker listed after that date, such as JPL,
  is skipped unless a newer local cache exists. Empty pre-listing chunks are
  skipped; an empty chunk after history begins aborts that ticker to prevent
  gaps. Failed downloads are not cached, and successful history caches are not
  automatically refreshed.
- `training/train_master.py` accepts `long|short|both` (default `both`) and
  `gru|lgb|rl|all` (default `all`). It backs up selected artifacts, unloads
  selected models, runs selected specialists and/or the shared PPO trainer,
  then reloads only schema-compatible artifacts. RL is common to all four base
  models. See `model/training/training.md` for lifecycle and foreground/
  background commands.
- Model artifacts live in `src/trade_ai/model/.run_cache/long/`, `short/`, and
  `rl/`. The PPO sidecar includes the exact base checkpoint hashes, so changing
  any base model invalidates the current meta-agent until it is retrained.

## Prediction Behavior and Known Limits

- Inference requires loaded long LightGBM, long GRU, short LightGBM, short GRU,
  and a schema-compatible PPO model. The policy selects a flat/long/short action
  and size multiplier; the endpoint does not place orders.
- The base models each emit classes in order `[-1, 0, 1]`, meaning
  stop/timeout/take-profit relative to that model's own direction. The PPO
  observation contains all four raw probability vectors and model agreement.
- PPO inference uses deterministic actions. For API compatibility,
  `signal` remains `-1`, `0`, or `1`; it now represents short, flat, or long.
  `confidence` is the selected side's mean take-profit probability, not a
  calibrated guarantee. A flat decision reports 0 confidence.
- GRU training and inference share the 30-candle sequence length. Session
  grouping uses calendar date and does not verify market hours or missing-minute
  continuity. The simulator likewise uses close-to-close price movement and
  does not model intrabar order, bid/ask spread, market impact, partial fills,
  or actual broker execution.
- The PPO policy is trained on historical simulation data. It can overfit even
  with chronological splits; evaluate it on a later untouched period and paper
  trade before relying on its decisions.

## Structured Brain API

- `GET /api/event/test` accepts `ticker` and optional `exchange` (default
  `NSE`), then returns normalized ticker and cached candle history in a
  `SERVER/SERVER_STATUS` response.
- `POST /api/brain/predict` accepts ticker, `NSE`/`BSE`, `days_back`, `1m`,
  optional caller candles (1–25,000), and optional portfolio context
  (`portfolio_position`, `unrealized_return`, `bars_in_position`, and optional
  `barrier_multiplier`, default 1.0). At least 30 prior trading sessions are loaded;
  `days_back` is capped at 90.
- Prediction success returns `MODEL/PREDICTION` with signal, confidence,
  processed candle count, latest `prediction_timestamp`, the four specialist
  probability arrays, long/short mean arrays, PPO action index, selected
  position size and barrier multiplier, and scaled direction-specific
  take-profit/stop-loss distances (null for flat). Class probability arrays use
  `[-1, 0, 1]` order. Previous
  `lgb_probabilities`, `gru_probabilities`, and `ensemble_probabilities` fields
  remain aliases for the long specialists.
- Invalid or insufficient requests use HTTP 422; unavailable base models or PPO
  use 503; upstream history errors use 502; unexpected errors use 500. Errors
  remain strings in `data.error`.
- OpenAlgo history and analyser endpoints use the same generic response
  envelope. Keep this file and `server/server.md` current when endpoint behavior
  changes.

## Source Map

- `src/trade_ai/config/trading_config.py`: trading schema, label barriers,
  simulator assumptions, and artifact directories.
- `src/trade_ai/model/features.py` and `model/training/gru_sequence_builder.py`:
  feature schemas and same-session GRU sequence helpers.
- `src/trade_ai/model/training/long/` and `short/`: directional labels,
  trainer entry points, and directional simulator wrappers.
- `src/trade_ai/model/training/labeling.py`: shared direction-aware barrier
  implementation; `training/rl/` joins base OOS predictions.
- `src/trade_ai/model/training/train_lgb.py`, `train_gru.py`, `train_rl.py`,
  and `train_master.py`: training implementations and orchestration.
- `src/trade_ai/model/training/trading_env.py`: PPO observation contract and
  historical simulator.
- `src/trade_ai/model/models.py`: four base models, PPO loading, schema checks,
  in-memory ownership, backups, unload, and reload lifecycle.
- `src/trade_ai/model/prediction_engine.py`: history orchestration, four-model
  inference, and policy decision.
- `src/trade_ai/openalgo/`, `src/trade_ai/util.py`, and
  `src/trade_ai/model/stock_metadata.py`: broker data client, candle utilities,
  and ticker metadata.
- `src/trade_ai/server/`: FastAPI routes and API guidebook. `model/modelTypes.py`,
  `openalgo/openalgoTypes.py`, and `universalTypes.py` own model, OpenAlgo, and
  global endpoint types respectively.
- The OpenAlgo SDK's `analyzerstatus()` method is synchronous; call it without
  `await` and normalize its dictionary response in `OpenAlgo.get_analyzer_status()`.

## Environment and Commands

- Python 3.14 or newer, `uv`, Gymnasium, Stable-Baselines3, and GNU OpenMP
  (`libgomp1` on Debian/Ubuntu) are required.
- `environmentLoader.py` requires `OPENALGO_API_KEY`, `OPENALGO_HOST`, and
  `TRADE_AI_CHAT_ORIGINS`, loading the repository-root `.env` while preserving
  existing process environment values. Never store credentials in tracked
  files or logs.
- Install dependencies with `uv sync`; start the app with
  `uv run python src/trade_ai/app.py`.
- Train using `uv run python -m trade_ai.model.training.train_master [long|short|both] [gru|lgb|rl|all]`.
- No test command is configured in `pyproject.toml`; do not add or run tests
  unless requested.
