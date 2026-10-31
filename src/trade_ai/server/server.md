# Server Endpoint API

## Server

`app.py` loads four directional base models and the unified PPO policy, then
runs Uvicorn on port `3010` in the main process. Prediction requests fetch
candles, compute the four model forecasts, and run the policy in the API
process. Synchronous SDK calls and CPU-heavy prediction work are offloaded to
worker threads so they do not block the asyncio event loop; there is no
separate prediction process or application-managed prediction queue. The API
returns decisions and size suggestions; it does not place broker orders.

Start the application from the repository root:

```sh
uv run python src/trade_ai/app.py
```

The server binds to `0.0.0.0`. Browser origins are controlled by
`TRADE_AI_CHAT_ORIGINS`, defaulting to `http://localhost:3011` and
`http://127.0.0.1:3011`.

## Endpoints

| Method | Path | Description | Response |
| --- | --- | --- | --- |
| `GET` | `/` | Basic API availability check (`SERVER/SERVER_STATUS`). | `{"intent":"SERVER","data":{"eventType":"SERVER_STATUS","data":{"service":"trade-ai","status":"online"},"error":null}}` |
| `GET` | `/api/event/test` | Validate event query parameters, fetch/cache recent OHLCV history, and return the normalized ticker with history as a JSON array of row objects (`SERVER/SERVER_STATUS`). | `{"intent":"SERVER","data":{"eventType":"SERVER_STATUS","data":{"ticker":"WIPRO","history":[{"timestamp":"2026-10-28T09:15:00","open":1510.5,"high":1512.0,"low":1509.0,"close":1511.5,"volume":1200}]},"error":null}}` |
| `POST` | `/api/brain/predict` | By default, run all four specialists and PPO; `target_model` can instead select one base specialist (`MODEL/PREDICTION`). | Meta-ensemble responses include signal, confidence, position size, bracket and target/stop distances, all specialist probabilities, side means, and action index. Specialist-only responses include its probabilities and predicted class; signal, position, and PPO/bracket fields are null. |
| `POST` | `/api/openalgo/history` | Fetch broker history through the OpenAlgo Python client (`OPENALGO/HISTORY`). | `{"intent":"OPENALGO","data":{"eventType":"HISTORY","data":{"symbol":"WIPRO","exchange":"BSE","interval":"1m","candles":[{"timestamp":1790842500,"open":1510.5,"high":1512.0,"low":1509.0,"close":1511.5,"volume":1200,"oi":null}]},"error":null}}` |
| `GET` | `/api/openalgo/analyser-status` | Fetch OpenAlgo analyser status (`OPENALGO/ANALYSER_STATUS`). | `{"intent":"OPENALGO","data":{"eventType":"ANALYSER_STATUS","data":{"available":true,"mode":"live","analyze_mode":true,"message":"OpenAlgo analyser is online."},"error":null}}` |

All endpoint responses use `UniversalServerResponse[intent, eventType,
responseDataType, errorDataType]`. Successful responses populate
`data.data`; errors populate `data.error` and leave `data.data` null. Endpoint
validation errors use the same envelope and event type with HTTP 422. The
backend event values are defined in `model/modelTypes.py`,
`openalgo/openalgoTypes.py`, and `universalTypes.py`; keep the chat's response
discriminators aligned with them until a shared event-types module is added.

`serverTypes.py` owns the generic response envelope. Keep module-related types
in their module's `*Types.py` file and import them into routes as needed;
OpenAlgo candle data and the validated `OhlcvCandle` model for caller-supplied
prediction candles live in `openalgo/openalgoTypes.py`. Route-specific
response contracts are defined with their routes. Errors are strings for the
current endpoints.

## Async and non-blocking endpoint rules

- Every FastAPI endpoint handler must be declared with `async def`.
- Await asynchronous I/O. Do not call synchronous network, database, filesystem,
  or broker SDK operations directly from an endpoint or an async helper; run
  blocking calls with `await asyncio.to_thread(...)` or an appropriate
  asynchronous client.
- Offload CPU-heavy feature engineering, model inference, large DataFrame
  transformations, and large response conversions from the event loop with
  `asyncio.to_thread(...)` or a background job. Keep only small, bounded work
  directly in the handler.
- Do not use `asyncio.run()` inside an ASGI request handler. It is for bridging
  from synchronous training/CLI code, not for endpoint code.
- Keep request and response sizes bounded where practical. Consider worker-pool
  saturation and large response serialization when reviewing server performance.

## Server review check

When the user asks to `evaluate server`, `check server`, or asks for a server
module review, audit all endpoint handlers and their downstream calls. Report
handlers that are not async, synchronous I/O or CPU-heavy work that can block
the event loop, large response or worker-pool performance risks, and any other
changes that may affect application or server performance. Include the affected
route or module and a concise finding for each issue.

The current routes follow this policy: history and analyser SDK calls run in
worker threads; prediction preprocessing, inference, caller-candle conversion,
and history candle conversion plus JSON encoding are offloaded; the health
check only performs small in-memory response construction. Error handlers
encode the typed response envelope into JSON-compatible values before building
`JSONResponse` objects.

### Event test

`GET /api/event/test` accepts `ticker` and an optional `exchange` query
parameter. `ticker` must contain 1–32 non-blank characters and is returned
trimmed and uppercased. `exchange` accepts `NSE` or `BSE` and defaults to
`NSE`. The route awaits the in-memory candle store, which fetches 30 trading
sessions of `1m` history through the previous trading date when the cache does
not cover that range and stores/returns a pandas DataFrame.
History is included as a JSON array of candle row objects. For example:
`/api/event/test?ticker=WIPRO&exchange=BSE`.

### OpenAlgo history

The history request body contains `symbol`, `exchange`, `interval`, `start_date`,
and `end_date`. Dates use `YYYY-MM-DD`; exchange is `NSE` or `BSE`; the
configured broker must support the requested interval. Accepted intervals are
`1m`, `2m`, `3m`, `4m`, `5m`, `10m`, `15m`, `30m`, `1h`, `1d`, `D`, and `W`.
The UI's `1d` value is translated to OpenAlgo's `D` interval.

Successful `data.data` is an object containing the requested `symbol`,
`exchange`, and `interval`, plus a `candles` array. Each candle has a
Unix-second `timestamp`, numeric OHLCV fields, and optional `oi`. OpenAlgo
failures are returned in `data.error` as a string.

Send the prediction endpoint a structured body. Portfolio fields are optional;
their defaults describe a flat position:

```json
{
  "ticker": "WIPRO",
  "exchange": "BSE",
  "target_model": "meta_ensemble",
  "days_back": 30,
  "interval": "1m",
  "portfolio_position": 0.0,
  "unrealized_return": 0.0,
  "bars_in_position": 0,
  "barrier_multiplier": 1.0
}
```

`target_model` is optional and defaults to `meta_ensemble`. It accepts
`long_lgb`, `long_gru`, `short_lgb`, or `short_gru` to query one loaded
specialist without requiring the other base models or the PPO policy.
The model's nested probability array is ordered `[-1, 0, 1]` (stop, timeout,
take profit relative to that specialist's direction), and its nested
`predicted_class` is the highest-probability class. A specialist
classifier does not choose a portfolio position, so `signal`, `position_size`,
and PPO/bracket fields are null in this mode; portfolio context fields are not
used. The fixed `breakdown` schema has root-level `target_model` and a null
`signal`; the selected direction
contains its model object (for example `breakdown.long.lgb`) with
`probabilities` and `predicted_class`. The opposite direction, other model,
and ensemble values are null. There is no root probability array; classifier
probabilities live only under their direction and model. `confidence` is that
specialist's take-profit probability. A missing selected model returns HTTP 503.

`barrier_multiplier` is optional and defaults to `1.0`. Pass the active value
(`0.75`, `1.0`, or `1.25`) when `portfolio_position` describes a position that
is already open; PPO preserves that position's bracket width. For a new
position, PPO chooses its own multiplier at entry.

`exchange` accepts `NSE` or `BSE`. The current inference pipeline accepts `1m`
candles only. To provide candles directly, include a `candles` array of objects
with `timestamp`, `open`, `high`, `low`, `close`, and `volume`; timestamps may
be ISO date/time strings or Unix timestamps in seconds or milliseconds.
Offset-aware values are converted to Asia/Kolkata market time, naive date/time
strings are treated as India local time, and Unix timestamps are interpreted as
UTC instants before conversion.
Values are validated, and arrays must contain 1–25,000 candles. `days_back` is limited to 90 trading sessions;
values below 30 use the 30-session minimum. If `candles` is omitted, the API
fetches history using `days_back`. The latest supplied candle date anchors the prediction.
At least 30 prior trading sessions ending on the previous exchange trading
date are retrieved from cached `dataStore` history or fetched from OpenAlgo.
Coverage checks require every expected exchange trading date. Gaps between
observed one-minute bars within a session are logged; they may reflect missing
source data or periods with no trades and are not synthesized.
The supplied candles are merged chronologically with this history before
inference, but remain transient and are not saved in the in-memory history
cache. The cache contains only the requested prior-session history, dropping
unrelated dates retained from earlier requests.
If `candles` is omitted, the route fetches the same prior history before today,
fetches today's candles, and merges those transient candles for inference.
`days_back` can request a longer session lookback; values below 30 still use 30
sessions. `portfolio_position` accepts a multiplier from -1.5 to +1.5;
`unrealized_return` is fractional from -1 to +1; `bars_in_position` is an
integer from 0 through 60. In the fixed `breakdown` schema, `long` and `short`
each contain `lgb` and `gru` objects with `probabilities` and
`predicted_class`, plus `ensemble_probabilities`. Meta-ensemble responses
populate both directions. The root contains the combined PPO output:
`signal`, `meta_action`, `position_size`, `barrier_multiplier`, and
`take_profit_pct`/`stop_loss_pct` distances, which are null for a flat action.
At entry, the policy selects direction, size, and a
bracket multiplier of `0.75`, `1.0`, or `1.25`. It scales the baseline 1%
target and 0.5% stop by the same multiplier, preserving their 2:1 ratio. The
response reports positive target/stop distance magnitudes; signal direction
determines which side of entry each distance lies on. ATR is relative
volatility context for PPO and does not set these distances.
Class array indices are ordered `[-1, 0, 1]` (stop, timeout, take profit
relative to that model's side). Fields that do not apply to the selected mode
are present as null, keeping the response shape stable. `signal` is `-1` for
short, `0` for flat, and `1` for long.
`confidence` is the chosen side's average take-profit probability; it is not a
calibrated guarantee. A flat action reports zero confidence. Success data also
includes `prediction_timestamp`, the latest candle used for inference. The
default meta-ensemble requires all four compatible base checkpoints and a PPO
policy whose schema and recorded base checkpoint hashes match the loaded
models; missing or stale artifacts return HTTP 503. A specialist-only request
requires only its selected compatible base checkpoint. A data retrieval or
inference failure is returned in `data.error`.
Successful prediction logs include this probability breakdown when inference
completes and immediately before the response is returned.
Failures also use HTTP 422 for invalid or insufficient input, 502 for upstream
history failures, 503 when models are unavailable, and 500 for unexpected
execution errors.

## Maintenance

Whenever an endpoint is added or modified, update this file.
