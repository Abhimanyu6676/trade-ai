# Server Endpoint API

## Server

`app.py` loads the prediction models, then runs Uvicorn on port `3010` in the
main process. Prediction requests fetch candles and run inference in the API
process; there is no background server thread or prediction queue.

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
| `POST` | `/api/brain/predict` | Run prediction using supplied candles or fetch history (`MODEL/PREDICTION`). | `{"intent":"MODEL","data":{"eventType":"PREDICTION","data":{"action":"PREDICT","ticker":"WIPRO","exchange":"BSE","days_back":null,"interval":"1m","signal":1,"confidence":0.82,"candles_processed":420},"error":null}}` |
| `POST` | `/api/openalgo/history` | Fetch broker history through the OpenAlgo Python client (`OPENALGO/HISTORY`). | `{"intent":"OPENALGO","data":{"eventType":"HISTORY","data":[{"timestamp":1790842500,"open":1510.5,"high":1512.0,"low":1509.0,"close":1511.5,"volume":1200,"oi":null}],"error":null}}` |
| `GET` | `/api/openalgo/analyser-status` | Fetch OpenAlgo analyser status (`OPENALGO/ANALYSER_STATUS`). | `{"intent":"OPENALGO","data":{"eventType":"ANALYSER_STATUS","data":{"available":true,"mode":"live","analyze_mode":true,"message":"OpenAlgo analyser is online."},"error":null}}` |

All endpoint responses use `UniversalServerResponse[intent, eventType,
responseDataType, errorDataType]`. Successful responses populate
`data.data`; errors populate `data.error` and leave `data.data` null. Endpoint
validation errors use the same envelope and event type with HTTP 422. The
backend event values are defined in `model/modelTypes.py`,
`openalgo/openalgoTypes.py`, and `universalTypes.py`; keep the chat's response
discriminators aligned with them until a shared event-types module is added.

Endpoint response payload and error types, including history candle types,
live in `serverTypes.py`, separate from model and OpenAlgo internal types.
Errors are strings for the current endpoints. Keep the generic envelope and
these API response contracts in `serverTypes.py` rather than duplicating
them in each route.

### OpenAlgo history

The history request body contains `symbol`, `exchange`, `interval`, `start_date`,
and `end_date`. Dates use `YYYY-MM-DD`; exchange is `NSE` or `BSE`; the
configured broker must support the requested interval. Accepted intervals are
`1m`, `2m`, `3m`, `4m`, `5m`, `10m`, `15m`, `30m`, `1h`, `1d`, `D`, and `W`.
The UI's `1d` value is translated to OpenAlgo's `D` interval.

Successful `data.data` is an array of candles with Unix-second `timestamp`,
numeric OHLCV fields, and optional `oi`. OpenAlgo failures are returned in
`data.error` as a string.

Send the prediction endpoint a structured body:

```json
{
  "ticker": "WIPRO",
  "exchange": "BSE",
  "days_back": 30,
  "interval": "1m"
}
```

`exchange` accepts `NSE` or `BSE`. The current inference pipeline accepts `1m`
candles only. To provide candles directly, include a `candles` array of objects
with `timestamp`, `open`, `high`, `low`, `close`, and `volume`; timestamps may
be ISO date/time strings or Unix timestamps. Values are validated, and arrays
are limited to 25,000 candles. If `candles` is omitted, the API fetches history
using `days_back`. A data retrieval or inference failure is returned in
`data.error`.

## Maintenance

Whenever an endpoint is added or modified, update this file.
