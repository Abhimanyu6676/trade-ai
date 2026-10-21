# Server Endpoint API

## Server

The FastAPI server is implemented in `server.py`. It runs Uvicorn on port
`3010` in a background thread when `app.py` starts. Launching the Gradio app
starts both services, so the API does not need a separate command.

Start it from the repository root with:

```sh
uv run python src/trade_ai/app.py
```

For API-only use, it can also be started directly with
`uv run python src/trade_ai/server/server.py`.

The server binds to `0.0.0.0`, so it can accept connections from outside the
local machine when the network permits it.

## Endpoints

| Method | Path | Description | Response |
| --- | --- | --- | --- |
| `GET` | `/` | Basic API availability check. | `{"message": "This is 'trade-ai' API route"}` |
| `POST` | `/api/chat` | Parse a message into a supported market action using the local parser. | `{"action": "market_analysis", "ticker": "SBIN", "days_back": 7}` |
| `GET` | `/api/chat?message=Check%20SBIN%20for%207%20days` | Parse a message supplied as a query parameter. | `{"action": "market_analysis", "ticker": "SBIN", "days_back": 7}` |

For POST, send `{"message": "Check SBIN for 7 days"}` in the request body.
For GET, URL-encode the message in the `message` query parameter. Both return
the parsed `action`, `ticker`, and `days_back` fields. The routes are defined
in `chat_route.py` and mounted under the `/api` prefix.

## Maintenance

Whenever an endpoint is added or modified, update this file.
