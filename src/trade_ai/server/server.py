"""FastAPI server entry point."""

import os
from threading import Lock, Thread

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trade_ai.server.chat_route import router as chat_router
from trade_ai.server.openalgo.openalgo_route import router as analyser_status_router

server_app = FastAPI()

# The Astro chat runs on a separate origin. Override this comma-separated list
# for deployments; local development defaults to the Astro port.
_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "TRADE_AI_CHAT_ORIGINS",
        "http://localhost:4000,http://127.0.0.1:4000",
    ).split(",")
    if origin.strip()
]
server_app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
_server_thread: Thread | None = None
_server_lock = Lock()


server_app.include_router(chat_router, prefix="/api")
server_app.include_router(analyser_status_router, prefix="/api/openalgo")


@server_app.get("/")
async def hello_world() -> dict[str, str]:
    """Return a simple health check response."""
    return {"message": "This is 'trade-ai' API route"}


def start_server() -> Thread:
    """Start Uvicorn on port 3010 in a background thread if not already running."""
    global _server_thread

    with _server_lock:
        if _server_thread is not None and _server_thread.is_alive():
            return _server_thread

        server = uvicorn.Server(uvicorn.Config(server_app, host="0.0.0.0", port=3010))
        _server_thread = Thread(target=server.run, name="trade-ai-api", daemon=True)
        _server_thread.start()
        return _server_thread
