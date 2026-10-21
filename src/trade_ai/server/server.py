"""FastAPI server entry point."""

from threading import Lock, Thread

import uvicorn
from fastapi import FastAPI

from trade_ai.server.chat_route import router as chat_router

server_app = FastAPI()
_server_thread: Thread | None = None
_server_lock = Lock()


server_app.include_router(chat_router, prefix="/api")


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
