"""FastAPI server entry point."""

import os
from dataclasses import dataclass
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from trade_ai.server.event_route import (
    router as event_router,
)
from trade_ai.server.openalgo_route import (
    AnalyserStatusServerResponse,
    HistoryServerResponse,
)
from trade_ai.server.openalgo_route import (
    router as openalgo_router,
)
from trade_ai.server.predict_route import (
    PredictionServerResponse,
)
from trade_ai.server.predict_route import (
    router as predict_router,
)
from trade_ai.server.serverTypes import (
    UniversalServerResponse,
    serverDataType,
)

server_app = FastAPI()

# The Astro chat runs on a separate origin. Override this comma-separated list
# for deployments; local development defaults to the Astro port.
_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "TRADE_AI_CHAT_ORIGINS",
        "http://localhost:3011,http://127.0.0.1:3011",
    ).split(",")
    if origin.strip()
]
server_app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
server_app.include_router(predict_router, prefix="/api/brain")
server_app.include_router(openalgo_router, prefix="/api/openalgo")
server_app.include_router(event_router, prefix="/api/event")


def _error_response(path: str, message: str) -> Any:
    if path == "/api/brain/predict":
        response = PredictionServerResponse(
            intent="MODEL",
            data=serverDataType(eventType="PREDICTION", error=message),
        )
    elif path == "/api/openalgo/history":
        response = HistoryServerResponse(
            intent="OPENALGO",
            data=serverDataType(eventType="HISTORY", error=message),
        )
    elif path == "/api/openalgo/analyser-status":
        response = AnalyserStatusServerResponse(
            intent="OPENALGO",
            data=serverDataType(eventType="ANALYSER_STATUS", error=message),
        )
    else:
        response = ServerStatusResponse(
            intent="SERVER",
            data=serverDataType(eventType="SERVER_STATUS", error=message),
        )
    return response


@server_app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    message = "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(_error_response(request.url.path, message)),
    )


@server_app.exception_handler(HTTPException)
async def handle_http_error(request: Request, error: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=jsonable_encoder(_error_response(request.url.path, str(error.detail))),
        headers=error.headers,
    )


@server_app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(_error_response(request.url.path, str(error))),
    )


# Server status endpoint response and error types
@dataclass
class serverStatusResponseDataType:
    service: str
    status: Literal["online"]


ServerStatusResponse = UniversalServerResponse[
    Literal["SERVER"],
    Literal["SERVER_STATUS"],
    serverStatusResponseDataType,
    str,
]


@server_app.get("/", response_model=ServerStatusResponse)
async def hello_world() -> ServerStatusResponse:
    """Return a simple health check response."""
    return ServerStatusResponse(
        intent="SERVER",
        data=serverDataType(
            eventType="SERVER_STATUS",
            data=serverStatusResponseDataType(service="trade-ai", status="online"),
        ),
    )


def run_server() -> None:
    """Run Uvicorn in the current process until shutdown."""
    uvicorn.run(server_app, host="0.0.0.0", port=3010)
