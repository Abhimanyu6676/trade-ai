"""OpenAlgo history and analyser status endpoints for browser clients."""

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, model_validator
from starlette.responses import JSONResponse

from trade_ai.logger import logger
from trade_ai.openalgo.helper import convert_OHLCV_dfToArray
from trade_ai.openalgo.openalgo_client import openalgo
from trade_ai.openalgo.openalgoTypes import (
    TYPE_OPENALGO_EXCHANGE,
    TYPE_OPENALGO_INTERVALS,
    OHLCVCandleDataType,
)
from trade_ai.server.serverTypes import (
    UniversalServerResponse,
    serverDataType,
)

router = APIRouter()


# ##     ## ####  ######  ########  #######  ########  ##    ##    ########   #######  ##     ## ######## ########
# ##     ##  ##  ##    ##    ##    ##     ## ##     ##  ##  ##     ##     ## ##     ## ##     ##    ##    ##
# ##     ##  ##  ##          ##    ##     ## ##     ##   ####      ##     ## ##     ## ##     ##    ##    ##
# #########  ##   ######     ##    ##     ## ########     ##       ########  ##     ## ##     ##    ##    ######
# ##     ##  ##        ##    ##    ##     ## ##   ##      ##       ##   ##   ##     ## ##     ##    ##    ##
# ##     ##  ##  ##    ##    ##    ##     ## ##    ##     ##       ##    ##  ##     ## ##     ##    ##    ##
# ##     ## ####  ######     ##     #######  ##     ##    ##       ##     ##  #######   #######     ##    ########


class HistoryRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    exchange: TYPE_OPENALGO_EXCHANGE
    interval: TYPE_OPENALGO_INTERVALS
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_date_range(self) -> HistoryRequest:
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self


@dataclass
class HistoryResponseDataType:
    symbol: str
    exchange: TYPE_OPENALGO_EXCHANGE
    interval: TYPE_OPENALGO_INTERVALS
    candles: list[OHLCVCandleDataType]


HistoryServerResponse = UniversalServerResponse[
    Literal["OPENALGO"],
    Literal["HISTORY"],
    HistoryResponseDataType,
    str,
]


def _build_history_json_response(
    request: HistoryRequest, history: pd.DataFrame
) -> JSONResponse:
    """Convert and encode a potentially large history response off the loop."""
    response = HistoryServerResponse(
        intent="OPENALGO",
        data=serverDataType(
            eventType="HISTORY",
            data=HistoryResponseDataType(
                symbol=request.symbol,
                exchange=request.exchange,
                interval=request.interval,
                candles=convert_OHLCV_dfToArray(history),
            ),
        ),
    )
    return JSONResponse(content=jsonable_encoder(response))


@router.post("/history", response_model=HistoryServerResponse)
async def get_history(request: HistoryRequest) -> HistoryServerResponse | JSONResponse:
    """Fetch historical candles from OpenAlgo in its documented candle shape."""
    logger.info("History requested on route '/api/openalgo/history'")
    try:
        history = await openalgo.get_history(
            symbol=request.symbol.strip().upper(),
            exchange=request.exchange,
            interval="D" if request.interval == "1d" else request.interval,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        if isinstance(history, pd.DataFrame):
            return await asyncio.to_thread(
                _build_history_json_response, request, history
            )
        return HistoryServerResponse(
            intent="OPENALGO",
            data=serverDataType(
                eventType="HISTORY",
                error=history.get("message", "OpenAlgo history request failed."),
            ),
        )
    except Exception as error:
        logger.exception("Error fetching OpenAlgo history")
        return HistoryServerResponse(
            intent="OPENALGO",
            data=serverDataType(eventType="HISTORY", error=str(error)),
        )


#    ###    ##    ##    ###    ##       ##    ##  ######  ######## ########     ########   #######  ##     ## ######## ########
#   ## ##   ###   ##   ## ##   ##        ##  ##  ##    ## ##       ##     ##    ##     ## ##     ## ##     ##    ##    ##
#  ##   ##  ####  ##  ##   ##  ##         ####   ##       ##       ##     ##    ##     ## ##     ## ##     ##    ##    ##
# ##     ## ## ## ## ##     ## ##          ##     ######  ######   ########     ########  ##     ## ##     ##    ##    ######
# ######### ##  #### ######### ##          ##          ## ##       ##   ##      ##   ##   ##     ## ##     ##    ##    ##
# ##     ## ##   ### ##     ## ##          ##    ##    ## ##       ##    ##     ##    ##  ##     ## ##     ##    ##    ##
# ##     ## ##    ## ##     ## ########    ##     ######  ######## ##     ##    ##     ##  #######   #######     ##    ########


# OpenAlgo analyser status endpoint response and error types
class AnalyserStatusResponseDataType(BaseModel):
    available: bool
    mode: str | None = None
    analyze_mode: bool | None = None
    message: str


AnalyserStatusServerResponse = UniversalServerResponse[
    Literal["OPENALGO"],
    Literal["ANALYSER_STATUS"],
    AnalyserStatusResponseDataType,
    str,
]


@router.get("/analyser-status", response_model=AnalyserStatusServerResponse)
async def get_analyser_status() -> AnalyserStatusServerResponse:
    """Return a small, credential-free summary of the OpenAlgo analyser state."""
    logger.info("Get Analyser Status request.")
    try:
        response = await openalgo.get_analyzer_status()
        if response.status == "success":
            return AnalyserStatusServerResponse(
                intent="OPENALGO",
                data=serverDataType(
                    eventType="ANALYSER_STATUS",
                    data=AnalyserStatusResponseDataType(
                        available=True,
                        analyze_mode=response.analyze_mode,
                        mode=response.mode,
                        message="OpenAlgo analyser is online.",
                    ),
                ),
            )
        return AnalyserStatusServerResponse(
            intent="OPENALGO",
            data=serverDataType(
                eventType="ANALYSER_STATUS",
                error=str(response.errMsg or "OpenAlgo analyser is unavailable."),
            ),
        )
    except Exception as error:
        return AnalyserStatusServerResponse(
            intent="OPENALGO",
            data=serverDataType(eventType="ANALYSER_STATUS", error=str(error)),
        )
