"""OpenAlgo history and analyser status endpoints for browser clients."""

import logging
import math
from datetime import date
from typing import Literal, cast

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from trade_ai.openalgo.openalgo_client import openalgo
from trade_ai.openalgo.openalgoTypes import (
    TYPE_OPENALGO_EXCHANGE,
    TYPE_OPENALGO_INTERVALS,
)
from trade_ai.server.serverTypes import (
    AnalyserStatusErrorDataType,
    AnalyserStatusResponseDataType,
    HistoryCandleDataType,
    HistoryErrorDataType,
    HistoryResponseDataType,
    UniversalServerResponse,
    serverDataType,
)

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

router = APIRouter()


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


HistoryCandle = HistoryCandleDataType
HistoryResponse = HistoryResponseDataType
HistoryServerResponse = UniversalServerResponse[
    Literal["OPENALGO"],
    Literal["HISTORY"],
    HistoryResponseDataType,
    HistoryErrorDataType,
]


def _history_candles(history: pd.DataFrame) -> list[HistoryCandle]:
    candles: list[HistoryCandle] = []
    for timestamp, row in history.iterrows():
        candle_timestamp = pd.Timestamp(cast(pd.Timestamp, timestamp))
        if candle_timestamp.tzinfo is None:
            # The SDK builds daily indexes from epoch seconds without attaching
            # a timezone; those values originate in UTC.
            candle_timestamp = candle_timestamp.tz_localize("UTC")
        candle_values = {
            name: float(row[name])
            for name in ("open", "high", "low", "close", "volume")
        }
        oi_value = row.get("oi")
        oi = (
            float(oi_value)
            if oi_value is not None and math.isfinite(float(oi_value))
            else None
        )
        candles.append(
            HistoryCandle(
                timestamp=int(candle_timestamp.timestamp()),
                **candle_values,
                oi=oi,
            )
        )
    return candles


@router.post("/history", response_model=HistoryServerResponse)
async def get_history(request: HistoryRequest) -> HistoryServerResponse:
    """Fetch historical candles from OpenAlgo in its documented candle shape."""
    try:
        history = await openalgo.get_history(
            symbol=request.symbol.strip().upper(),
            exchange=request.exchange,
            interval="D" if request.interval == "1d" else request.interval,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        if isinstance(history, pd.DataFrame):
            return HistoryServerResponse(
                intent="OPENALGO",
                data=serverDataType(
                    eventType="HISTORY", data=_history_candles(history)
                ),
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


AnalyserStatusResponse = AnalyserStatusResponseDataType


AnalyserStatusServerResponse = UniversalServerResponse[
    Literal["OPENALGO"],
    Literal["ANALYSER_STATUS"],
    AnalyserStatusResponseDataType,
    AnalyserStatusErrorDataType,
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
                    data=AnalyserStatusResponse(
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
