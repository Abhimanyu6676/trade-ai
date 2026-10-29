"""Structured prediction endpoint with blocking work offloaded to worker threads."""

import asyncio
from dataclasses import dataclass
from typing import Literal, cast

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field, field_validator

from trade_ai.model.prediction_engine import (
    MAX_PREDICTION_LOOKBACK_SESSIONS,
    PredictionError,
    predict_with_history,
)
from trade_ai.openalgo.openalgoTypes import TYPE_OPENALGO_EXCHANGE, OhlcvCandle
from trade_ai.server.serverTypes import (
    UniversalServerResponse,
    serverDataType,
)

router = APIRouter()


class PredictionRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    exchange: TYPE_OPENALGO_EXCHANGE = "NSE"
    days_back: int = Field(
        default=7, ge=1, le=MAX_PREDICTION_LOOKBACK_SESSIONS
    )
    interval: Literal["1m"] = "1m"
    candles: list[OhlcvCandle] | None = Field(
        default=None, min_length=1, max_length=25000
    )

    @field_validator("ticker")
    @classmethod
    def ticker_must_not_be_blank(cls, value: str) -> str:
        ticker = value.strip()
        if not ticker:
            raise ValueError("ticker cannot be blank")
        return ticker


# Prediction endpoint response and error types
@dataclass
class PredictionResponseDataType:
    ticker: str
    exchange: Literal["NSE", "BSE"]
    days_back: int | None
    interval: Literal["1m"]
    signal: Literal[-1, 0, 1]
    confidence: float
    candles_processed: int


PredictionServerResponse = UniversalServerResponse[
    Literal["MODEL"],
    Literal["PREDICTION"],
    PredictionResponseDataType,
    str,
]


@router.post("/predict", response_model=PredictionServerResponse)
async def predict(
    request: PredictionRequest, response: Response
) -> PredictionServerResponse:
    """Run inference with 30 prior trading sessions and target-day candles."""
    try:
        ticker = request.ticker.strip().upper()
        candles_input = request.candles
        supplied = (
            await asyncio.to_thread(
                lambda candles: [candle.model_dump() for candle in candles],
                candles_input,
            )
            if candles_input is not None
            else None
        )
        candles, signal, confidence = await predict_with_history(
            ticker,
            request.exchange,
            supplied,
            sessions=request.days_back,
        )
        return PredictionServerResponse(
            intent="MODEL",
            data=serverDataType(
                eventType="PREDICTION",
                data=PredictionResponseDataType(
                    ticker=ticker,
                    exchange=request.exchange,
                    days_back=(
                        None if request.candles is not None else request.days_back
                    ),
                    interval=request.interval,
                    signal=cast(Literal[-1, 0, 1], int(signal)),
                    confidence=float(confidence),
                    candles_processed=len(candles),
                ),
            ),
        )
    except PredictionError as error:
        response.status_code = error.status_code
        return PredictionServerResponse(
            intent="MODEL",
            data=serverDataType(eventType="PREDICTION", error=str(error)),
        )
    except Exception as error:
        response.status_code = 500
        return PredictionServerResponse(
            intent="MODEL",
            data=serverDataType(eventType="PREDICTION", error=str(error)),
        )
