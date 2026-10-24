"""Structured prediction endpoint with blocking work offloaded to worker threads."""

import asyncio
from dataclasses import dataclass
from typing import Literal, cast

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from trade_ai.model.data_engine import fetch_and_clean_candles, predict_from_candles
from trade_ai.openalgo.openalgoTypes import TYPE_OPENALGO_EXCHANGE, OhlcvCandle
from trade_ai.server.serverTypes import (
    UniversalServerResponse,
    serverDataType,
)

router = APIRouter()


class PredictionRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    exchange: TYPE_OPENALGO_EXCHANGE = "NSE"
    days_back: int = Field(default=7, ge=1, le=3650)
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


def _predict_validated_candles(ticker: str, candles: list[OhlcvCandle]):
    """Convert and predict caller candles outside the event loop."""
    return predict_from_candles(
        ticker,
        [candle.model_dump() for candle in candles],
    )


@router.post("/predict", response_model=PredictionServerResponse)
async def predict(request: PredictionRequest) -> PredictionServerResponse:
    """Run inference on supplied candles or fetch history from OpenAlgo."""
    try:
        ticker = request.ticker.strip().upper()
        if request.candles is not None:
            candles, signal, confidence = await asyncio.to_thread(
                _predict_validated_candles,
                ticker,
                request.candles,
            )
            days_back = None
        else:
            candles, signal, confidence = await fetch_and_clean_candles(
                ticker,
                request.days_back,
                request.interval,
                request.exchange,
            )
            if candles is None:
                return PredictionServerResponse(
                    intent="MODEL",
                    data=serverDataType(eventType="PREDICTION", error=str(signal)),
                )
            days_back = request.days_back
        return PredictionServerResponse(
            intent="MODEL",
            data=serverDataType(
                eventType="PREDICTION",
                data=PredictionResponseDataType(
                    ticker=ticker,
                    exchange=request.exchange,
                    days_back=days_back,
                    interval=request.interval,
                    signal=cast(Literal[-1, 0, 1], int(signal)),
                    confidence=float(confidence),
                    candles_processed=len(candles),
                ),
            ),
        )
    except Exception as error:
        return PredictionServerResponse(
            intent="MODEL",
            data=serverDataType(eventType="PREDICTION", error=str(error)),
        )
