"""Structured prediction endpoint; inference is handled by the main app thread."""

import math
from datetime import datetime
from typing import Literal, cast

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator, model_validator

from trade_ai.model.data_engine import fetch_and_clean_candles, predict_from_candles
from trade_ai.server.serverTypes import (
    PredictionErrorDataType,
    PredictionResponseDataType,
    UniversalServerResponse,
    serverDataType,
)

router = APIRouter()


class OhlcvCandle(BaseModel):
    timestamp: str | int | float
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: str | float) -> str | int | float:
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(
                    "timestamp must be an ISO date/time or Unix timestamp"
                ) from error
        elif not math.isfinite(value) or value <= 0:
            raise ValueError("timestamp must be a positive Unix timestamp")
        return value

    @model_validator(mode="after")
    def validate_candle(self) -> OhlcvCandle:
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("OHLCV values must be finite numbers")
        if self.volume < 0 or self.high < max(self.open, self.close, self.low):
            raise ValueError("Candle high or volume is invalid")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("Candle low is invalid")
        return self


class PredictionRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    exchange: Literal["NSE", "BSE"] = "NSE"
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


PredictionResponse = PredictionResponseDataType
PredictionServerResponse = UniversalServerResponse[
    Literal["MODEL"],
    Literal["PREDICTION"],
    PredictionResponseDataType,
    PredictionErrorDataType,
]


@router.post("/predict", response_model=PredictionServerResponse)
async def predict(request: PredictionRequest) -> PredictionServerResponse:
    """Run inference on supplied candles or fetch history from OpenAlgo."""
    try:
        ticker = request.ticker.strip().upper()
        if request.candles is not None:
            candles, signal, confidence = predict_from_candles(
                ticker,
                [candle.model_dump() for candle in request.candles],
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
                data=PredictionResponse(
                    action="PREDICT",
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
