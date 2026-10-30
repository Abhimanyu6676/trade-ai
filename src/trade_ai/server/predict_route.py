"""Structured prediction endpoint with blocking work offloaded to worker threads."""

import asyncio
from dataclasses import dataclass
from typing import Literal, cast

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field, field_validator

from trade_ai.config.trading_config import BARRIER_MULTIPLIERS
from trade_ai.logger import logger
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
    portfolio_position: float = Field(default=0.0, ge=-1.5, le=1.5)
    unrealized_return: float = Field(default=0.0, ge=-1.0, le=1.0)
    bars_in_position: int = Field(default=0, ge=0, le=60)
    barrier_multiplier: float = Field(default=1.0)

    @field_validator("ticker")
    @classmethod
    def ticker_must_not_be_blank(cls, value: str) -> str:
        ticker = value.strip()
        if not ticker:
            raise ValueError("ticker cannot be blank")
        return ticker

    @field_validator("barrier_multiplier")
    @classmethod
    def barrier_multiplier_must_be_supported(cls, value: float) -> float:
        if value not in BARRIER_MULTIPLIERS:
            raise ValueError(
                f"barrier_multiplier must be one of {BARRIER_MULTIPLIERS}"
            )
        return value

# Prediction endpoint response and error types
@dataclass
class PredictionBreakdownDataType:
    lgb_probabilities: list[float]
    gru_probabilities: list[float]
    ensemble_probabilities: list[float]
    long_lgb_probabilities: list[float]
    long_gru_probabilities: list[float]
    short_lgb_probabilities: list[float]
    short_gru_probabilities: list[float]
    long_ensemble_probabilities: list[float]
    short_ensemble_probabilities: list[float]
    meta_action: int
    position_size: float
    barrier_multiplier: float | None
    take_profit_pct: float | None
    stop_loss_pct: float | None


@dataclass
class PredictionResponseDataType:
    ticker: str
    exchange: Literal["NSE", "BSE"]
    days_back: int | None
    interval: Literal["1m"]
    signal: Literal[-1, 0, 1]
    confidence: float
    position_size: float
    candles_processed: int
    prediction_timestamp: str
    breakdown: PredictionBreakdownDataType


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
        candles, signal, confidence, breakdown = await predict_with_history(
            ticker,
            request.exchange,
            supplied,
            sessions=request.days_back,
            portfolio_position=float(request.portfolio_position),
            unrealized_return=request.unrealized_return,
            bars_in_position=request.bars_in_position,
            barrier_multiplier=request.barrier_multiplier,
        )
        logger.info(
            "Sending prediction response for %s:%s: signal=%s, confidence=%.4f, "
            "candles_processed=%s, breakdown=%s",
            ticker,
            request.exchange,
            signal,
            confidence,
            len(candles),
            breakdown,
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
                    position_size=float(breakdown["position_size"]),
                    candles_processed=len(candles),
                    prediction_timestamp=str(candles["timestamp"].iloc[-1]),
                    breakdown=PredictionBreakdownDataType(
                        lgb_probabilities=breakdown["lgb_probabilities"],
                        gru_probabilities=breakdown["gru_probabilities"],
                        ensemble_probabilities=breakdown["ensemble_probabilities"],
                        long_lgb_probabilities=breakdown[
                            "long_lgb_probabilities"
                        ],
                        long_gru_probabilities=breakdown[
                            "long_gru_probabilities"
                        ],
                        short_lgb_probabilities=breakdown[
                            "short_lgb_probabilities"
                        ],
                        short_gru_probabilities=breakdown[
                            "short_gru_probabilities"
                        ],
                        long_ensemble_probabilities=breakdown[
                            "long_ensemble_probabilities"
                        ],
                        short_ensemble_probabilities=breakdown[
                            "short_ensemble_probabilities"
                        ],
                        meta_action=int(breakdown["meta_action"]),
                        position_size=float(breakdown["position_size"]),
                        barrier_multiplier=breakdown["barrier_multiplier"],
                        take_profit_pct=breakdown["take_profit_pct"],
                        stop_loss_pct=breakdown["stop_loss_pct"],
                    ),
                ),
            ),
        )
    except PredictionError as error:
        response.status_code = error.status_code
        logger.warning(
            "Sending prediction error response for %s:%s (HTTP %s): %s",
            request.ticker.strip().upper(),
            request.exchange,
            error.status_code,
            error,
        )
        return PredictionServerResponse(
            intent="MODEL",
            data=serverDataType(eventType="PREDICTION", error=str(error)),
        )
    except Exception as error:
        response.status_code = 500
        logger.exception(
            "Sending unexpected prediction error response for %s:%s",
            request.ticker.strip().upper(),
            request.exchange,
        )
        return PredictionServerResponse(
            intent="MODEL",
            data=serverDataType(eventType="PREDICTION", error=str(error)),
        )
