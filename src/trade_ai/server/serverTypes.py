from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

from trade_ai.model.modelTypes import TYPE_MODEL_OUTPUTS_EVENTS
from trade_ai.openalgo.openalgoTypes import TYPE_OPENALGO_OUTPUTS_EVENTS
from trade_ai.universalTypes import (
    TYPE_SERVER_OUTPUT_EVENTS,
    UNIVERSAL_INTENTS,
)

SERVER_RESPONSE_EVENTS = (
    TYPE_MODEL_OUTPUTS_EVENTS | TYPE_OPENALGO_OUTPUTS_EVENTS | TYPE_SERVER_OUTPUT_EVENTS
)

IntentT = TypeVar("IntentT", bound=UNIVERSAL_INTENTS)
EventT = TypeVar("EventT", bound=SERVER_RESPONSE_EVENTS)
ResponseDataT = TypeVar("ResponseDataT")
ErrorDataT = TypeVar("ErrorDataT")


# Prediction endpoint response and error types
class PredictionResponseDataType(BaseModel):
    action: Literal["PREDICT"]
    ticker: str
    exchange: Literal["NSE", "BSE"]
    days_back: int | None
    interval: Literal["1m"]
    signal: Literal[-1, 0, 1]
    confidence: float
    candles_processed: int


PredictionErrorDataType = str


# OpenAlgo history endpoint response and error types
class HistoryCandleDataType(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float | None = None


HistoryResponseDataType = list[HistoryCandleDataType]
HistoryErrorDataType = str


# OpenAlgo analyser status endpoint response and error types
class AnalyserStatusResponseDataType(BaseModel):
    available: bool
    mode: str | None = None
    analyze_mode: bool | None = None
    message: str


AnalyserStatusErrorDataType = str


# Shared server error type
class serverErrorType(BaseModel):
    message: str
    error: str | None = None


# Server status endpoint response and error types
class serverStatusResponseDataType(BaseModel):
    service: str
    status: Literal["online"]


ServerStatusErrorDataType = str


# Generic UniversalServerResponse envelope types
class serverDataType(BaseModel, Generic[EventT, ResponseDataT, ErrorDataT]):
    eventType: EventT
    data: ResponseDataT | None = None
    error: ErrorDataT | None = None


class UniversalServerResponse(
    BaseModel, Generic[IntentT, EventT, ResponseDataT, ErrorDataT]
):
    intent: IntentT
    data: serverDataType[EventT, ResponseDataT, ErrorDataT]
