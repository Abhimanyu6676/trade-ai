from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

TYPE_OPENALGO_INTENT = Literal["OPENALGO"]
TYPE_OPENALGO_INPUT_EVENTS = Literal["GET_HISTORY", "GET_ANALYSER_STATUS"]
TYPE_OPENALGO_OUTPUTS_EVENTS = Literal["HISTORY", "ANALYSER_STATUS"]
TYPE_OPENALGO_EVENTS = TYPE_OPENALGO_INPUT_EVENTS | TYPE_OPENALGO_OUTPUTS_EVENTS

TYPE_OPENALGO_EXCHANGE = Literal["NSE", "BSE"]

TYPE_OPENALGO_INTERVALS = Literal[
    "1m", "2m", "3m", "4m", "5m", "10m", "15m", "30m", "1h", "1d", "D", "W"
]


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


@dataclass
class OHLCVCandleDataType:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float | None = None


@dataclass
class get_analyser_status_response:
    status: Literal["success", "error"]
    analyze_mode: bool | None = None
    mode: Literal["analyze", "live"] | None = None
    errMsg: str | None = None
