"""Event API routes."""

from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, field_validator

from trade_ai.logger import logger
from trade_ai.openalgo.openalgoTypes import TYPE_OPENALGO_EXCHANGE
from trade_ai.server.serverTypes import (
    UniversalServerResponse,
    serverDataType,
)

router = APIRouter()


class EventRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    exchange: TYPE_OPENALGO_EXCHANGE = "NSE"

    @field_validator("ticker")
    @classmethod
    def ticker_must_not_be_blank(cls, value: str) -> str:
        ticker = value.strip()
        if not ticker:
            raise ValueError("ticker cannot be blank")
        return ticker


# Event endpoint response type
@dataclass
class EventResponseDataType:
    ticker: str


EventApiResponse = UniversalServerResponse[
    Literal["SERVER"],
    Literal["SERVER_STATUS"],
    EventResponseDataType,
    str,
]


@router.get("/test")
async def predict(request: Annotated[EventRequest, Query()]) -> EventApiResponse:
    """Return the normalized ticker from the event test request."""
    try:
        ticker = request.ticker.strip().upper()

        logger.info(f"This is new test route requesting data for {ticker}")
        return EventApiResponse(
            intent="SERVER",
            data=serverDataType(
                eventType="SERVER_STATUS",
                data=EventResponseDataType(ticker=ticker),
            ),
        )
    except Exception as error:
        return EventApiResponse(
            intent="SERVER",
            data=serverDataType(eventType="SERVER_STATUS", error=str(error)),
        )
