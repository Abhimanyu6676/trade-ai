"""Event API routes."""

from dataclasses import dataclass
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, field_validator

from trade_ai.logger import logger
from trade_ai.model.dataStore import dataStore
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
    history: list[dict[str, object]] | None


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
        exchange = request.exchange

        logger.info(f"This is new test route requesting data for {ticker}")
        history = await dataStore.get_candles(ticker, exchange)

        return EventApiResponse(
            intent="SERVER",
            data=serverDataType(
                eventType="SERVER_STATUS",
                data=EventResponseDataType(
                    ticker=ticker,
                    history=(
                        cast(list[dict[str, object]], history.to_dict(orient="records"))
                        if history is not None
                        else None
                    ),
                ),
            ),
        )
    except Exception as error:
        return EventApiResponse(
            intent="SERVER",
            data=serverDataType(eventType="SERVER_STATUS", error=str(error)),
        )
