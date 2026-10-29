"""OpenAlgo client lifecycle and API operations."""

import asyncio
import os
from datetime import date as Date
from datetime import timedelta
from typing import Any, Literal, TypedDict, cast

import pandas as pd
from openalgo import api

from trade_ai.environmentLoader import load_environment
from trade_ai.logger import logger
from trade_ai.util import lowercase_ohlcv_columns

from .openalgoTypes import (
    TYPE_OPENALGO_EXCHANGE,
    TYPE_OPENALGO_INTERVALS,
    OpenAlgoHolidaysResponse,
    get_analyser_status_response,
)

DEFAULT_OPENALGO_HOST = "https://openalgo-1.sternetsmart.com"


class OpenAlgoHistoryError(TypedDict, total=False):
    """Error dictionary returned by the OpenAlgo Python SDK history method."""

    status: Literal["error"]
    message: str
    error_type: str
    raw_data: list[dict[str, int | float]]


OpenAlgoHistoryResult = pd.DataFrame | OpenAlgoHistoryError


class OpenAlgo:
    """Lazily initialize and reuse the OpenAlgo SDK client."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def initiate(self) -> Any:
        """Create the SDK client once from the current environment settings."""
        if self._client is not None:
            return self._client

        load_environment()
        api_key = os.environ.get("OPENALGO_API_KEY")
        if not api_key:
            raise RuntimeError("OPENALGO_API_KEY is required to connect to OpenAlgo")

        host = os.environ.get("OPENALGO_HOST", DEFAULT_OPENALGO_HOST)
        self._client = api(api_key=api_key, host=host)
        return self._client

    async def get_history(
        self,
        *,
        symbol: str,
        exchange: TYPE_OPENALGO_EXCHANGE,
        interval: TYPE_OPENALGO_INTERVALS,
        start_date: Date,
        end_date: Date,
    ) -> OpenAlgoHistoryResult:
        """Fetch normalized OHLCV history or a typed OpenAlgo error response."""

        try:
            # The OpenAlgo Python SDK history method is synchronous, so run it
            # off the event loop. It returns a DataFrame or an error dictionary.
            history = await asyncio.to_thread(
                self.initiate().history,
                symbol=symbol.upper(),
                exchange=exchange,
                interval=interval,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                source="api",
            )

            if isinstance(history, pd.DataFrame):
                return lowercase_ohlcv_columns(history)
            return OpenAlgoHistoryError(
                status="error",
                message=str(history.get("message", "OpenAlgo history request failed.")),
                error_type=str(history.get("error_type", "openalgo_error")),
            )

        except Exception as e:
            logger.error("Error Fetching History Data: %s", e)
            return OpenAlgoHistoryError(
                status="error",
                message="Error Fetching History Data",
                error_type="client_error",
            )

    async def get_holidays(
        self, year: int | None = None
    ) -> OpenAlgoHolidaysResponse | None:
        """Return the holiday calendar for a year, or ``None`` on failure."""

        try:
            client = self.initiate()
            holidays = await asyncio.to_thread(client.holidays, year=year)
            if not isinstance(holidays, dict) or holidays.get("status") != "success":
                error_message = (
                    holidays.get("message", "OpenAlgo holidays request failed.")
                    if isinstance(holidays, dict)
                    else "OpenAlgo returned an invalid holidays response."
                )
                logger.error("Error Fetching Holidays: %s", error_message)
                return None

            logger.info(
                "Fetched %s market holidays for %s",
                len(holidays.get("data", [])),
                holidays.get("year", year),
            )
            return cast(OpenAlgoHolidaysResponse, holidays)
        except Exception as e:
            logger.error("Error Fetching Holidays: %s", e)
            return None

    async def getPreviousTradeDay(
        self,
        date: Date | None = None,
        days: int = 1,
        exchange: TYPE_OPENALGO_EXCHANGE = "NSE",
    ) -> Date | None:
        """Return the date ``days`` trading sessions before ``date`` (exclusive)."""

        if isinstance(days, bool) or not isinstance(days, int) or days < 1:
            raise ValueError("days must be a positive integer")

        reference_date = date or Date.today()
        candidate = reference_date - timedelta(days=1)
        calendars: dict[int, tuple[set[Date], set[Date]]] = {}
        trading_days_found = 0

        while trading_days_found < days:
            if candidate.year not in calendars:
                holiday_dates: set[Date] = set()
                open_dates: set[Date] = set()
                calendar = await self.get_holidays(candidate.year)
                if calendar is not None:
                    for holiday in calendar.get("data", []):
                        try:
                            holiday_date = Date.fromisoformat(holiday["date"])
                        except KeyError, TypeError, ValueError:
                            continue
                        if exchange in holiday.get("closed_exchanges", []):
                            holiday_dates.add(holiday_date)
                        if any(
                            isinstance(session, dict)
                            and session.get("exchange") == exchange
                            for session in holiday.get("open_exchanges", [])
                        ):
                            open_dates.add(holiday_date)
                calendars[candidate.year] = (holiday_dates, open_dates)

            holiday_dates, open_dates = calendars[candidate.year]
            is_closed = candidate in holiday_dates
            is_open = candidate in open_dates

            if is_open or (candidate.weekday() < 5 and not is_closed):
                trading_days_found += 1
                if trading_days_found == days:
                    logger.info(
                        "%s trading day(s) before %s for %s: %s",
                        days,
                        reference_date,
                        exchange,
                        candidate,
                    )
                    return candidate

            candidate -= timedelta(days=1)

    async def get_analyzer_status(self) -> get_analyser_status_response:
        """Return the OpenAlgo analyzer status response."""

        try:
            # The OpenAlgo SDK's analyzerstatus() is synchronous and returns a dict.
            client = self.initiate()
            statusRes = await asyncio.to_thread(client.analyzerstatus)
            if statusRes.get("status") == "success":
                data = statusRes.get("data", {})
                if not isinstance(data, dict):
                    data = {}
                mode = data.get("mode", None)
                analyze_mode = data.get("analyze_mode", None)
                return get_analyser_status_response(
                    status="success", analyze_mode=analyze_mode, mode=mode
                )
            return get_analyser_status_response(
                status="error",
                errMsg="OpenAlgo responded, but status check was unsuccessful.",
            )
        except Exception as e:
            logger.error("Error Fetching Analyser Status: %s", e)
            return get_analyser_status_response(status="error", errMsg=str(e))


openalgo = OpenAlgo()
