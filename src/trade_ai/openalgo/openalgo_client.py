"""OpenAlgo client lifecycle and API operations."""

import asyncio
import os
from datetime import date
from typing import Any, Literal, TypedDict

import pandas as pd
from openalgo import api

from trade_ai.logger import logger
from trade_ai.util import lowercase_ohlcv_columns

from .openalgoTypes import (
    TYPE_OPENALGO_EXCHANGE,
    TYPE_OPENALGO_INTERVALS,
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
        start_date: date,
        end_date: date,
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
