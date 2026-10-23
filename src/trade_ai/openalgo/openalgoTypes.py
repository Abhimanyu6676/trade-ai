from dataclasses import dataclass
from typing import Literal

TYPE_OPENALGO_INTENT = Literal["OPENALGO"]
TYPE_OPENALGO_INPUT_EVENTS = Literal["GET_HISTORY", "GET_ANALYSER_STATUS"]
TYPE_OPENALGO_OUTPUTS_EVENTS = Literal["HISTORY", "ANALYSER_STATUS"]
TYPE_OPENALGO_EVENTS = TYPE_OPENALGO_INPUT_EVENTS | TYPE_OPENALGO_OUTPUTS_EVENTS

TYPE_OPENALGO_EXCHANGE = Literal["NSE", "BSE"]

TYPE_OPENALGO_INTERVALS = Literal[
    "1m", "2m", "3m", "4m", "5m", "10m", "15m", "30m", "1h", "1d", "D", "W"
]


@dataclass
class get_analyser_status_response:
    status: Literal["success", "error"]
    analyze_mode: bool | None = None
    mode: Literal["analyze", "live"] | None = None
    errMsg: str | None = None
