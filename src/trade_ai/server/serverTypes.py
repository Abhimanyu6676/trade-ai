from dataclasses import dataclass

from trade_ai.model.modelTypes import TYPE_MODEL_OUTPUTS_EVENTS
from trade_ai.openalgo.openalgoTypes import (
    TYPE_OPENALGO_OUTPUTS_EVENTS,
)
from trade_ai.universalTypes import (
    TYPE_SERVER_OUTPUT_EVENTS,
    UNIVERSAL_INTENTS,
)

SERVER_RESPONSE_EVENTS = (
    TYPE_MODEL_OUTPUTS_EVENTS | TYPE_OPENALGO_OUTPUTS_EVENTS | TYPE_SERVER_OUTPUT_EVENTS
)


# Generic UniversalServerResponse envelope types
@dataclass
class serverDataType[E, DATA, ERR]:
    eventType: E
    data: DATA | None = None
    error: ERR | None = None


@dataclass
class UniversalServerResponse[
    I: UNIVERSAL_INTENTS,
    E: SERVER_RESPONSE_EVENTS,
    DATA,
    ERR,
]:
    intent: I
    data: serverDataType[E, DATA, ERR]
