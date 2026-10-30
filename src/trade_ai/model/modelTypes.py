from typing import Literal

TYPE_MODEL_INTENT = Literal["MODEL"]
TYPE_MODEL_INPUT_EVENTS = Literal["GET_PREDICTION", "GET_STOCK_STATUS"]
TYPE_MODEL_OUTPUTS_EVENTS = Literal["PREDICTION", "STOCK_STATUS"]
TYPE_MODEL_EVENTS = TYPE_MODEL_INPUT_EVENTS | TYPE_MODEL_OUTPUTS_EVENTS

ModelName = Literal["long_gru", "long_lgb", "short_gru", "short_lgb"]
MODEL_NAMES: tuple[ModelName, ...] = (
    "long_lgb",
    "long_gru",
    "short_lgb",
    "short_gru",
)
