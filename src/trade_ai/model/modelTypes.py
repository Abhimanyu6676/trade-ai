from dataclasses import dataclass
from typing import Literal

TYPE_MODEL_INTENT = Literal["MODEL"]
TYPE_MODEL_INPUT_EVENTS = Literal["GET_PREDICTION", "GET_STOCK_STATUS"]
TYPE_MODEL_OUTPUTS_EVENTS = Literal["PREDICTION", "STOCK_STATUS"]
TYPE_MODEL_EVENTS = TYPE_MODEL_INPUT_EVENTS | TYPE_MODEL_OUTPUTS_EVENTS

ModelName = Literal["long_gru", "long_lgb", "short_gru", "short_lgb"]
PredictionTargetModel = Literal[
    "meta_ensemble",
    "long_gru",
    "long_lgb",
    "short_gru",
    "short_lgb",
]
PredictionClass = Literal[-1, 0, 1]


@dataclass(frozen=True)
class ModelPredictionBreakdown:
    probabilities: list[float]
    predicted_class: PredictionClass


@dataclass(frozen=True)
class DirectionPredictionBreakdown:
    lgb: ModelPredictionBreakdown | None
    gru: ModelPredictionBreakdown | None
    ensemble_probabilities: list[float] | None


@dataclass(frozen=True)
class PredictionBreakdown:
    target_model: PredictionTargetModel
    signal: PredictionClass | None
    long: DirectionPredictionBreakdown | None
    short: DirectionPredictionBreakdown | None
    meta_action: int | None
    position_size: float | None
    barrier_multiplier: float | None
    take_profit_pct: float | None
    stop_loss_pct: float | None


MODEL_NAMES: tuple[ModelName, ...] = (
    "long_lgb",
    "long_gru",
    "short_lgb",
    "short_gru",
)
