"""Versioned trading rules and artifact locations for directional models."""

from pathlib import Path
from typing import Literal

Direction = Literal["long", "short"]
ModelFamily = Literal["gru", "lgb"]

# Increment whenever a trained artifact's contract changes: features, labels,
# side-specific behavior, RL observation fields, or action meanings. The more
# specific feature and label versions remain in manifests for readable errors.
SCHEMA_VERSION = 3

LONG_TAKE_PROFIT_PCT = 0.01
LONG_STOP_LOSS_PCT = 0.005
SHORT_TAKE_PROFIT_PCT = -LONG_TAKE_PROFIT_PCT
SHORT_STOP_LOSS_PCT = LONG_STOP_LOSS_PCT
BARRIER_MULTIPLIERS = (0.75, 1.0, 1.25)

MAX_HOLDING_PERIOD = 60
ENTRY_START_HOUR = 10
ENTRY_START_MINUTE = 0
ENTRY_END_HOUR = 14
ENTRY_END_MINUTE = 30
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 15

# A one-way 5 bp execution-cost assumption is used only by the historical RL
# simulator. Set to zero only when intentionally modeling frictionless fills.
TRANSACTION_COST_PCT = 0.0005
DISAGREEMENT_PENALTY = 0.0001
DRAWDOWN_PENALTY = 0.0001
POSITION_SIZE_LEVELS = (0.0, 0.5, 1.0, 1.5)

MODEL_ROOT = Path(__file__).resolve().parents[1] / "model" / ".run_cache"
LONG_MODEL_DIR = MODEL_ROOT / "long"
SHORT_MODEL_DIR = MODEL_ROOT / "short"
RL_MODEL_DIR = MODEL_ROOT / "rl"
RL_MODEL_PATH = RL_MODEL_DIR / "meta_agent.zip"
RL_SCHEMA_PATH = RL_MODEL_DIR / "meta_agent.schema.json"
RL_OOS_DIR = RL_MODEL_DIR / "oos"


def barrier_percentages(
    direction: Direction, multiplier: float = 1.0
) -> tuple[float, float]:
    """Return positive target/stop distances scaled by the selected bracket."""
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")
    if multiplier not in BARRIER_MULTIPLIERS:
        raise ValueError(
            f"barrier multiplier must be one of {BARRIER_MULTIPLIERS}"
        )
    take_profit = (
        abs(LONG_TAKE_PROFIT_PCT)
        if direction == "long"
        else abs(SHORT_TAKE_PROFIT_PCT)
    )
    stop_loss = (
        abs(LONG_STOP_LOSS_PCT)
        if direction == "long"
        else abs(SHORT_STOP_LOSS_PCT)
    )
    return take_profit * multiplier, stop_loss * multiplier


def model_artifact_paths(
    direction: Direction, family: ModelFamily
) -> tuple[Path, Path | None]:
    """Return the checkpoint and optional sidecar path for one base model."""
    directory = LONG_MODEL_DIR if direction == "long" else SHORT_MODEL_DIR
    if family == "gru":
        return directory / "gru_model.pth", None
    checkpoint = directory / "lgb_model.txt"
    return checkpoint, Path(f"{checkpoint}.schema.json")
