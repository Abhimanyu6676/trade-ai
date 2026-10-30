"""Coordinate directional base-model runs and unified PPO training."""

import argparse
import threading
from typing import Literal

from trade_ai.logger import logger
from trade_ai.model.modelTypes import ModelName
from trade_ai.model.models import get_model_manager

from .long.train_long_gru import train_long_gru
from .long.train_long_lgb import train_long_lgb
from .short.train_short_gru import train_short_gru
from .short.train_short_lgb import train_short_lgb
from .train_rl import train_meta_agent

TrainingMindset = Literal["long", "short", "both"]
TrainingComponent = Literal["gru", "lgb", "rl", "all"]
MINDSET_DIRECTIONS: dict[TrainingMindset, tuple[Literal["long", "short"], ...]] = {
    "long": ("long",),
    "short": ("short",),
    "both": ("long", "short"),
}

TRAINING_LOCK = threading.Lock()
TRAINING_THREAD: threading.Thread | None = None
TRAINING_STATE = "idle"
TRAINING_ERROR: str | None = None


def _selected_base_models(
    mindset: TrainingMindset, component: TrainingComponent
) -> tuple[ModelName, ...]:
    """Resolve selected directional/family pairs into model-manager keys."""
    families = ("lgb", "gru") if component == "all" else (component,)
    return tuple(
        f"{direction}_{family}"
        for direction in MINDSET_DIRECTIONS[mindset]
        for family in families
        if family in ("lgb", "gru")
    )  # type: ignore[return-value]


def run_training(
    mindset: TrainingMindset = "both",
    component: TrainingComponent = "all",
) -> bool:
    """Train requested specialists and optionally the unified meta-agent.

    RL is a common component over all four frozen base models. Selecting it with
    a single mindset still validates and trains from the combined OOS streams.
    """
    directions = MINDSET_DIRECTIONS[mindset]
    selected_models = _selected_base_models(mindset, component)
    train_rl = component in ("rl", "all")
    model_manager = get_model_manager()
    model_manager.backup_current_models(selected_models)
    if selected_models or train_rl:
        model_manager.backup_meta_agent()
        model_manager.unload_meta_agent()
    if selected_models:
        model_manager.unload_models(selected_models)

    training_succeeded = False
    try:
        all_checkpoints_saved = True
        if component in ("lgb", "all"):
            for direction in directions:
                logger.info("Starting %s LightGBM training.", direction)
                saved = (
                    train_long_lgb()
                    if direction == "long"
                    else train_short_lgb()
                )
                if not saved:
                    all_checkpoints_saved = False
                    logger.warning(
                        "%s LightGBM did not beat its validation baseline; "
                        "its existing checkpoint remains unchanged.",
                        direction,
                    )

        if component in ("gru", "all"):
            for direction in directions:
                logger.info("Starting %s GRU training.", direction)
                if direction == "long":
                    train_long_gru(epochs=20)
                else:
                    train_short_gru(epochs=20)
                logger.info("%s GRU training completed.", direction)

        rl_trained = True
        if train_rl:
            logger.info("Starting unified PPO meta-agent training.")
            rl_trained = train_meta_agent()
            if not rl_trained:
                logger.warning(
                    "The RL policy did not beat the flat selection baseline; "
                    "its previous artifact was not replaced."
                )

        training_succeeded = True
        return all_checkpoints_saved and rl_trained
    except Exception:
        logger.exception("Training failed for %s/%s.", mindset, component)
        raise
    finally:
        if training_succeeded:
            if selected_models:
                model_manager.reload_models(selected_models)
            if selected_models or train_rl:
                model_manager.reload_meta_agent()
        else:
            logger.warning(
                "Skipping model reload after failed %s/%s training; affected models "
                "will remain unloaded until reloaded explicitly or the app restarts.",
                mindset,
                component,
            )


def run_full_training_pipeline() -> bool:
    """Backward-compatible entry point for the full long/short/RL pipeline."""
    return run_training("both", "all")


def _background_training_worker(
    mindset: TrainingMindset, component: TrainingComponent
) -> None:
    global TRAINING_STATE, TRAINING_ERROR
    try:
        completed = run_training(mindset, component)
        with TRAINING_LOCK:
            TRAINING_STATE = "completed" if completed else "completed_with_skips"
            TRAINING_ERROR = None
        logger.info(
            "Training completed for %s/%s (all checkpoints accepted=%s).",
            mindset,
            component,
            completed,
        )
    except Exception as error:
        with TRAINING_LOCK:
            TRAINING_STATE = "failed"
            TRAINING_ERROR = str(error)
        logger.error("Background training failed: %s", error)


def start_master_training(
    mindset: TrainingMindset = "both",
    component: TrainingComponent = "all",
) -> str:
    """Start selected training jobs asynchronously in a daemon thread."""
    global TRAINING_THREAD, TRAINING_STATE, TRAINING_ERROR
    with TRAINING_LOCK:
        if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
            return "Training is already running in the background."
        TRAINING_STATE = "running"
        TRAINING_ERROR = None
        TRAINING_THREAD = threading.Thread(
            target=_background_training_worker,
            args=(mindset, component),
            name="trade-ai-training",
            daemon=True,
        )
        TRAINING_THREAD.start()
    return f"Training started for: {mindset}/{component}."


def get_training_status() -> dict[str, str | None]:
    """Return the current training state and optional error message."""
    with TRAINING_LOCK:
        return {"state": TRAINING_STATE, "error": TRAINING_ERROR}


def main() -> None:
    """Parse two optional positional selectors without interactive prompts."""
    parser = argparse.ArgumentParser(
        description="Train long/short specialists and the unified PPO policy."
    )
    parser.add_argument(
        "mindset",
        nargs="?",
        choices=("long", "short", "both"),
        default="both",
        help="Directional specialists to train (default: both).",
    )
    parser.add_argument(
        "component",
        nargs="?",
        choices=("gru", "lgb", "rl", "all"),
        default="all",
        help="Model component to train (default: all).",
    )
    args = parser.parse_args()
    run_training(args.mindset, args.component)


if __name__ == "__main__":
    main()
