"""Select and coordinate training runs for the model training modules."""

import argparse
import threading
from typing import Literal

from trade_ai.logger import logger
from trade_ai.model.models import get_model_manager
from trade_ai.model.modelTypes import ModelName

from .train_gru import train_gru_model
from .train_lgb import train_lightgbm_model

TrainingSelection = Literal["gru", "lgb", "both"]
TRAINING_MODELS: dict[TrainingSelection, tuple[ModelName, ...]] = {
    "gru": ("gru",),
    "lgb": ("lgb",),
    "both": ("lgb", "gru"),
}

TRAINING_LOCK = threading.Lock()
TRAINING_THREAD: threading.Thread | None = None
TRAINING_STATE = "idle"
TRAINING_ERROR: str | None = None


def run_training(training_selection: TrainingSelection = "both") -> bool:
    """Train one selected model or both, then reload trained checkpoints."""
    selected_models = TRAINING_MODELS[training_selection]
    model_manager = get_model_manager()
    model_manager.backup_current_models(selected_models)
    model_manager.unload_models(selected_models)

    training_succeeded = False
    try:
        if training_selection in ("lgb", "both"):
            logger.info("Starting LightGBM training.")
            train_lightgbm_model()
            logger.info("LightGBM training completed.")

        if training_selection in ("gru", "both"):
            logger.info("Starting GRU training.")
            train_gru_model(epochs=20)
            logger.info("GRU training completed.")

        training_succeeded = True
        return True
    except Exception:
        logger.exception("Training failed for selection %s.", training_selection)
        raise
    finally:
        if training_succeeded:
            model_manager.reload_models(selected_models)
        else:
            logger.warning(
                "Skipping model reload after failed %s training; selected models "
                "will remain unloaded until reloaded explicitly.",
                training_selection,
            )


def run_full_training_pipeline() -> bool:
    """Backward-compatible entry point for training both models."""
    return run_training("both")


def _background_training_worker(training_selection: TrainingSelection) -> None:
    global TRAINING_STATE, TRAINING_ERROR

    try:
        run_training(training_selection)
        with TRAINING_LOCK:
            TRAINING_STATE = "completed"
            TRAINING_ERROR = None
        logger.info("Training completed successfully for %s.", training_selection)
    except Exception as error:
        with TRAINING_LOCK:
            TRAINING_STATE = "failed"
            TRAINING_ERROR = str(error)
        logger.error("Background training failed: %s", error)


def start_master_training(training_selection: TrainingSelection = "both") -> str:
    """Start selected training jobs asynchronously in a daemon thread."""
    global TRAINING_THREAD, TRAINING_STATE, TRAINING_ERROR

    with TRAINING_LOCK:
        if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
            return "Training is already running in the background."

        TRAINING_STATE = "running"
        TRAINING_ERROR = None
        TRAINING_THREAD = threading.Thread(
            target=_background_training_worker,
            args=(training_selection,),
            name="trade-ai-model-training",
            daemon=True,
        )
        TRAINING_THREAD.start()

    return f"Training started for: {training_selection}."


def get_training_status() -> dict[str, str | None]:
    """Return the current training state and optional error message."""
    with TRAINING_LOCK:
        return {"state": TRAINING_STATE, "error": TRAINING_ERROR}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train selected trade-ai models.")
    parser.add_argument(
        "model",
        nargs="?",
        choices=("gru", "lgb", "both"),
        default="both",
        help="Model to train (default: both).",
    )
    run_training(parser.parse_args().model)
