import logging
import threading

from .train_gru import train_gru_model
from .train_lgb import train_lightgbm_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global thread and state control (modeled after your app's requirements)
TRAINING_LOCK = threading.Lock()
TRAINING_THREAD = None
TRAINING_STATE = "idle"  # "idle", "running", "completed", "failed"
TRAINING_ERROR = None


def run_full_training_pipeline() -> bool:
    """Orchestrates the complete training run for both brains:

    1. Trains LightGBM (Monolithic tabular tree booster)
    2. Trains GRU (Sequential PyTorch deep learning neural network)
    """
    try:
        logger.info("🚀 Starting Master Training Pipeline: Phase 1 (LightGBM)...")
        train_lightgbm_model()
        logger.info("✅ LightGBM training successfully completed.")

        logger.info("🚀 Starting Master Training Pipeline: Phase 2 (GRU)...")
        train_gru_model(epochs=10)
        logger.info("✅ GRU training successfully completed.")

        return True
    except Exception as e:
        logger.error("❌ Master Training Pipeline failed: %s", e, exc_info=True)
        raise e


def _background_training_worker():
    """Target worker function for running training in a background thread."""
    global TRAINING_STATE, TRAINING_ERROR

    try:
        success = run_full_training_pipeline()
        if not success:
            raise RuntimeError("Training pipeline returned failure status.")

        with TRAINING_LOCK:
            TRAINING_STATE = "completed"
            TRAINING_ERROR = None
        logger.info(
            "✅ [Background Training] All models trained and saved successfully."
        )

        # Optional: Reload inference engines if applicable
        # from src.prediction_engine import reload_engines
        # reload_engines()

    except Exception as error:
        with TRAINING_LOCK:
            TRAINING_STATE = "failed"
            TRAINING_ERROR = str(error)
        logger.error("❌ [Background Training] Failed: %s", error)


def start_master_training() -> str:
    """Initiates the master training runner asynchronously in a daemon thread.

    Prevents blocking the web server UI/API.
    """
    global TRAINING_THREAD, TRAINING_STATE, TRAINING_ERROR

    with TRAINING_LOCK:
        if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
            return "⚠️ Training is already running in the background."

        TRAINING_STATE = "running"
        TRAINING_ERROR = None
        TRAINING_THREAD = threading.Thread(
            target=_background_training_worker,
            name="trade-ai-master-training",
            daemon=True,
        )
        TRAINING_THREAD.start()

    return (
        "🚀 Master training pipeline started successfully in the background. "
        "Monitor console logs for live progress."
    )


def get_training_status() -> dict:
    """Returns current state of training for UI or API status checks."""
    with TRAINING_LOCK:
        return {"state": TRAINING_STATE, "error": TRAINING_ERROR}


if __name__ == "__main__":
    # Allows running the full pipeline synchronously from the terminal if desired
    run_full_training_pipeline()
