"""Own the in-memory LightGBM and GRU model instances and their lifecycle."""

import logging
import os
import shutil
import threading
from datetime import datetime
from typing import Literal

import lightgbm as lgb
import torch
from torch import nn

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
LGB_MODEL_PATH = os.path.join(CACHE_DIR, "lgb_trading_model.txt")
GRU_MODEL_PATH = os.path.join(CACHE_DIR, "gru_trading_model.pth")

ModelName = Literal["gru", "lgb"]
MODEL_NAMES: tuple[ModelName, ...] = ("gru", "lgb")

logger = logging.getLogger(__name__)


class GRUTradingBrain(nn.Module):
    """Two-layer GRU architecture with dropout and a linear classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])


ModelInstance = lgb.Booster | GRUTradingBrain


class ModelManager:
    """Keep loaded models available and own loading, backup, and recovery."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()
        self._models: dict[ModelName, ModelInstance | None] = {
            "gru": None,
            "lgb": None,
        }
        self.load_models()

    def get_model(self, model_name: ModelName) -> ModelInstance | None:
        """Return the currently loaded model instance, if available."""
        with self._lock:
            return self._models[model_name]

    def save_model_to_memory(self, model_name: ModelName, model: ModelInstance) -> None:
        """Store a loaded instance for use by inference and other callers."""
        with self._lock:
            self._models[model_name] = model

    def load_models(self, model_names: tuple[ModelName, ...] = MODEL_NAMES) -> bool:
        """Load selected checkpoints into memory; warn when a checkpoint is absent."""
        loaded_any = False
        with self._lock:
            for model_name in model_names:
                try:
                    if model_name == "lgb":
                        if not os.path.isfile(LGB_MODEL_PATH):
                            logger.warning(
                                "No LightGBM model file exists at %s; train it first.",
                                LGB_MODEL_PATH,
                            )
                            self._models[model_name] = None
                            continue
                        model = lgb.Booster(model_file=LGB_MODEL_PATH)
                    else:
                        if not os.path.isfile(GRU_MODEL_PATH):
                            logger.warning(
                                "No GRU model file exists at %s; train it first.",
                                GRU_MODEL_PATH,
                            )
                            self._models[model_name] = None
                            continue
                        state_dict = torch.load(
                            GRU_MODEL_PATH,
                            map_location=self.device,
                            weights_only=True,
                        )
                        input_weights = state_dict.get("gru.weight_ih_l0")
                        if input_weights is None:
                            raise ValueError(
                                "GRU checkpoint is missing gru.weight_ih_l0"
                            )
                        model = GRUTradingBrain(
                            input_dim=int(input_weights.shape[1])
                        ).to(self.device)
                        model.load_state_dict(state_dict)
                        model.eval()

                    self.save_model_to_memory(model_name, model)
                    loaded_any = True
                    logger.info("Loaded %s model from disk.", model_name.upper())
                except Exception:
                    self._models[model_name] = None
                    logger.exception("Failed to load %s model.", model_name.upper())
        return loaded_any

    def unload_models(self, model_names: tuple[ModelName, ...] = MODEL_NAMES) -> None:
        """Release selected model references from memory."""
        with self._lock:
            for model_name in model_names:
                self._models[model_name] = None
            if "gru" in model_names and self.device.type == "cuda":
                torch.cuda.empty_cache()
        logger.info("Unloaded models: %s", ", ".join(model_names))

    def reload_models(self, model_names: tuple[ModelName, ...] = MODEL_NAMES) -> bool:
        """Reload selected checkpoints after training or recovery."""
        return self.load_models(model_names)

    def backup_current_models(
        self, model_names: tuple[ModelName, ...] = MODEL_NAMES
    ) -> list[str]:
        """Back up loaded model files using a timestamped filename."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_paths: list[str] = []
        with self._lock:
            for model_name in model_names:
                source_path = self._model_path(model_name)
                if self._models[model_name] is None or not os.path.isfile(source_path):
                    continue
                stem, extension = os.path.splitext(source_path)
                backup_path = f"{stem}_backup_{timestamp}{extension}"
                shutil.copy2(source_path, backup_path)
                backup_paths.append(backup_path)
                logger.info("Backed up %s model to %s", model_name.upper(), backup_path)
        return backup_paths

    @staticmethod
    def _model_path(model_name: ModelName) -> str:
        return GRU_MODEL_PATH if model_name == "gru" else LGB_MODEL_PATH


_model_manager: ModelManager | None = None
_model_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Return the process-wide model manager, loading checkpoints on first use."""
    global _model_manager
    if _model_manager is None:
        with _model_manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager
