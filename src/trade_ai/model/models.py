"""Own long/short base models and the unified PPO policy lifecycle."""

import hashlib
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import torch
from torch import nn

from trade_ai.config.trading_config import (
    RL_MODEL_PATH,
    RL_SCHEMA_PATH,
    SCHEMA_VERSION,
    Direction,
    ModelFamily,
    model_artifact_paths,
)
from trade_ai.logger import logger
from trade_ai.model.features import (
    FEATURE_SCHEMA_VERSION,
    GRU_MODEL_INPUT_COLUMNS,
    LGB_MODEL_INPUT_COLUMNS,
)
from trade_ai.model.modelTypes import MODEL_NAMES, ModelName
from trade_ai.model.training.labeling import LABEL_SCHEMA_VERSION
from trade_ai.model.training.trading_env import (
    OBSERVATION_FEATURES,
    UNIFIED_ACTION_CONTRACT,
)

LONG_LGB_MODEL_PATH, LONG_LGB_MODEL_SCHEMA_PATH = model_artifact_paths("long", "lgb")
LONG_GRU_MODEL_PATH, _ = model_artifact_paths("long", "gru")
SHORT_LGB_MODEL_PATH, SHORT_LGB_MODEL_SCHEMA_PATH = model_artifact_paths("short", "lgb")
SHORT_GRU_MODEL_PATH, _ = model_artifact_paths("short", "gru")

# Retained aliases for callers from the previous long-only package layout.
LGB_MODEL_PATH = LONG_LGB_MODEL_PATH
LGB_MODEL_SCHEMA_PATH = LONG_LGB_MODEL_SCHEMA_PATH
GRU_MODEL_PATH = LONG_GRU_MODEL_PATH

MODEL_ARTIFACT_PATHS: dict[ModelName, Path] = {
    "long_lgb": LONG_LGB_MODEL_PATH,
    "long_gru": LONG_GRU_MODEL_PATH,
    "short_lgb": SHORT_LGB_MODEL_PATH,
    "short_gru": SHORT_GRU_MODEL_PATH,
}
MODEL_DIRECTIONS: dict[ModelName, Direction] = {
    "long_lgb": "long",
    "long_gru": "long",
    "short_lgb": "short",
    "short_gru": "short",
}
MODEL_FAMILIES: dict[ModelName, ModelFamily] = {
    "long_lgb": "lgb",
    "long_gru": "gru",
    "short_lgb": "lgb",
    "short_gru": "gru",
}


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
    """Load, unload, back up, and validate all inference artifacts."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()
        self._models: dict[ModelName, ModelInstance | None] = {
            model_name: None for model_name in MODEL_NAMES
        }
        self._meta_agent: Any | None = None
        self.load_models()
        self.load_meta_agent()

    @staticmethod
    def _normalize_name(model_name: ModelName | str) -> ModelName:
        """Map legacy long-only names to their new artifact keys."""
        aliases = {"gru": "long_gru", "lgb": "long_lgb"}
        normalized = aliases.get(model_name, model_name)
        if normalized not in MODEL_ARTIFACT_PATHS:
            raise ValueError(f"Unknown base model name: {model_name}")
        return normalized  # type: ignore[return-value]

    def get_model(self, model_name: ModelName | str) -> ModelInstance | None:
        """Return a currently loaded long/short base-model instance."""
        with self._lock:
            return self._models[self._normalize_name(model_name)]

    def get_meta_agent(self) -> Any | None:
        """Return the loaded PPO decision policy, if its schemas match."""
        with self._lock:
            return self._meta_agent

    def load_models(self, model_names: tuple[ModelName, ...] = MODEL_NAMES) -> bool:
        """Load selected checkpoints only after direction and schema validation."""
        loaded_any = False
        with self._lock:
            for model_name in model_names:
                direction = MODEL_DIRECTIONS[model_name]
                family = MODEL_FAMILIES[model_name]
                checkpoint_path, schema_path = model_artifact_paths(direction, family)
                try:
                    if not checkpoint_path.is_file():
                        logger.warning(
                            "No %s %s checkpoint exists at %s; train it first.",
                            direction,
                            family.upper(),
                            checkpoint_path,
                        )
                        self._models[model_name] = None
                        continue
                    if family == "lgb":
                        if schema_path is None or not schema_path.is_file():
                            raise ValueError("LightGBM checkpoint is missing its schema sidecar")
                        with schema_path.open(encoding="utf-8") as schema_file:
                            checkpoint = json.load(schema_file)
                        self._validate_base_schema(checkpoint, model_name)
                        if checkpoint.get("checkpoint_sha256") != self._sha256_file(
                            checkpoint_path
                        ):
                            raise ValueError(
                                "LightGBM checkpoint does not match its schema sidecar; "
                                "retrain it"
                            )
                        model: ModelInstance = lgb.Booster(
                            model_file=str(checkpoint_path)
                        )
                        expected_columns = LGB_MODEL_INPUT_COLUMNS
                        if tuple(model.feature_name()) != expected_columns:
                            raise ValueError(
                                "LightGBM feature names do not match the model schema"
                            )
                    else:
                        checkpoint = torch.load(
                            checkpoint_path,
                            map_location=self.device,
                            weights_only=True,
                        )
                        self._validate_base_schema(checkpoint, model_name)
                        state_dict = checkpoint.get("state_dict")
                        if not isinstance(state_dict, dict):
                            raise ValueError("GRU checkpoint is missing its state_dict")
                        input_weights = state_dict.get("gru.weight_ih_l0")
                        if input_weights is None:
                            raise ValueError("GRU checkpoint is missing gru.weight_ih_l0")
                        if int(input_weights.shape[1]) != len(GRU_MODEL_INPUT_COLUMNS):
                            raise ValueError(
                                "GRU input width does not match the model schema"
                            )
                        model = GRUTradingBrain(
                            input_dim=int(input_weights.shape[1])
                        ).to(self.device)
                        model.load_state_dict(state_dict)
                        model.eval()

                    self._models[model_name] = model
                    loaded_any = True
                    logger.info("Loaded %s %s base model.", direction, family.upper())
                except Exception:
                    self._models[model_name] = None
                    logger.exception("Failed to load %s base model.", model_name)
        return loaded_any

    @staticmethod
    def _validate_base_schema(checkpoint: object, model_name: ModelName) -> None:
        """Reject checkpoints trained under old strategy, labels, or features."""
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{model_name} checkpoint must contain schema metadata")
        expected_direction = MODEL_DIRECTIONS[model_name]
        expected_columns = (
            GRU_MODEL_INPUT_COLUMNS
            if MODEL_FAMILIES[model_name] == "gru"
            else LGB_MODEL_INPUT_COLUMNS
        )
        expected = {
            "trading_schema_version": SCHEMA_VERSION,
            "direction": expected_direction,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
        }
        for key, expected_value in expected.items():
            if checkpoint.get(key) != expected_value:
                raise ValueError(
                    f"{model_name} checkpoint {key} is incompatible; retrain it"
                )
        if tuple(checkpoint.get("feature_columns", ())) != expected_columns:
            raise ValueError(
                f"{model_name} checkpoint feature columns do not match this code; "
                "retrain it"
            )

    def load_meta_agent(self) -> bool:
        """Load PPO only when its state contract and all four bases still match."""
        with self._lock:
            self._meta_agent = None
            if not RL_MODEL_PATH.is_file() or not RL_SCHEMA_PATH.is_file():
                logger.warning("No validated RL meta-agent exists; train it first.")
                return False
            try:
                with RL_SCHEMA_PATH.open(encoding="utf-8") as schema_file:
                    schema = json.load(schema_file)
                if schema.get("trading_schema_version") != SCHEMA_VERSION:
                    raise ValueError("RL trading schema is incompatible; retrain it")
                if schema.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                    raise ValueError("RL feature schema is incompatible; retrain it")
                if schema.get("label_schema_version") != LABEL_SCHEMA_VERSION:
                    raise ValueError("RL label schema is incompatible; retrain it")
                if schema.get("observation_features") != list(OBSERVATION_FEATURES):
                    raise ValueError("RL observation columns changed; retrain it")
                if schema.get("action_contract") != list(UNIFIED_ACTION_CONTRACT):
                    raise ValueError("RL action mapping changed; retrain it")
                if schema.get("agent_sha256") != self._sha256_file(RL_MODEL_PATH):
                    raise ValueError(
                        "RL checkpoint does not match its schema sidecar; retrain it"
                    )
                expected_hashes: dict[str, str] = {}
                for direction in ("long", "short"):
                    for family in ("lgb", "gru"):
                        model_path, _ = model_artifact_paths(direction, family)
                        if not model_path.is_file():
                            raise ValueError(
                                f"RL requires a loaded {direction} {family} model"
                            )
                        expected_hashes[f"{direction}_{family}"] = self._sha256_file(
                            model_path
                        )
                if schema.get("base_checkpoint_sha256") != expected_hashes:
                    raise ValueError(
                        "RL base checkpoint hashes do not match the loaded models; "
                        "retrain the meta-agent"
                    )
                from stable_baselines3 import PPO

                self._meta_agent = PPO.load(str(RL_MODEL_PATH), device="cpu")
                logger.info("Loaded validated RL meta-agent from %s", RL_MODEL_PATH)
                return True
            except Exception:
                logger.exception("Failed to load RL meta-agent; decisions will abstain.")
                self._meta_agent = None
                return False

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Return a checkpoint digest for meta-agent provenance validation."""
        digest = hashlib.sha256()
        with path.open("rb") as artifact_file:
            for block in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def unload_models(self, model_names: tuple[ModelName, ...] = MODEL_NAMES) -> None:
        """Release selected base model references from memory."""
        with self._lock:
            for model_name in model_names:
                self._models[model_name] = None
            if any("gru" in name for name in model_names) and self.device.type == "cuda":
                torch.cuda.empty_cache()
        logger.info("Unloaded base models: %s", ", ".join(model_names))

    def unload_meta_agent(self) -> None:
        """Release the PPO policy before training or replacing its checkpoint."""
        with self._lock:
            self._meta_agent = None

    def reload_models(self, model_names: tuple[ModelName, ...] = MODEL_NAMES) -> bool:
        """Reload selected base model artifacts."""
        return self.load_models(model_names)

    def reload_meta_agent(self) -> bool:
        """Reload the PPO policy after training."""
        return self.load_meta_agent()

    def backup_current_models(
        self, model_names: tuple[ModelName, ...] = MODEL_NAMES
    ) -> list[str]:
        """Back up selected checkpoints and schema sidecars with timestamps."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_paths: list[str] = []
        with self._lock:
            for model_name in model_names:
                direction = MODEL_DIRECTIONS[model_name]
                family = MODEL_FAMILIES[model_name]
                model_path, schema_path = model_artifact_paths(direction, family)
                if not model_path.is_file():
                    continue
                backup_path = model_path.with_name(
                    f"{model_path.stem}_backup_{timestamp}{model_path.suffix}"
                )
                shutil.copy2(model_path, backup_path)
                backup_paths.append(str(backup_path))
                if schema_path is not None and schema_path.is_file():
                    backup_schema = Path(f"{backup_path}.schema.json")
                    shutil.copy2(schema_path, backup_schema)
                    backup_paths.append(str(backup_schema))
                logger.info("Backed up %s to %s", model_name, backup_path)
        return backup_paths

    def backup_meta_agent(self) -> list[str]:
        """Back up an existing RL checkpoint and its schema sidecar."""
        if not RL_MODEL_PATH.is_file():
            return []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = RL_MODEL_PATH.with_name(
            f"meta_agent_backup_{timestamp}{RL_MODEL_PATH.suffix}"
        )
        shutil.copy2(RL_MODEL_PATH, backup_path)
        if RL_SCHEMA_PATH.is_file():
            shutil.copy2(
                RL_SCHEMA_PATH,
                backup_path.with_name(f"{backup_path.stem}.schema.json"),
            )
        return [str(backup_path)]


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
