import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from trade_ai.features import compute_normalized_features

from .train_gru import GRUTradingBrain

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
LGB_MODEL_PATH = os.path.join(CACHE_DIR, "lgb_trading_model.txt")
GRU_MODEL_PATH = os.path.join(CACHE_DIR, "gru_trading_model.pth")

logger = logging.getLogger(__name__)


class EnsemblePredictionEngine:
    """Loads and orchestrates both LightGBM and GRU models to produce

    unified ensemble trading signals based on Triple Barrier probabilities.
    """

    def __init__(self):
        self.lgb_model = None
        self.gru_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()

    def load_models(self):
        """Safely loads both model checkpoints from the cache directory."""
        # 1. Load LightGBM Booster
        if os.path.exists(LGB_MODEL_PATH):
            try:
                self.lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)
                logger.info("📂 Loaded LightGBM model from %s", LGB_MODEL_PATH)
            except Exception as e:
                logger.error("❌ Failed to load LightGBM model: %s", e)

        # 2. Load GRU Neural Network
        if os.path.exists(GRU_MODEL_PATH):
            try:
                # Note: We initialize with dummy input_dim; PyTorch state_dict will configure weights properly
                # We assume standard 64 hidden units and 2 layers matching train_gru.py
                # We can dynamically infer input_dim or load it if saved. Here we instantiate on first predict if needed.
                logger.info(
                    "📂 Found GRU checkpoint at %s (deferred instantiation)",
                    GRU_MODEL_PATH,
                )
            except Exception as e:
                logger.error("❌ Failed to check GRU model path: %s", e)

    def reload_engines(self) -> bool:
        """Hot-reloads models after background training finishes."""
        self.load_models()
        return self.lgb_model is not None or self.gru_model is not None

    def unload_engines(self):
        """Unloads models to free memory before a new training run."""
        self.lgb_model = None
        self.gru_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 Unloaded prediction engines from memory.")

    def predict(
        self, raw_df: pd.DataFrame, ticker: str, sequence_length: int = 30
    ) -> tuple[int, float, dict]:
        """Generates an ensemble prediction for the latest candle.

        Returns:
            signal (int): -1 (Sell/Stop-Loss), 0 (Hold/Timeout), 1 (Buy/Take-Profit)
            confidence (float): Probability score of the chosen action
            probabilities (dict): Full class probability breakdown for both brains
        """
        if self.lgb_model is None and self.gru_model is None:
            logger.warning("⚠️ No trained models loaded in prediction engine.")
            return 0, 0.0, {"error": "Models not loaded"}

        # 1. Separate Feature Engineering & Metadata Encoding
        df_lgb_feat = compute_normalized_features(raw_df, ticker, model_type="lgb")

        df_gru_feat = compute_normalized_features(raw_df, ticker, model_type="gru")

        # Exclude non-feature columns
        exclude_cols = ["timestamp", "target"]
        lgb_feature_cols = [
            col for col in df_lgb_feat.columns if col not in exclude_cols
        ]
        gru_feature_cols = [
            col for col in df_gru_feat.columns if col not in exclude_cols
        ]

        if len(df_lgb_feat) < sequence_length:
            return 0, 0.0, {"error": "Insufficient rows for sequence length"}

        # Extract latest snapshot for LightGBM
        latest_snapshot = df_lgb_feat[lgb_feature_cols].iloc[[-1]]

        # Extract rolling window sequence for GRU
        sequence_features = (
            df_gru_feat[gru_feature_cols]
            .iloc[-sequence_length:]
            .to_numpy(dtype=np.float32)
        )

        lgb_probs = None
        gru_probs = None

        # --- LightGBM Inference ---
        if self.lgb_model is not None:
            # lgb_model.predict outputs probabilities for multi-class [class_0, class_1, class_2]
            raw_lgb_pred = self.lgb_model.predict(latest_snapshot)

            # Explicitly cast to numpy array to satisfy static type checkers
            pred_arr = np.asarray(raw_lgb_pred)

            if pred_arr.ndim > 0 and len(pred_arr) > 0:
                lgb_probs = pred_arr[0]  # array of 3 probabilities

        # --- GRU Inference ---
        if self.gru_model is None and os.path.exists(GRU_MODEL_PATH):
            # Lazy-load GRU model dynamically once input dimension is known
            try:
                input_dim = len(gru_feature_cols)
                self.gru_model = GRUTradingBrain(
                    input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=3
                ).to(self.device)
                self.gru_model.load_state_dict(
                    torch.load(GRU_MODEL_PATH, map_location=self.device)
                )
                self.gru_model.eval()
                logger.info("Successfully lazy-loaded GRU prediction brain.")
            except Exception as e:
                logger.error("Failed to initialize GRU brain for inference: %s", e)

        if self.gru_model is not None:
            self.gru_model.eval()
            with torch.no_grad():
                x_tensor = (
                    torch.tensor(sequence_features, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                logits = self.gru_model(x_tensor)
                # Apply Softmax to convert raw logits to probabilities
                probs = torch.softmax(logits, dim=1)
                gru_probs = probs.cpu().numpy()[0]

        # --- Ensemble Blending (50% LightGBM + 50% GRU) ---
        if lgb_probs is not None and gru_probs is not None:
            final_probs = (0.5 * lgb_probs) + (0.5 * gru_probs)
        elif lgb_probs is not None:
            final_probs = lgb_probs
        elif gru_probs is not None:
            final_probs = gru_probs
        else:
            return 0, 0.0, {"error": "Inference computation failed"}

        # Map class index back to Triple Barrier target space:
        # Index 0 -> Class 0 (-1: Stop-Loss)
        # Index 1 -> Class 1 ( 0: Timeout)
        # Index 2 -> Class 2 ( 1: Take-Profit)
        class_mapping = {0: -1, 1: 0, 2: 1}
        best_class_idx = int(np.argmax(final_probs))
        signal = class_mapping[best_class_idx]
        confidence = float(final_probs[best_class_idx])

        breakdown = {
            "lgb_probabilities": lgb_probs.tolist() if lgb_probs is not None else None,
            "gru_probabilities": gru_probs.tolist() if gru_probs is not None else None,
            "ensemble_probabilities": final_probs.tolist(),
        }

        return signal, confidence, breakdown


# Global singleton engine instance for application use
_engine_instance = None


def get_prediction_engine() -> EnsemblePredictionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EnsemblePredictionEngine()
    return _engine_instance


def run_ai_prediction_inference(
    raw_df: pd.DataFrame, ticker: str, sequence_length: int = 30
) -> tuple[int, float]:
    """Wrapper function matching data engine calls."""
    engine = get_prediction_engine()
    signal, confidence, _ = engine.predict(raw_df, ticker, sequence_length)
    return signal, confidence


def reload_prediction_brain() -> bool:
    engine = get_prediction_engine()
    return engine.reload_engines()


def unload_prediction_brain():
    engine = get_prediction_engine()
    engine.unload_engines()
