"""Train and validate the unified PPO policy on frozen OOS base predictions."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from trade_ai.config.trading_config import (
    RL_MODEL_DIR,
    RL_MODEL_PATH,
    RL_SCHEMA_PATH,
    SCHEMA_VERSION,
    model_artifact_paths,
)
from trade_ai.logger import logger
from trade_ai.model.features import FEATURE_SCHEMA_VERSION

from .labeling import LABEL_SCHEMA_VERSION
from .rl.oos_data import (
    load_combined_final_predictions,
    load_combined_oos_predictions,
    sha256_file,
)
from .trading_env import (
    MetaTradingEnv,
    UNIFIED_ACTION_CONTRACT,
    OBSERVATION_FEATURES,
)


def _chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split walk-forward OOS forecasts into chronological PPO fit and selection."""
    session_dates = sorted(frame["session_date"].astype(str).drop_duplicates())
    if len(session_dates) < 2:
        raise RuntimeError("RL training requires at least two distinct OOS sessions")
    selection_index = max(
        1, min(len(session_dates) - 1, int(len(session_dates) * 0.80))
    )
    selection_start = session_dates[selection_index]
    training = frame.loc[frame["session_date"].astype(str) < selection_start].copy()
    selection = frame.loc[frame["session_date"].astype(str) >= selection_start].copy()
    if training.empty or selection.empty:
        raise RuntimeError("RL chronological fit/selection split is empty")
    return training, selection


def _evaluate_all_episodes(
    policy: PPO, env: MetaTradingEnv
) -> tuple[float, float, int]:
    """Evaluate a deterministic policy once on every available ticker/session."""
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    for episode_index in range(len(env.episodes)):
        observation, _ = env.reset(options={"episode_index": episode_index})
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(
                int(np.asarray(action).reshape(-1)[0])
            )
            total_reward += float(reward)
            steps += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    if not episode_rewards:
        raise RuntimeError(
            "Cannot evaluate PPO because the evaluation split has no episodes"
        )
    return (
        float(np.mean(episode_rewards)),
        float(np.mean(episode_lengths)),
        len(episode_rewards),
    )


def _base_checkpoint_hashes() -> dict[str, str]:
    """Return content hashes for the four checkpoints used by the policy."""
    result: dict[str, str] = {}
    for direction in ("long", "short"):
        for family in ("lgb", "gru"):
            checkpoint, _ = model_artifact_paths(direction, family)
            if not checkpoint.is_file():
                raise RuntimeError(
                    f"RL training requires {direction} {family} checkpoint: {checkpoint}"
                )
            result[f"{direction}_{family}"] = sha256_file(checkpoint)
    return result


def _write_schema(data: dict[str, object]) -> None:
    """Atomically store PPO artifact metadata."""
    RL_SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with TemporaryDirectory(dir=RL_SCHEMA_PATH.parent) as temp_dir:
            temp_file = Path(temp_dir) / "meta_agent.schema.json"
            temp_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            temp_path = str(temp_file)
            os.replace(temp_path, RL_SCHEMA_PATH)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def train_meta_agent(total_timesteps: int = 50_000, seed: int = 42) -> bool:
    """Fit and select PPO on walk-forward OOS data, then report final holdout once.

    Base models remain frozen. The chronological OOS validation tail is kept
    outside PPO fitting and selects the candidate. The newest base-model
    holdout is evaluated after selection and never decides whether it is saved.
    """
    if total_timesteps < 1_000:
        raise ValueError("total_timesteps must be at least 1000")
    base_hashes = _base_checkpoint_hashes()
    oos_frame = load_combined_oos_predictions()
    training_frame, selection_frame = _chronological_split(oos_frame)
    final_frame = load_combined_final_predictions()
    training_env = MetaTradingEnv(training_frame)
    selection_env = MetaTradingEnv(selection_frame)
    final_env = MetaTradingEnv(final_frame)
    if (
        len(training_env.episodes) < 2
        or not selection_env.episodes
        or not final_env.episodes
    ):
        raise RuntimeError(
            "RL training requires multiple fit episodes plus later selection "
            "and final-holdout episodes."
        )

    RL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Training unified PPO policy: train_rows=%d train_episodes=%d "
        "selection_rows=%d selection_episodes=%d final_holdout_rows=%d "
        "final_holdout_episodes=%d "
        "timesteps=%d",
        len(training_frame),
        len(training_env.episodes),
        len(selection_frame),
        len(selection_env.episodes),
        len(final_frame),
        len(final_env.episodes),
        total_timesteps,
    )

    with TemporaryDirectory(dir=RL_MODEL_DIR) as temp_dir:
        callback = EvalCallback(
            selection_env,
            best_model_save_path=temp_dir,
            log_path=temp_dir,
            eval_freq=max(1_000, total_timesteps // 10),
            n_eval_episodes=min(10, len(selection_env.episodes)),
            deterministic=True,
            verbose=0,
        )
        policy = PPO(
            "MlpPolicy",
            training_env,
            seed=seed,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            verbose=1,
            device="cpu",
        )
        policy.learn(total_timesteps=total_timesteps, callback=callback)
        best_path = Path(temp_dir) / "best_model.zip"
        candidate = (
            PPO.load(best_path, env=selection_env)
            if best_path.is_file()
            else policy
        )
        selection_mean_reward, selection_mean_length, selection_episodes = (
            _evaluate_all_episodes(candidate, selection_env)
        )
        logger.info(
            "PPO chronological selection: episodes=%d mean_reward=%.5f "
            "mean_episode_steps=%.1f",
            selection_episodes,
            selection_mean_reward,
            selection_mean_length,
        )
        if selection_mean_reward <= 0:
            logger.warning(
                "PPO did not beat the flat, zero-reward baseline on selection data; "
                "the existing meta-agent artifact was not replaced."
            )
            return False

        final_mean_reward, final_mean_length, final_episodes = _evaluate_all_episodes(
            candidate, final_env
        )
        logger.info(
            "PPO untouched final holdout report: episodes=%d mean_reward=%.5f "
            "mean_episode_steps=%.1f; this result does not gate saving",
            final_episodes,
            final_mean_reward,
            final_mean_length,
        )

        staged_prefix = Path(temp_dir) / "selected_meta_agent"
        candidate.save(str(staged_prefix))
        staged_agent = Path(f"{staged_prefix}.zip")
        if _base_checkpoint_hashes() != base_hashes:
            raise RuntimeError(
                "A base model changed while PPO was training; refusing to save "
                "a policy trained on stale walk-forward predictions."
            )
        os.replace(staged_agent, RL_MODEL_PATH)

    _write_schema(
        {
            "trading_schema_version": SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "observation_features": list(OBSERVATION_FEATURES),
            "action_contract": list(UNIFIED_ACTION_CONTRACT),
            "agent_sha256": sha256_file(RL_MODEL_PATH),
            "class_order": [-1, 0, 1],
            "base_checkpoint_sha256": base_hashes,
            "selection_mean_reward": selection_mean_reward,
            "selection_mean_episode_steps": selection_mean_length,
            "selection_episodes": selection_episodes,
            "final_holdout_mean_reward": final_mean_reward,
            "final_holdout_mean_episode_steps": final_mean_length,
            "final_holdout_episodes": final_episodes,
            "total_timesteps": total_timesteps,
            "seed": seed,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    logger.info("Saved selected PPO meta-agent to %s", RL_MODEL_PATH)
    return True
