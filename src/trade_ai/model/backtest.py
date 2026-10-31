"""Run an out-of-sample backtest or extended evaluation for the PPO meta-agent.

Supports loading a 3-month dataset (1-month warm-up buffer + 2-month backtest window),
running features and base model predictions, initializing MetaTradingEnv, and
evaluating the trained PPO meta-agent checkpoint.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from trade_ai.config.trading_config import RL_MODEL_PATH
from trade_ai.logger import logger
from trade_ai.model.training.trading_env import MetaTradingEnv
from trade_ai.model.training.train_rl import _evaluate_all_episodes


def run_backtest(
    data_path: str | Path,
    target_start_date: str,
    model_path: str | Path = RL_MODEL_PATH,
) -> dict[str, float | int]:
    """Execute backtest over a specified time window using a warm-up buffer.

    Parameters:
    - data_path: Path to CSV containing raw OHLCV and feature data (e.g., 3 months total).
    - target_start_date: The cutoff date ('YYYY-MM-DD') where the actual 2-month
      backtest evaluation starts (data before this is used as the indicator warm-up buffer).
    - model_path: Path to the trained PPO meta-agent zip file.
    """
    data_path = Path(data_path)
    model_path = Path(model_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"Backtest dataset not found: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained PPO meta-agent checkpoint not found: {model_path}"
        )

    logger.info("Loading backtest dataset from %s", data_path)
    df = pd.read_csv(data_path)

    # Ensure timestamp parsing and chronological sorting
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["ticker", "timestamp"], kind="stable").reset_index(drop=True)
    df["session_date"] = df["timestamp"].dt.date.astype(str)

    # Slice the DataFrame to keep only the target backtest window (dropping the warm-up month)
    logger.info(
        "Applying warm-up filter: evaluating sessions from %s onward", target_start_date
    )
    eval_df = df.loc[df["session_date"] >= target_start_date].copy()

    if eval_df.empty:
        raise ValueError(
            f"Evaluation dataset is empty after filtering for start date >= {target_start_date}"
        )

    # Initialize Environment and Load PPO Policy
    logger.info("Initializing MetaTradingEnv for backtest evaluation...")
    env = MetaTradingEnv(eval_df)

    logger.info("Loading trained PPO meta-agent from %s", model_path)
    policy = PPO.load(model_path, env=env)

    # Run Evaluation using your existing robust function
    logger.info(
        "Running deterministic backtest evaluation across %d episodes...",
        len(env.episodes),
    )
    mean_reward, mean_length, total_episodes = _evaluate_all_episodes(policy, env)

    # Detailed trade tracking across episodes
    total_trades = 0
    for episode_index in range(len(env.episodes)):
        observation, _ = env.reset(options={"episode_index": episode_index})
        terminated = truncated = False
        in_trade = False

        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(
                int(np.asarray(action).reshape(-1)[0])
            )

            # Count when a new non-zero position is opened
            if info["position"] != 0.0 and not in_trade:
                total_trades += 1
                in_trade = True
            elif info["position"] == 0.0:
                in_trade = False

    results = {
        "total_episodes": total_episodes,
        "mean_episode_reward": mean_reward,
        "mean_episode_length": mean_length,
        "total_trades_executed": total_trades,
    }

    logger.info("--- Backtest Results Summary ---")
    logger.info("Total Episodes Evaluated: %d", results["total_episodes"])
    logger.info("Mean Episode Reward: %.5f", results["mean_episode_reward"])
    logger.info("Mean Episode Steps: %.1f", results["mean_episode_length"])
    logger.info("Total Trades Executed: %d", results["total_trades_executed"])

    return results


if __name__ == "__main__":
    # Example usage:
    # Point this to your 3-month dataset CSV (1-month buffer + 2-month target window)
    DATASET_CSV = "path/to/your/3_month_combined_data.csv"
    TARGET_START = "2026-03-01"  # Start date of your 2-month backtest window

    run_backtest(data_path=DATASET_CSV, target_start_date=TARGET_START)
