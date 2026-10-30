read `/home/ubuntu/trading/trade-ai/src/trade_ai/model/training/gemini.md` first for context of next course of action.

and then below is the action prompt, read & execute


Here is a complete, structured instruction set and a ready-to-use prompt designed specifically for your coding agent. It organizes your codebase into separate long and short modular folders, centralizes your configuration parameters (like your 1% TP / 0.5% SL rules), sets up the RL environments, and introduces an interactive training_master.py wizard.


Plaintext
```csharp
src/
├── config/
│   └── trading_config.py        # Centralized P&L rules, schema version, paths
├── training/
│   ├── long/
│   │   ├── train_long_gru.py
│   │   ├── train_long_lgb.py
│   │   └── trading_env_long.py
│   └── short/
│       ├── train_short_gru.py
│       ├── train_short_lgb.py
│       └── trading_env_short.py
└── training_master.py           # Interactive CLI master script

```

# PROMPT

```markdown
You are an expert quantitative systems engineer. Your task is to refactor, reorganize, and expand our institutional algorithmic trading repository to support clean modular separation between Long and Short models, centralized configuration management, and a unified interactive training CLI wizard.

Please implement the following exact requirements across the codebase:

### 1. Centralized Configuration (`src/config/trading_config.py`)
- Create a configuration file that holds global system settings:
  - **Schema Version:** `SCHEMA_VERSION = 1` (Incrementing this invalidates all legacy model checkpoints).
  - **Trade Targets (Long Baseline):** 
    - Take Profit (TP): `+1.0%` relative to entry price.
    - Stop Loss (SL): `-0.5%` relative to entry price.
  - **Trade Targets (Short Dynamic Inversion):** 
    - Short models and labeling scripts must import these percentages from this centralized file and automatically invert them: Short TP = `-1.0%` relative to entry price; Short SL = `+0.5%` relative to entry price.
  - **Model Artifact Paths:** Define clear directories for checkpoints (`./models/long/` and `./models/short/`).

### 2. Directory Reorganization & Renaming
- Rename existing training scripts:
  - `train_gru.py` -> `src/training/long/train_long_gru.py`
  - `train_lgb.py` -> `src/training/long/train_long_lgb.py`
- Create the short training directory: `src/training/short/`
- Create short counterparts:
  - `src/training/short/train_short_gru.py`
  - `src/training/short/train_short_lgb.py`
- Ensure labeling scripts for short (`labels_short.py`) pull the inverted percentages directly from `trading_config.py`.

### 3. RL Meta-Agent Environments (`trading_env.py`)
- Implement separate RL training environments or modular config flags for Long and Short in their respective folders:
  - `src/training/long/trading_env_long.py`
  - `src/training/short/trading_env_short.py`
- **Environment State Space:** Observes frozen base model probabilities (`[P_lgb, P_gru]`), agreement metrics, market volatility (ATR), and portfolio state.
- **Environment Reward Function:** Rewards clean wins (matching the 1% / 0.5% targets) and penalizes drawdowns, high turnover, or trading during model disagreement.

### 4. Interactive Training Master Script (`src/training_master.py`)
- Build a command-line interactive wizard (supporting both CLI arguments and interactive prompts via `input()`) that asks:
  1. **Which mindset to train?** Options: `[long, short, or both]`
  2. **Which model component to train?** Options: `[gru, lgb, rl, or all]`
- Based on the user's selection, the script should dynamically route execution to the correct training scripts inside `src/training/long/` or `src/training/short/`, injecting the correct `SCHEMA_VERSION` checks.

- train_master.py should take two optional args first for which set to train long, short or both and is both by default & second is which model to train gru, lgb or RL[common for both long/short] and is all by default. no interactive shell selector options

- also add the training example command in training.md for foreground and background, attached or detached shell will logging in file for detached. just as we are doing it now

### 5. Validation & Safety Checks
- Ensure all scripts validate dataset schema versions before loading historical weights. If an artifact was saved under an older schema version, raise an explicit error prompting re-training.
- Keep all file I/O clean, typed, and fully documented with docstrings.
```

# labels

You will have two distinct sets of labels (one set for Long, one set for Short) because long trades and short trades are looking for opposite market mechanics.
Here is why they must be separate, even though they pull from the exact same centralized configuration rule:
1. Why They Are Separate Files (labels.py vs labels_short.py)
Long Labels (labels.py): Looks forward to see if price goes up by +1.0% before dropping down by -0.5%.
Short Labels (labels_short.py): Looks forward to see if price goes down by -1.0% before rising up by +0.5%.
Because they look for opposite directional outcomes in the price action, you need two separate labeling scripts (or a single script with a direction parameter) that generate independent target columns in your datasets.
2. How the Centralized Config Keeps It Clean
Even though you have two separate sets of labels, you do not have to hardcode percentages in multiple places. As outlined in your architecture, both labeling scripts pull from your centralized trading_config.py:
trading_config.py defines your baseline risk-reward: PROFIT_TARGET_PCT = 1.0 and STOP_LOSS_PCT = 0.5.
Long Labeling Script: Imports +1.0% as Upper Barrier and -0.5% as Lower Barrier.
Short Labeling Script: Imports those exact same variables but automatically inverts them (-1.0% for target profit downward, +0.5% for stop loss upward).
If you ever want to change your risk profile in the future (e.g., shifting from 1% / 0.5% to 1.5% / 0.75%), you only change it once in trading_config.py, and both your Long and Short training/labeling pipelines instantly pick up the new targets!

## use Centralized Config for labels & Keeps It Clean, use names labels_long.py & labels_short.py and keep them in respective folder

put everything related to short and long in their respective folders and extract comment code and put it in `training/util.py`


# models topology
The Total Model Count
You will have 5 models in total working together as a single strategy ensemble:
Long Entry LightGBM
Long Entry GRU
Short Entry LightGBM
Short Entry GRU
Unified RL Meta-Agent (sitting on top of all 4 base models to make the final execution decision: Long, Short, or Flat).
2. The Features & Schema Versioning
Same Features: Yes, all your base models (the LGBs and GRUs) share the same core multi-timeframe feature engineering pipeline (features.py), ensuring they look at the exact same normalized market data (RSI, ATR, EMA 50, MACD, FVG, and Liquidity Sweeps).
Schema Integration & Invalidation: Yes! As we set up in your centralized configuration (trading_config.py), you will track a global SCHEMA_VERSION.
If you ever modify your features (e.g., add a new indicator or change an input column), you simply increment SCHEMA_VERSION = 2.
Your training scripts and live inference pipelines will automatically check this version. If an older model checkpoint was saved under SCHEMA_VERSION = 1, the system will recognize the mismatch, throw an error, and require you to retrain your models so you never accidentally run live inference on stale or mismatched features.
