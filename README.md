# Autonomous AI Trading Pipeline (4GB VPS Optimized)

A lightweight, high-performance algorithmic trading and deep learning framework built for resource-constrained production environments. It trains custom GRU neural networks on multi-year historical minute data fetched through OpenAlgo, prioritizing financial expectancy over raw directional accuracy.

---

## Features
- **Memory-Safe Training:** Processes hundreds of thousands of historical candle rows via optimized mini-batch loaders (`batch_size=256`), preventing Linux OOM process terminations on 4GB VPS setups.
- **OpenAlgo Integration:** Seamlessly fetches, caches, and standardizes multi-year 1-minute historical datasets (`tickData/`).
- **Financial Expectancy Loss (`ReturnWeightedBCELoss`):** Weights classification accuracy based on actual forward return magnitudes.
- **Fault-Tolerant Execution:** Built-in retry loops and per-ticker exception handling ensure uninterrupted training batches.
- **Incremental Checkpointing:** Automatically saves model weights after each epoch to `.run_cache/gru_trading_model.pth`.

---

## Project Structure
```text
├── .run_cache/                             # Model state checkpoints (.pth)
├── tickData/                               # Cached historical 1-minute CSV datasets
├── src/
│   └── trade_ai/
│       ├── __init__.py
│       ├── ai_parser.py                    # NLP parser
│       ├── app.py                          # Main app
│       ├── data_engine.py                  # Data fetching from openalgo & manipulation
│       ├── features.py                     # Feature engineering & normalization utilities
│       ├── model.py                        # Main model
│       └── train.py                        # Main training script & batched data loader
|
├── app.py                                  # Live inference application / UI launcher
├── requirements.json /.txt                 # Project dependencies
└── README.md