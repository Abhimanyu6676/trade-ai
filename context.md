# Project Context & Architecture

## System Overview
- **Project Name:** Autonomous GRU Trading Pipeline
- **Target Hardware:** 4GB RAM VPS (Linux)
- **Broker / Data Layer:** OpenAlgo API (1-minute historical OHLCV data ingestion)
- **Core Model:** Gated Recurrent Unit (GRU) neural network handling 27 engineered technical features.
- **Loss Function:** Custom `ReturnWeightedBCELoss` blending direction prediction with profit magnitude weighting.

## Technical Stack & Optimizations
- **Data Pipeline:** 
  - Local caching via `tickData/` CSV directory to avoid redundant API pulls.
  - Automatic multi-chunk monthly historical fetching (2020–2026) with progress tracking and network retry logic.
- **Memory Management:**
  - Sliding sequence creation via list structures converted safely to tensors.
  - PyTorch `TensorDataset` and `DataLoader` batching (`batch_size=256`).
- **Checkpointing:**
  - Weights persistently stored at `.run_cache/gru_trading_model.pth`.
  - Automatic state verification and loading on startup.

## Current Milestone Status
- **Resolved Issues:** Fixed OOM memory crashes during large dataset training by introducing mini-batching; resolved `KeyError` casing exceptions via strict `standardize_columns` enforcement; added resilient per-stock try-except loops to gracefully skip incomplete API payloads (e.g., empty tickers).
- **Active Phase:** Model training execution and convergence monitoring (observing loss curves smoothly decreasing across epochs, e.g., dropping toward `0.0005`).

---
---

# Project Status Summary: Custom AI Stock Prediction Platform


## 1. System Architecture Overview
We have built a fully modular, decoupled **Model-View-Controller (MVC)** stock data collection and machine learning inference platform from scratch. The system runs on a **low-resource 4GB RAM Ubuntu vCPU VPS**. To optimize memory usage, we configured a **4GB swap space file** on the OS and implemented active tensor dimension bounds to prevent Out-of-Memory crashes. 

The application uses an **SSH Port Forwarding Tunnel (Port 7860)** to securely stream a private browser interface layout to the local development host machine without exposing ports to the public internet.

[ User Text Prompt ]
│
▼
[ app.py (UI Entry Point) ]
│
├──► [ ai_parser.py (Ollama Qwen 0.5B NLP Parser) ] ──► Returns Ticker & Days Back
│
└──► [ data_engine.py (Core Orchestrator) ]
│
├──► [ OpenAlgo Python SDK ] ──► Queries Remote Public HTTPS Domain
│
├──► [ features.py (Math Layer) ] ──► Computes Indicators (Pure Pandas Lists)
│
└──► [ model.py (PyTorch AI Neural Layer) ] ──► Ingests Tensors ──► Outputs Forecast

---

## 2. Active File Blueprint Configurations

### File 1: `ai_parser.py` (NLP Extraction Layer)
Uses an ultra-lightweight, local `qwen2.5:0.5b` model via Ollama to perform Named Entity Recognition (NER). It interprets unstructured text sentences and converts them into precise computer-readable JSON arguments using a static context baseline date anchor of **2026-10-20**.
* **RAM Optimization Ceilings:** Restricts parameters to `num_predict: 40` and `num_ctx: 512` to prevent vCPU spikes.

### File 2: `features.py` (Mathematical Technical Indicators)
Calculates core technical indicator matrices completely from scratch. 
* **`calculate_ema(df, period=9)`**: Computes the Exponential Moving Average baseline.
* **`calculate_rsi(df, period=14)`**: Computes Wilder's Relative Strength Index.
* **Critical Fix:** Uses native Python lists (`.tolist()`) to execute loops and smooth out inputs. This completely bypasses Pandas' silent Chained Indexing Assignment Protection and NumPy's read-only view locks, which previously caused calculations to crash on 1-minute timeframes and 30-day lookbacks.

### File 3: `model.py` (Proprietary PyTorch Machine Learning Brain)
Defines your custom predictive neural network architecture compiled from scratch using CPU-optimized `torch` tensors.
* **Architecture:** Utilizes a **Gated Recurrent Unit (GRU)** sequence layer (input_dim=3 processing Close, EMA_9, RSI_14) mapped to a fully connected dense linear layer (`nn.Linear`) and a `Sigmoid` activation function classification head.
* **Output:** Produces a floating probability map coordinates where values `≥ 0.50` generate a **Bullish Forecast Long Signal**, and values `< 0.50` generate a **Bearish Forecast Short Signal**.

### File 4: `data_engine.py` (Pipeline Coordinator)
Initializes your secure, cross-server public OpenAlgo domain SDK connection (`https://yourdomain.com`).
* **Process Flow:** Connects to the SDK function, handles timezone-aware timestamp index extractions, forces uniform capitalized column conversions (`Timestamp`, `Open`, `High`, `Low`, `Close`, `Volume`), truncates tables to a maximum row footprint layout boundary of **5,000 rows** to safeguard RAM, runs your custom indicators math, and passes the sequence matrix arrays straight through a forward pass of your custom PyTorch GRU model layers.

### File 5: `app.py` (Web Dashboard Interface View)
Serves strictly as the entry controller point and Gradio rendering interface framework layout. Displays text prompt bars, parameter drop-down selectors (Interval, Exchange), and dynamic text monitor widgets displaying **Active Stock**, **Active Exchange**, **Custom PyTorch AI Forecast Signal**, and **Neural Classification Confidence %** right above the candle data grid view table.

---

## 3. Environment Workspace States
* **OS Distribution Target:** Ubuntu Linux VPS (vCPU only, 4GB RAM + 4GB Active Swap File).
* **Python Environment Target:** Virtual isolation layer named `(aliens)` located at `/home/ubuntu/scratch-ai/aienv/`.
* **Core Libraries Installed:** `gradio`, `ollama`, `pandas`, `openalgo`, `torch` (CPU wheel distribution release).
* **Current Operational Status:** **100% Stable.** Natural prompt messages are accurately parsed, data streams pull smoothly across multi-timeframe configurations (1m, 1h, D) over a 30-day window, and indicator values populate without memory faults or read-only crashes.

---

## 4. Context Rules
always provide summarized context of work done, changes and progress in markdown, so I can provide the context to AI agent in future session and we don’t have to start from beginning, and we have idea where we left of

## 5. Next Step Strategy Horizons
When returning to this codebase configuration state, select one of the following development horizons to continue building the algorithmic system:
1. **`train.py` Core Math Engine:** Write your own backpropagation learning loop script from scratch. This includes implementing a loss function function matrix (`BCELoss`), an optimizer (`Adam`), and loading historic training sets to adjust your `model.py` GRU weights so it can accurately forecast real market movements based on historical indicator trends.
2. **Automated Order Execution Module:** Connect your PyTorch AI output forecasting logic straight to an execution pipeline file that will dynamically pass `client.place_order()` packets back to your live broker through OpenAlgo whenever a high-confidence trading signal is generated.

## 6. Last Gemini chat suggestions
3. Should we write a dedicated script file named train.py to build the mathematical training loop (loss functions, optimization weights, backpropagation math) so your model can learn from historical data?
4. Or should we connect this to an automated execution file that automatically places live orders to your broker via OpenAlgo whenever a Buy or Sell signal is triggered?

---

Project Context: Custom AI Stock Prediction & Execution Platform

​1. System Architecture & Setup Baseline

​Environment: Low-resource 4GB RAM Ubuntu vCPU VPS with a 4GB swap file on OS disk storage to prevent Out-of-Memory (OOM) crashes.
​Access/UI: Gradio web interface served via an SSH Port Forwarding Tunnel (Port 7860). Python venv (aliens) located at /home/ubuntu/scratch-ai/aienv/.
​Architecture Pattern: Decoupled, modular Model-View-Controller (MVC).
​app.py → Web UI entry controller.
​ai_parser.py → Local Ollama qwen2.5:0.5b model for NLP query parsing (Entity extraction with context anchor 2026-10-20; constrained via num_predict: 40, num_ctx: 512).
​data_engine.py → SDK interface to OpenAlgo domain ([https://yourdomain.com](https://yourdomain.com)), table truncations to 5,000 rows max to protect RAM, and pipeline management.
​features.py → Math layer running native Python list loops (.tolist()) to compute indicators, completely bypassing Pandas chained indexing and NumPy read-only view locks.
​model.py → PyTorch GRU neural architecture outputting a classification probability map (\ge 0.50 = Bullish Long, < 0.50 = Bearish Short).
​2. Updated Strategy & Mathematical Specification

​The primary objective is an Intraday Multi-Timeframe Confluence & Trend Filtering Strategy operating on a live 1-minute data feed.

​Core Logic Requirements:

​Macro Trend Gate (10-Day Moving Average):
​Calculate a 10-day daily moving average.
​If the macro average change has already reversed/flattened against the trade direction, block all trades. Only trade in the direction of the dominant 10-day macro trend.
​Multi-Timeframe Resampling (1m, 5m, 30m):
​Fetch/stream 1-minute live data.
​On-the-fly resample 1-minute candles into 5-minute and 30-minute aggregated dataframes inside features.py.
​Multi-EMA Stack & Crossover Alignment:
​Calculate EMA 9, EMA 21, and EMA 50 across all three timeframes (1m, 5m, 30m).
​Long Execution Trigger: Requires 30m (EMA 9 > 21 > 50) + 5m (EMA 9 > 21 > 50) + 1m EMA 9 crossing above EMA 21.
​AI Signal Validation:
​Pass the aligned multi-timeframe feature matrix into the PyTorch GRU model.
​Only accept trade signals when the model output meets high confidence thresholds.
​3. Persistence & Memory Safeguards

​Volatility Management: VPS reboots wipe active RAM. Learned neural weights must be written to disk (.pth checkpoint files via torch.save()) and loaded on startup (torch.load()).
​Training Architecture: Sequential (stock-by-stock) batch learning loop to keep memory usage under the 4GB ceiling when training across deep historical data.
​4. Next Step Implementation Roadmap

​Expand features.py: Write native resampling logic to derive 5m and 30m bars from 1m data, calculate EMA 9, 21, 50 across all tiers, and calculate the 10-day daily moving average.
​Build Strategy Decision Gate: Create a multi-timeframe confluence validation function (validate_trade_confluence()).
​Develop train.py: Build the sequential backpropagation training script using BCELoss and Adam optimizer with persistent .pth checkpointing.
​Connect Execution Module: Integrate high-confidence validated signals into OpenAlgo’s client.place_order() execution payloads.
​Would you like to start by coding the updated features.py file with the multi-timeframe resampling and EMA 9/21/50 calculations?
