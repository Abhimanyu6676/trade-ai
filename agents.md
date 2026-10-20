# AI Collaborator Agent Guidelines & Roles

## Core Objective
Maintain, optimize, and execute an autonomous, memory-efficient AI trading pipeline designed for resource-constrained environments (4GB RAM VPS). The system integrates **OpenAlgo** for data ingestion, a custom **ReturnWeightedBCELoss** optimization function, and a **GRU neural network** for real-time equity market prediction.

## Operational Directives
1. **Memory Efficiency First:** 
   - Never load entire multi-year 1-minute historical datasets (e.g., 400k+ row CSVs) into PyTorch tensors all at once. 
   - Always enforce mini-batching via PyTorch `DataLoader` (`batch_size=256`) and use garbage collection (`gc.collect()`) between iterations to prevent OS-level OOM ("killed") crashes.
2. **Robust Data Hygiene:**
   - Always apply the `standardize_columns()` utility to incoming broker or cached CSV payloads to normalize OHLCV casing (`Open`, `High`, `Low`, `Close`, `Volume`, `Timestamp`).
3. **Fault Tolerance:**
   - Wrap network and file operations in resilient `try...except` blocks with retry mechanisms (e.g., 3-attempt delays for OpenAlgo API calls) to ensure pipeline continuity if a single ticker or chunk fails.
4. **Resilient State Management:**
   - Save model checkpoints (`.pth` files) periodically after every successful training epoch inside `.run_cache/` to ensure seamless resumption or live inference deployment.