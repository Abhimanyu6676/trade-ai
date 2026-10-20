## Environment setup

1. initial setup by command
```
uv init --name trade-ai
```

2. always install the libraries in same venv

## Error handling

1. If no exec error is thrown
```
# 1. Ensure your local execution cache exists (bypasses noexec VPS restrictions)
mkdir -p .run_cache
export TMPDIR=$(pwd)/.run_cache
export PIP_CACHE_DIR=$(pwd)/.run_cache

# 2. Launch the application
uv run python src/trade_ai/app.py
```
