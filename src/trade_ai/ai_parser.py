import datetime
import json
import re

import ollama

CURRENT_DATE = datetime.date.today().isoformat()
TARGET_MODEL = "qwen2.5:0.5b"
RESET_BRAIN_PATTERN = re.compile(
    r"\b(?:reset|clear|forget|restart)\b.{0,40}\b(?:brain|model|weights|checkpoint)\b"
    r"|\bstart\s+(?:as\s+)?(?:a\s+)?(?:new|fresh)(?:\s+(?:brain|model))?\b",
    re.IGNORECASE,
)
START_TRAINING_PATTERN = re.compile(
    r"\b(?:start|begin|run)\s+(?:the\s+)?training\b"
    r"|\btrain\s+(?:the\s+)?(?:model|brain)\b",
    re.IGNORECASE,
)


def extract_market_intent(user_prompt: str):
    """
    Detect reset requests directly, otherwise use a local 0.5B LLM to determine
    the requested action and extract the ticker symbol and lookback days.
    """
    if RESET_BRAIN_PATTERN.search(user_prompt):
        return "reset_brain", "", 0
    if START_TRAINING_PATTERN.search(user_prompt):
        return "start_training", "", 0

    system_instruction = f"""
    You are a financial parsing engine. Today's current date is exactly {CURRENT_DATE}.
    Determine if the user wants to check 'openalgo_status' (e.g., checking analyzer status, connection, or bridge) or perform 'market_analysis'.
    Also extract the stock ticker symbol and lookback days.
    Return ONLY a raw JSON object. Do not wrap it in markdown code blocks.
    
    JSON Template structure:
    {{
      "action": "market_analysis" or "openalgo_status",
      "ticker": "UPPERCASE_SYMBOL",
      "days_back": integer
    }}
    """
    try:
        response = ollama.generate(
            model=TARGET_MODEL,
            system=system_instruction,
            prompt=user_prompt,
            format="json",
            options={"temperature": 0.0, "num_predict": 50, "num_ctx": 512},
        )
        data = json.loads(response["response"].strip())
        action = data.get("action", "market_analysis")
        ticker = data.get("ticker", "SBIN")
        days_back = data.get("days_back", 7)
        return action, ticker, days_back
    except Exception as e:
        print(f"[AI Parser Error] Falling back to defaults: {e}")
        return "market_analysis", "SBIN", 7
