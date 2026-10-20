import json

import ollama

CURRENT_DATE = "2026-10-20"
TARGET_MODEL = "qwen2.5:0.5b"


def extract_market_intent(user_prompt: str):
    """Uses a local 0.5B LLM to extract financial parameters or detect status requests."""
    system_instruction = f"""
    You are a financial parsing engine. Today's current date is exactly {CURRENT_DATE}.
    Determine if the user wants to check 'openalgo_status' (e.g., checking analyzer status, connection, or sandbox mode) or perform 'market_analysis'.
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
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
        print(f"[AI Module Error] Falling back to defaults: {e}")
        return "market_analysis", "SBIN", 7
