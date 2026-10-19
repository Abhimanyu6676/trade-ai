import json

import ollama

CURRENT_DATE = "2026-10-20"
TARGET_MODEL = "qwen2.5:0.5b"


def extract_market_intent(user_prompt: str):
    """Uses a local 0.5B LLM to extract financial parameters from unstructured text."""
    system_instruction = f"""
    You are a financial parsing engine. Today's current date is exactly {CURRENT_DATE}.
    Extract the stock ticker and lookback days from the user text.
    Return ONLY a raw JSON object. Do not wrap it in markdown code blocks.
    
    JSON Template structure:
    {{
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
            options={"temperature": 0.0, "num_predict": 40, "num_ctx": 512},
        )
        data = json.loads(response["response"].strip())
        return data.get("ticker", "SBIN"), data.get("days_back", 7)
    except Exception as e:
        print(f"[AI Module Error] Falling back to defaults: {e}")
        return "SBIN", 7
