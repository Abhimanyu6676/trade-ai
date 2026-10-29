"""Load application environment settings and expose them as JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPLICATION_ENVIRONMENT_VARIABLES = (
    "OPENALGO_API_KEY",
    "OPENALGO_HOST",
    "TRADE_AI_CHAT_ORIGINS",
)


def load_environment() -> dict[str, str]:
    """Load the repository-root ``.env`` and return configured app settings.

    Existing process environment values take precedence over values in ``.env``.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return {
        name: os.environ[name]
        for name in APPLICATION_ENVIRONMENT_VARIABLES
        if name in os.environ
    }


def validate_environment() -> dict[str, str]:
    """Load settings and fail if any declared application setting is blank."""
    settings = load_environment()
    missing = [
        name
        for name in APPLICATION_ENVIRONMENT_VARIABLES
        if not settings.get(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "Missing required application environment variables: "
            + ", ".join(missing)
        )
    return settings


def get_environment_json() -> str:
    """Return configured application settings as a JSON object string."""
    return json.dumps(load_environment())
