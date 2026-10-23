"""OpenAlgo analyser status endpoint for browser clients."""

from typing import Any

from fastapi import APIRouter

from trade_ai.openalgo_client import openalgo

router = APIRouter()


@router.get("/analyser-status")
def get_analyser_status() -> dict[str, Any]:
    """Return a small, credential-free summary of the OpenAlgo analyser state."""
    try:
        response = openalgo.get_analyzer_status()
    except Exception:
        return {
            "available": False,
            "mode": None,
            "analyze_mode": None,
            "message": "Unable to connect to the OpenAlgo analyser status service.",
        }

    successful = isinstance(response, dict) and response.get("status") == "success"
    data = response.get("data", {}) if isinstance(response, dict) else {}
    if not isinstance(data, dict):
        data = {}

    return {
        "available": successful,
        "mode": data.get("mode"),
        "analyze_mode": data.get("analyze_mode"),
        "message": "OpenAlgo analyser is online."
        if successful
        else "OpenAlgo responded, but analyser status is unavailable.",
    }
