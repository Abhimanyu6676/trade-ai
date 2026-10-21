"""Chat endpoint that parses a user's market request."""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from trade_ai.ai_parser import extract_market_intent

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="The user's message to parse.")


def parse_message(message: str) -> dict[str, str | int]:
    """Parse a chat message into a supported market action."""
    action, ticker, days_back = extract_market_intent(message)
    return {
        "action": action,
        "ticker": ticker,
        "days_back": days_back,
    }


@router.post("/chat")
def chat_post(request: ChatRequest) -> dict[str, str | int]:
    """Parse a chat message supplied in the request body."""
    return parse_message(request.message)


@router.get("/chat")
def chat_get(message: str = Query(min_length=1)) -> dict[str, str | int]:
    """Parse a chat message supplied as a query parameter."""
    return parse_message(message)
