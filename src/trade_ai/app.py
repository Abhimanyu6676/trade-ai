"""Main trade-ai process: load the brain and serve the API."""

import logging

from trade_ai.model.models import get_model_manager
from trade_ai.server.server import run_server

logger = logging.getLogger(__name__)


def main() -> None:
    """Load prediction models, then run FastAPI in this process."""
    get_model_manager()
    logger.info("Trade AI brain loaded; API server is running on port 3010.")
    run_server()


if __name__ == "__main__":
    main()
