"""Enum-backed stock market-cap and sector metadata."""

from enum import Enum
from typing import TypedDict

import pandas as pd


class MarketCap(str, Enum):
    """Supported market-cap categories for stock metadata."""

    LARGE = "large"
    MID = "mid"
    SMALL = "small"


class Sector(str, Enum):
    """Supported sector categories for stock metadata."""

    BANKING = "banking"
    IT = "it"
    AUTO = "auto"
    ENERGY = "energy"
    CONSUMER = "consumer"
    METALS = "metals"
    OTHER = "other"


class StockMetadata(TypedDict):
    cap: MarketCap
    sector: Sector


STOCK_METADATA: dict[str, StockMetadata] = {
    # Large Caps
    "RELIANCE": {"cap": MarketCap.LARGE, "sector": Sector.ENERGY},
    "TCS": {"cap": MarketCap.LARGE, "sector": Sector.IT},
    "INFY": {"cap": MarketCap.LARGE, "sector": Sector.IT},
    "SBIN": {"cap": MarketCap.LARGE, "sector": Sector.BANKING},
    "HDFCBANK": {"cap": MarketCap.LARGE, "sector": Sector.BANKING},
    "TMPV": {"cap": MarketCap.LARGE, "sector": Sector.AUTO},
    # "HINDZINC": {"cap": MarketCap.LARGE, "sector": Sector.METALS},
    # Mid Caps
    "FEDERALBNK": {"cap": MarketCap.MID, "sector": Sector.BANKING},
    "COFORGE": {"cap": MarketCap.MID, "sector": Sector.IT},
    "VOLTAS": {"cap": MarketCap.MID, "sector": Sector.CONSUMER},
    # Small Caps
    "SUZLON": {"cap": MarketCap.SMALL, "sector": Sector.ENERGY},
    "KPITTECH": {"cap": MarketCap.SMALL, "sector": Sector.IT},
}

DEFAULT_STOCK_METADATA: StockMetadata = {
    "cap": MarketCap.MID,
    "sector": Sector.OTHER,
}

def _category_label(category: Enum) -> str:
    """Make a readable feature suffix while preserving short acronyms like IT."""
    return "".join(
        part if len(part) <= 2 else part.capitalize()
        for part in category.name.split("_")
    )


def _category_feature_name(prefix: str, category: Enum) -> str:
    return f"{prefix}_{_category_label(category)}"


CAP_FEATURE_COLUMNS = tuple(
    _category_feature_name("Cap", category) for category in MarketCap
)
SECTOR_FEATURE_COLUMNS = tuple(
    _category_feature_name("Sector", category) for category in Sector
)
METADATA_FEATURE_COLUMNS = (*CAP_FEATURE_COLUMNS, *SECTOR_FEATURE_COLUMNS)


def encode_categorical_metadata(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add one-hot columns generated from the market-cap and sector enums."""
    metadata = STOCK_METADATA.get(ticker.upper(), DEFAULT_STOCK_METADATA)
    cap = metadata["cap"]
    sector = metadata["sector"]

    if not isinstance(cap, MarketCap) or not isinstance(sector, Sector):
        raise TypeError("Stock metadata cap and sector must use their defined enums")

    for category in MarketCap:
        feature_name = _category_feature_name("Cap", category)
        df[feature_name] = 1.0 if cap is category else 0.0

    for category in Sector:
        feature_name = _category_feature_name("Sector", category)
        df[feature_name] = 1.0 if sector is category else 0.0

    return df
