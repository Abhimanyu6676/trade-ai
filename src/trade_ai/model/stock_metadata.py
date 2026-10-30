"""Stable categorical codes for stock market-cap and sector metadata."""

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
    INFRASTRUCTURE = "infrastructure"
    FINANCIAL_SERVICES = "financial_services"
    PHARMACEUTICALS = "pharmaceuticals"
    TELECOM = "telecom"
    INDUSTRIALS = "industrials"
    CHEMICALS = "chemicals"


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
    "ADANIPORTS": {"cap": MarketCap.LARGE, "sector": Sector.INFRASTRUCTURE},
    "ADANIPOWER": {"cap": MarketCap.LARGE, "sector": Sector.ENERGY},
    "JIOFIN": {"cap": MarketCap.LARGE, "sector": Sector.FINANCIAL_SERVICES},
    # Jio Platforms listed 2026-10-28, after the AMFI Jan-Jun 2026 window;
    # classify large from its listed market cap. OpenAlgo/NSE symbol is JPL.
    "JPL": {"cap": MarketCap.LARGE, "sector": Sector.TELECOM},
    "AXISBANK": {"cap": MarketCap.LARGE, "sector": Sector.BANKING},
    "SUNPHARMA": {"cap": MarketCap.LARGE, "sector": Sector.PHARMACEUTICALS},
    "TATASTEEL": {"cap": MarketCap.LARGE, "sector": Sector.METALS},
    "ONGC": {"cap": MarketCap.LARGE, "sector": Sector.ENERGY},
    "IOC": {"cap": MarketCap.LARGE, "sector": Sector.ENERGY},
    "GAIL": {"cap": MarketCap.LARGE, "sector": Sector.ENERGY},
    # "BHAIL" did not resolve in OpenAlgo; BHEL is the verified NSE symbol.
    "BHEL": {"cap": MarketCap.LARGE, "sector": Sector.INDUSTRIALS},
    "POWERGRID": {"cap": MarketCap.LARGE, "sector": Sector.ENERGY},
    "ITC": {"cap": MarketCap.LARGE, "sector": Sector.CONSUMER},
    "WIPRO": {"cap": MarketCap.LARGE, "sector": Sector.IT},
    "ADANIGREEN": {"cap": MarketCap.LARGE, "sector": Sector.ENERGY},
    "MOTHERSON": {"cap": MarketCap.LARGE, "sector": Sector.AUTO},
    "LT": {"cap": MarketCap.LARGE, "sector": Sector.INDUSTRIALS},
    "CIPLA": {"cap": MarketCap.LARGE, "sector": Sector.PHARMACEUTICALS},
    "ETERNAL": {"cap": MarketCap.LARGE, "sector": Sector.CONSUMER},
    "HINDALCO": {"cap": MarketCap.LARGE, "sector": Sector.METALS},
    # "HINDZINC": {"cap": MarketCap.LARGE, "sector": Sector.METALS},
    # Mid Caps
    "YESBANK": {"cap": MarketCap.MID, "sector": Sector.BANKING},
    "AUROPHARMA": {"cap": MarketCap.MID, "sector": Sector.PHARMACEUTICALS},
    "KALYANKJIL": {"cap": MarketCap.MID, "sector": Sector.CONSUMER},
    "HINDCOPPER": {"cap": MarketCap.MID, "sector": Sector.METALS},
    "ZYDUSLIFE": {"cap": MarketCap.MID, "sector": Sector.PHARMACEUTICALS},
    "GLENMARK": {"cap": MarketCap.MID, "sector": Sector.PHARMACEUTICALS},
    "OIL": {"cap": MarketCap.MID, "sector": Sector.ENERGY},
    "FEDERALBNK": {"cap": MarketCap.MID, "sector": Sector.BANKING},
    "COFORGE": {"cap": MarketCap.MID, "sector": Sector.IT},
    "VOLTAS": {"cap": MarketCap.MID, "sector": Sector.CONSUMER},
    # Small Caps
    "BANDHANBNK": {"cap": MarketCap.SMALL, "sector": Sector.BANKING},
    "AARTIIND": {"cap": MarketCap.SMALL, "sector": Sector.CHEMICALS},
    "RBLBANK": {"cap": MarketCap.SMALL, "sector": Sector.BANKING},
    "JUBLFOOD": {"cap": MarketCap.SMALL, "sector": Sector.CONSUMER},
    "POWERMECH": {"cap": MarketCap.SMALL, "sector": Sector.INDUSTRIALS},
    "CENTURYPLY": {"cap": MarketCap.SMALL, "sector": Sector.INDUSTRIALS},
    "SUZLON": {"cap": MarketCap.SMALL, "sector": Sector.ENERGY},
    "KPITTECH": {"cap": MarketCap.SMALL, "sector": Sector.IT},
}

DEFAULT_STOCK_METADATA: StockMetadata = {
    "cap": MarketCap.MID,
    "sector": Sector.OTHER,
}

# These integer identifiers are categorical values for LightGBM, not ordinal
# measurements. Keep existing codes fixed and append new categories with new
# codes so adding a sector never changes the model's input width or old codes.
CAP_CATEGORY_CODES: dict[MarketCap, int] = {
    MarketCap.LARGE: 0,
    MarketCap.MID: 1,
    MarketCap.SMALL: 2,
}
SECTOR_CATEGORY_CODES: dict[Sector, int] = {
    Sector.BANKING: 0,
    Sector.IT: 1,
    Sector.AUTO: 2,
    Sector.ENERGY: 3,
    Sector.CONSUMER: 4,
    Sector.METALS: 5,
    Sector.OTHER: 6,
    Sector.INFRASTRUCTURE: 7,
    Sector.FINANCIAL_SERVICES: 8,
    Sector.PHARMACEUTICALS: 9,
    Sector.TELECOM: 10,
    Sector.INDUSTRIALS: 11,
    Sector.CHEMICALS: 12,
}
METADATA_FEATURE_COLUMNS = ("Cap", "Sector")


def encode_categorical_metadata(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add stable integer codes for LightGBM categorical metadata inputs."""
    metadata = STOCK_METADATA.get(ticker.upper(), DEFAULT_STOCK_METADATA)
    cap = metadata["cap"]
    sector = metadata["sector"]

    if not isinstance(cap, MarketCap) or not isinstance(sector, Sector):
        raise TypeError("Stock metadata cap and sector must use their defined enums")

    df["Cap"] = CAP_CATEGORY_CODES[cap]
    df["Sector"] = SECTOR_CATEGORY_CODES[sector]
    return df
