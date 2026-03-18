from dataclasses import dataclass
from typing import List

@dataclass
class StockListing:
    symbol: str
    name: str
    exchange: str
    sector: str
    market_cap_tier: str   # large, mid, small
    price: float
    tags: List[str]

CATALOG: List[StockListing] = [
    StockListing("AAPL",  "Apple Inc.",               "NASDAQ", "Technology",    "large", 178.5,  ["tech","consumer","smartphone"]),
    StockListing("MSFT",  "Microsoft Corporation",    "NASDAQ", "Technology",    "large", 420.1,  ["tech","cloud","enterprise"]),
    StockListing("GOOGL", "Alphabet Inc.",             "NASDAQ", "Technology",    "large", 165.2,  ["tech","advertising","ai"]),
    StockListing("AMZN",  "Amazon.com Inc.",           "NASDAQ", "Consumer",      "large", 185.0,  ["ecommerce","cloud","retail"]),
    StockListing("TSLA",  "Tesla Inc.",                "NASDAQ", "Automotive",    "large", 172.8,  ["ev","energy","auto"]),
    StockListing("NVDA",  "NVIDIA Corporation",        "NASDAQ", "Technology",    "large", 875.0,  ["ai","gpu","semiconductors"]),
    StockListing("META",  "Meta Platforms Inc.",       "NASDAQ", "Technology",    "large", 502.3,  ["social","advertising","vr"]),
    StockListing("JPM",   "JPMorgan Chase",            "NYSE",   "Financials",    "large", 198.4,  ["bank","finance","dividends"]),
    StockListing("WMT",   "Walmart Inc.",              "NYSE",   "Consumer",      "large", 62.3,   ["retail","grocery","ecommerce"]),
    StockListing("V",     "Visa Inc.",                 "NYSE",   "Financials",    "large", 279.5,  ["payments","finance","dividends"]),
    StockListing("JNJ",   "Johnson & Johnson",         "NYSE",   "Healthcare",    "large", 152.1,  ["pharma","healthcare","dividends"]),
    StockListing("XOM",   "Exxon Mobil",               "NYSE",   "Energy",        "large", 118.6,  ["oil","energy","dividends"]),
    StockListing("AMD",   "Advanced Micro Devices",    "NASDAQ", "Technology",    "large", 172.4,  ["ai","gpu","semiconductors"]),
    StockListing("PLTR",  "Palantir Technologies",     "NYSE",   "Technology",    "mid",   24.5,   ["ai","data","government"]),
    StockListing("RIVN",  "Rivian Automotive",         "NASDAQ", "Automotive",    "mid",   11.2,   ["ev","auto","startup"]),
]
