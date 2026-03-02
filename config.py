# config.py
SYMBOLS = ["ES", "NQ", "ZN", "GC", "HG", "SI", "DX", "6E", "CL"]

PERIODS = {
    "2Y": 2,
    "3Y": 3,
    "5Y": 5,
    "10Y": 10,
}

# map to TradeStation / yfinance symbols
TS_SYMBOL_MAP = {
    "ES": "ES",      # E‑mini S&P 500
    "NQ": "NQ",
    "ZN": "ZN",      # 10‑year T‑Note
    "GC": "GC",      # Gold
    "HG": "HG",      # Copper
    "SI": "SI",      # Silver
    "DX": "DXY",     # US Dollar Index
    "6E": "6E",      # EUR/USD
    "CL": "CL",      # Crude Oil
}

# yfinance tickers (often same as TS, but adjusted for FX)
# e.g., "EURUSD=X" for 6E, "DX‑USD" for DX, etc.
YFINANCE_TICKER_MAP = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "ZN": "ZN=F",
    "GC": "GC=F",
    "HG": "HG=F",
    "SI": "SI=F",
    "DX": "DX=F",    # often not available; fallback: use index or synthetic
    "6E": "EURUSD=X",
    "CL": "CL=F",
}

# main futures session (US Eastern time)
FUTURES_SESSION = ("09:30", "16:00")  # can be tuned per symbol

# event filters (NFP, Triple Witching, etc.)
EVENT_FILTERS = [
    "NFP",
    "Triple Witching",
    # "Fed Meeting", ...
]