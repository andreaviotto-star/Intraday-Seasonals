# bot/config_bot.py
# All configurable parameters for the trading bot.
# Every secret is loaded from .env — never hard-code credentials here.

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# DRY-RUN MODE
# When True: all logic runs, signals are logged, Telegram alerts
# are sent — but NO real orders are submitted to TradeStation.
# Set to False only when you are ready to trade live.
# ─────────────────────────────────────────────────────────────
DRY_RUN: bool = True

# ─────────────────────────────────────────────────────────────
# SYMBOLS TO WATCH  (subset of config.SYMBOLS)
# ─────────────────────────────────────────────────────────────
WATCHED_SYMBOLS: list = ["ES", "NQ"]

# How many top seasonal entry slots to schedule per symbol
TOP_SLOTS_PER_SYMBOL: int = 3

# ─────────────────────────────────────────────────────────────
# SIGNAL WEIGHTS  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────
WEIGHTS: dict = {
    "seasonal":  0.40,   # historical edge from seasonal_stats
    "sentiment": 0.25,   # news sentiment via OpenAI
    "gamma":     0.20,   # 0DTE gamma exposure via Tradier
    "volume":    0.15,   # bar-range surge (volume proxy)
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# ─────────────────────────────────────────────────────────────
# SIGNAL THRESHOLDS
# ─────────────────────────────────────────────────────────────
# Minimum composite score to fire a Telegram alert
MIN_COMPOSITE_SCORE: float = 0.55

# Seconds to wait for /approve or /reject on Telegram (5 minutes)
APPROVAL_TIMEOUT_SEC: int = 300

# Years of historical data to use for seasonal analysis
LOOKBACK_YEARS: int = 3

# ─────────────────────────────────────────────────────────────
# SEASONAL EDGE FILTERS  (mirrors app.py slider defaults)
# ─────────────────────────────────────────────────────────────
MIN_WIN_RATE: float       = 55.0
MIN_PROFIT_FACTOR: float  = 1.2
MAX_P_VALUE: float        = 0.05
MIN_OBS: int              = 30

# ─────────────────────────────────────────────────────────────
# RISK RULES  (non-negotiable)
# ─────────────────────────────────────────────────────────────
# Stop-loss placed at STOP_LOSS_MULTIPLIER × std-of-avg-path from entry
STOP_LOSS_MULTIPLIER: float = 1.0

# Never open a new trade if this many are already open
MAX_OPEN_TRADES: int = 1

# ─────────────────────────────────────────────────────────────
# DEFAULT ORDER QUANTITIES  (contracts per symbol)
# ─────────────────────────────────────────────────────────────
DEFAULT_QTY: dict = {
    "ES": 1,
    "NQ": 1,
    "GC": 1,
    "CL": 1,
    "ZN": 1,
    "6E": 1,
    "HG": 1,
    "SI": 1,
    "DX": 1,
}

# ─────────────────────────────────────────────────────────────
# TRADESTATION API CREDENTIALS
# ─────────────────────────────────────────────────────────────
TS_CLIENT_ID: str      = os.getenv("TRADESTATION_CLIENT_ID", "")
TS_CLIENT_SECRET: str  = os.getenv("TRADESTATION_CLIENT_SECRET", "")
TS_REFRESH_TOKEN: str  = os.getenv("TRADESTATION_REFRESH_TOKEN", "")

# Account ID used for placing orders (different from API credentials)
TS_ACCOUNT_ID: str     = os.getenv("TRADESTATION_ACCOUNT_ID", "")

# TradeStation REST base URL
TS_BASE_URL: str = "https://api.tradestation.com/v3"

# Continuous-contract symbol map for TradeStation (data + orders)
# NOTE: For live order placement TS may require the front-month
# contract (e.g. "@ESM25"). Override via TRADESTATION_SYMBOL_OVERRIDE
# in your .env if market orders reject. Format: "ES=@ESM25,NQ=@NQM25"
TS_SYMBOL_MAP: dict = {
    "ES": "@ES", "NQ": "@NQ", "ZN": "@TY",
    "GC": "@GC", "HG": "@HG", "SI": "@SI",
    "DX": "@DX", "6E": "@EC", "CL": "@CL",
}
_override_raw = os.getenv("TRADESTATION_SYMBOL_OVERRIDE", "")
if _override_raw:
    for _pair in _override_raw.split(","):
        if "=" in _pair:
            _k, _v = _pair.split("=", 1)
            TS_SYMBOL_MAP[_k.strip()] = _v.strip()

# ─────────────────────────────────────────────────────────────
# TRADIER API  (for 0DTE option gamma exposure)
# ─────────────────────────────────────────────────────────────
TRADIER_API_KEY: str  = os.getenv("TRADIER_API_KEY", "")
# Use "https://sandbox.tradier.com/v1" for paper-trading
TRADIER_BASE_URL: str = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1")

# Proxy ETF used for gamma exposure calculation per futures symbol
# (Tradier options chains cover ETFs, not CME futures directly)
OPTION_CHAIN_MAP: dict = {
    "ES": "SPY",   # S&P 500 E-mini → SPDR S&P 500 ETF
    "NQ": "QQQ",   # Nasdaq-100 E-mini → Invesco QQQ Trust
    "GC": "GLD",   # Gold futures → SPDR Gold Shares
    "CL": "USO",   # Crude Oil → United States Oil Fund
    "6E": "FXE",   # EUR/USD → Invesco CurrencyShares Euro
    "ZN": "TLT",   # 10-yr T-Note → iShares 20+ Year Treasury
    "SI": "SLV",   # Silver → iShares Silver Trust
    "HG": "CPER",  # Copper → United States Copper Index Fund
    "DX": "UUP",   # Dollar Index → Invesco DB US Dollar Index
}

# ─────────────────────────────────────────────────────────────
# OPENAI API  (for news sentiment scoring)
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ─────────────────────────────────────────────────────────────
# TELEGRAM BOT  (for alerts and /approve / /reject)
# ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str   = os.getenv("TELEGRAM_CHAT_ID", "")

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
LOG_DIR: str  = "logs"
LOG_FILE: str = "bot_signals.csv"
