# bot/logger.py
# Thread-safe CSV logger for all bot signals and trade records.
# Every signal computation and every order (real or DRY_RUN) is
# appended to a CSV in the logs/ directory with a UTC timestamp.
#
# Independently testable:
#   python -c "from bot.logger import log_signal; log_signal({'symbol':'ES'}, 'dry_run')"

import csv
import os
import threading
from datetime import datetime, timezone

from bot.config_bot import LOG_DIR, LOG_FILE

# Thread lock — multiple scheduler jobs may log concurrently
_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────
SIGNAL_FIELDS = [
    "timestamp_utc",
    "symbol",
    "direction",            # +1 long, -1 short
    "composite_score",
    "seasonal_score",
    "sentiment_score",
    "gamma_score",
    "volume_score",
    "seasonal_label",       # e.g. "Mon 09:30"
    "seasonal_avg",         # mean % return at that slot
    "seasonal_std",         # std of % return at that slot
    "seasonal_win_rate",
    "seasonal_profit_factor",
    "seasonal_count",       # number of historical observations
    "decision",             # sent | approved | rejected | timeout | skipped | dry_run
    "notes",
]

TRADE_FIELDS = [
    "timestamp_utc",
    "symbol",
    "direction",            # +1 long, -1 short
    "quantity",
    "entry_order_id",
    "stop_order_id",
    "entry_price",
    "stop_price",
    "stop_loss_pct",        # stop distance as % of entry
    "status",               # placed | failed | dry_run | kill_switch
    "notes",
]


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _abs_path(filename: str) -> str:
    _ensure_dir()
    return os.path.join(LOG_DIR, filename)


def _append_row(filepath: str, fields: list, row: dict) -> None:
    """Append one row to a CSV, writing the header if the file is new."""
    file_exists = os.path.exists(filepath)
    with _lock:
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def log_signal(signal: dict, decision: str, notes: str = "") -> None:
    """
    Log a composite signal computation result.

    Parameters
    ----------
    signal   : dict returned by signal_engine.compute_composite_signal()
    decision : one of "sent", "approved", "rejected", "timeout",
               "skipped", "dry_run", "error"
    notes    : optional free-text annotation
    """
    row = {
        **signal,
        "timestamp_utc": signal.get("timestamp", _utc_now()),
        "decision": decision,
        "notes": notes,
    }
    _append_row(_abs_path(LOG_FILE), SIGNAL_FIELDS, row)


def log_trade(
    symbol: str,
    direction: int,
    quantity: int,
    entry_order_id: str,
    stop_order_id: str,
    entry_price: float,
    stop_price: float,
    status: str,
    notes: str = "",
) -> None:
    """
    Log a trade placement (or DRY_RUN equivalent).

    Parameters
    ----------
    status : "placed" | "failed" | "dry_run" | "kill_switch"
    """
    stop_loss_pct = (
        abs(entry_price - stop_price) / entry_price * 100
        if entry_price and entry_price != 0
        else 0.0
    )
    row = {
        "timestamp_utc": _utc_now(),
        "symbol": symbol,
        "direction": direction,
        "quantity": quantity,
        "entry_order_id": entry_order_id,
        "stop_order_id": stop_order_id,
        "entry_price": round(entry_price, 4) if entry_price else "",
        "stop_price": round(stop_price, 4) if stop_price else "",
        "stop_loss_pct": round(stop_loss_pct, 4),
        "status": status,
        "notes": notes,
    }
    _append_row(_abs_path("bot_trades.csv"), TRADE_FIELDS, row)


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke-test — creates logs/bot_signals.csv and logs/bot_trades.csv
    dummy_signal = {
        "symbol": "ES", "direction": 1, "composite_score": 0.62,
        "seasonal_score": 0.70, "sentiment_score": 0.60,
        "gamma_score": 0.55, "volume_score": 0.50,
        "seasonal_label": "Mon 09:30", "seasonal_avg": 0.15,
        "seasonal_std": 0.20, "seasonal_win_rate": 62.0,
        "seasonal_profit_factor": 1.45, "seasonal_count": 120,
    }
    log_signal(dummy_signal, decision="dry_run", notes="smoke test")
    log_trade(
        symbol="ES", direction=1, quantity=1,
        entry_order_id="DRY-001", stop_order_id="DRY-002",
        entry_price=5100.25, stop_price=5090.25,
        status="dry_run", notes="smoke test",
    )
    print(f"Wrote test rows to {LOG_DIR}/")
