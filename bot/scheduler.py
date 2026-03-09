# bot/scheduler.py
# APScheduler setup: discovers the top seasonal entry slots for each
# watched symbol, converts their labels to cron expressions in US/Eastern
# time, and registers a job for each slot.
#
# When a job fires it:
#   1. Computes the composite signal.
#   2. If composite >= MIN_COMPOSITE_SCORE, sends a Telegram alert.
#   3. Waits up to 5 minutes for /approve or /reject.
#   4. On approval, places the trade via OrderManager.
#   5. Logs every signal and decision to CSV.
#
# Independently testable:
#   python -c "
#     from bot.scheduler import build_seasonal_schedule
#     for sym, slots in build_seasonal_schedule().items():
#         print(sym, slots)
#   "

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bot.config_bot import (
    WATCHED_SYMBOLS, TOP_SLOTS_PER_SYMBOL, MIN_COMPOSITE_SCORE,
    DRY_RUN,
)
from bot.signal_engine import build_path_data, compute_composite_signal
from bot.order_manager import OrderManager
from bot import logger as bot_logger
from bot import telegram_gate

try:
    from utils.seasonal_stats import compute_edge_score
except ImportError as e:
    raise ImportError("utils/seasonal_stats.py not found on sys.path") from e

# APScheduler is lazily imported in create_scheduler() so that the rest of
# this module (label_to_cron, build_seasonal_schedule) remains testable
# without requiring the package to be installed.

logger = logging.getLogger(__name__)

# Day-of-week labels used in seasonal path labels → APScheduler cron dow
_DOW_MAP = {"Sun": "sun", "Mon": "mon", "Tue": "tue",
            "Wed": "wed", "Thu": "thu", "Fri": "fri"}


# ═══════════════════════════════════════════════════════════════
# SLOT DISCOVERY
# ═══════════════════════════════════════════════════════════════

def label_to_cron(label: str) -> Optional[dict]:
    """
    Convert a seasonal path label such as "Mon 09:30" into a dict of
    kwargs suitable for APScheduler's CronTrigger.

    Parameters
    ----------
    label : str — e.g. "Mon 09:30", "Fri 14:00"

    Returns None if the label cannot be parsed.
    """
    try:
        parts = label.strip().split()
        if len(parts) != 2:
            return None
        dow_str, time_str = parts
        dow = _DOW_MAP.get(dow_str)
        if dow is None:
            return None
        hour, minute = map(int, time_str.split(":"))
        return {"day_of_week": dow, "hour": hour, "minute": minute}
    except Exception:
        return None


def build_seasonal_schedule() -> dict:
    """
    For each symbol in WATCHED_SYMBOLS, compute the average weekly path,
    rank entry slots by edge score, and return the top N slots.

    Returns
    -------
    dict mapping symbol → list of dicts:
        {"ct": float, "label": str, "path_data": pd.DataFrame,
         "avg": float, "std": float, "win_rate": float}
    """
    schedule = {}
    for symbol in WATCHED_SYMBOLS:
        logger.info(f"[schedule] Building path data for {symbol}...")
        path_data = build_path_data(symbol)

        if path_data is None or path_data.empty:
            logger.warning(f"[schedule] No path data for {symbol} — skipping")
            continue

        edge_df = compute_edge_score(path_data)
        if edge_df.empty:
            logger.warning(f"[schedule] No qualifying edge slots for {symbol} — skipping")
            continue

        slots = []
        for ct, row in edge_df.head(TOP_SLOTS_PER_SYMBOL).iterrows():
            label = str(row.get("label", f"CT={ct:.2f}"))
            cron  = label_to_cron(label)
            if cron is None:
                logger.warning(f"[schedule] Cannot parse label '{label}' for {symbol} — skip")
                continue
            slots.append({
                "ct":        ct,
                "label":     label,
                "path_data": path_data,   # shared reference — only one copy per symbol
                "avg":       float(row.get("avg", 0.0)),
                "std":       float(row.get("std", 0.0)),
                "win_rate":  float(row.get("win_rate", 50.0)),
                "cron":      cron,
            })
            logger.info(
                f"[schedule] {symbol} slot: {label}  "
                f"win={row.get('win_rate', 0):.1f}%  "
                f"avg={row.get('avg', 0):.3f}%"
            )

        if slots:
            schedule[symbol] = slots
        else:
            logger.warning(f"[schedule] All slots for {symbol} had unparseable labels")

    return schedule


# ═══════════════════════════════════════════════════════════════
# JOB FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_signal_check(
    symbol: str,
    entry_ct: float,
    path_data: pd.DataFrame,
    order_manager: OrderManager,
) -> None:
    """
    APScheduler job: called at each scheduled entry time.

    Full pipeline:
      compute_composite_signal → Telegram alert → wait_for_approval
      → place_trade → log everything
    """
    now = datetime.now(timezone.utc)
    logger.info(f"[job] ▶ {symbol}  slot_ct={entry_ct:.2f}  time={now.isoformat()}")

    # ── Step 1: Compute composite signal ─────────────────────
    try:
        signal = compute_composite_signal(symbol, entry_ct, path_data, now)
    except Exception as e:
        msg = f"Signal computation failed for {symbol}: {e}"
        logger.error(f"[job] {msg}")
        telegram_gate.send_error_notification(f"{symbol} signal error", str(e))
        bot_logger.log_signal(
            {"symbol": symbol, "timestamp": now.isoformat()},
            decision="error", notes=msg,
        )
        return

    composite = signal["composite_score"]

    # ── Step 2: Guard — skip if below threshold ───────────────
    if composite < MIN_COMPOSITE_SCORE:
        logger.info(
            f"[job] {symbol} composite={composite:.3f} < "
            f"threshold={MIN_COMPOSITE_SCORE} — skipping"
        )
        bot_logger.log_signal(signal, decision="skipped",
                              notes=f"score {composite:.3f} below {MIN_COMPOSITE_SCORE}")
        return

    # ── Step 3: Guard — max 1 open trade ─────────────────────
    if order_manager.is_position_open():
        logger.warning(f"[job] {symbol} skipped — position already open")
        bot_logger.log_signal(signal, decision="skipped",
                              notes="position already open (MAX_OPEN_TRADES=1)")
        return

    # ── Step 4: Send Telegram alert ───────────────────────────
    logger.info(f"[job] {symbol} composite={composite:.3f} → sending alert")
    try:
        telegram_gate.send_alert(signal)
    except Exception as e:
        logger.error(f"[job] Failed to send Telegram alert: {e}")
        # Do NOT abort — log and attempt to wait anyway

    bot_logger.log_signal(signal, decision="sent")

    # ── Step 5: Wait for user approval ───────────────────────
    decision = telegram_gate.wait_for_approval()
    logger.info(f"[job] {symbol} decision={decision}")
    bot_logger.log_signal(signal, decision=decision)

    if decision not in ("approve", "dry_run"):
        logger.info(f"[job] {symbol} not approved ({decision}) — no trade placed")
        return

    # ── Step 6: Place trade ───────────────────────────────────
    std_pct = signal.get("seasonal_std", 0.20)

    try:
        trade = order_manager.place_trade(
            symbol=symbol,
            direction=signal["direction"],
            std_pct=std_pct,
        )
    except ConnectionError as e:
        # TradeStation API unreachable → abort immediately and notify
        msg = f"TradeStation API unreachable: {e}"
        logger.critical(f"[job] ABORT {symbol}: {msg}")
        telegram_gate.send_error_notification(f"{symbol} API unreachable", str(e))
        bot_logger.log_trade(
            symbol=symbol, direction=signal["direction"], quantity=0,
            entry_order_id="", stop_order_id="",
            entry_price=0, stop_price=0,
            status="aborted", notes=msg,
        )
        return
    except Exception as e:
        msg = f"place_trade error: {e}"
        logger.error(f"[job] {symbol}: {msg}")
        telegram_gate.send_error_notification(f"{symbol} order error", str(e))
        bot_logger.log_trade(
            symbol=symbol, direction=signal["direction"], quantity=0,
            entry_order_id="", stop_order_id="",
            entry_price=0, stop_price=0,
            status="failed", notes=msg,
        )
        return

    # ── Step 7: Log trade & notify ────────────────────────────
    trade_status = trade.get("status", "unknown")

    bot_logger.log_trade(
        symbol=symbol,
        direction=signal["direction"],
        quantity=trade.get("quantity", 0),
        entry_order_id=trade.get("entry_order_id", ""),
        stop_order_id=trade.get("stop_order_id", ""),
        entry_price=trade.get("entry_price", 0),
        stop_price=trade.get("stop_price", 0),
        status=trade_status,
        notes=trade.get("reason", ""),
    )

    if trade_status in ("placed", "dry_run"):
        telegram_gate.send_trade_placed_notification(trade)
    elif trade_status in ("aborted", "partial"):
        telegram_gate.send_error_notification(
            f"{symbol} trade issue",
            trade.get("reason", trade_status),
        )

    logger.info(f"[job] ◀ {symbol} done  status={trade_status}")


# ═══════════════════════════════════════════════════════════════
# SCHEDULER FACTORY
# ═══════════════════════════════════════════════════════════════

def create_scheduler(order_manager: OrderManager):
    """
    Build and start an APScheduler BackgroundScheduler with one cron job
    per top seasonal entry slot per symbol.

    All jobs fire in US/Eastern time to match the seasonal labels.
    Returns the running scheduler instance (call .shutdown() to stop).
    """
    # Lazy import so the rest of the module works without apscheduler installed
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        raise ImportError(
            "APScheduler is required to run the bot.\n"
            "Install it with: pip install apscheduler\n"
            "Or: pip install -r bot/requirements_bot.txt"
        )

    schedule_map = build_seasonal_schedule()

    if not schedule_map:
        logger.error(
            "[scheduler] No schedulable slots found. "
            "Make sure data/ cache files exist and run the dashboard once first."
        )

    scheduler = BackgroundScheduler(timezone="America/New_York")

    total_jobs = 0
    for symbol, slots in schedule_map.items():
        for slot in slots:
            cron = slot["cron"]
            job_id = f"{symbol}_{slot['label'].replace(' ', '_')}"

            scheduler.add_job(
                func=run_signal_check,
                trigger=CronTrigger(
                    day_of_week=cron["day_of_week"],
                    hour=cron["hour"],
                    minute=cron["minute"],
                    timezone="America/New_York",
                ),
                kwargs={
                    "symbol":        symbol,
                    "entry_ct":      slot["ct"],
                    "path_data":     slot["path_data"],
                    "order_manager": order_manager,
                },
                id=job_id,
                name=f"{symbol} @ {slot['label']}",
                misfire_grace_time=120,   # allow up to 2-min late fire
                coalesce=True,            # skip duplicate fires if bot lagged
                max_instances=1,          # never run the same job twice at once
            )
            logger.info(
                f"[scheduler] Scheduled {symbol} "
                f"@ {slot['label']} (cron {cron})"
            )
            total_jobs += 1

    logger.info(f"[scheduler] {total_jobs} job(s) registered")
    scheduler.start()
    return scheduler


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Discovering seasonal schedule...")
    s = build_seasonal_schedule()
    for sym, slots in s.items():
        print(f"\n{sym}:")
        for slot in slots:
            print(f"  {slot['label']:12s}  CT={slot['ct']:.2f}  "
                  f"win={slot['win_rate']:.1f}%  cron={slot['cron']}")
