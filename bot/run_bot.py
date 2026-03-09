#!/usr/bin/env python3
# bot/run_bot.py
# Main entry point for the Intraday Seasonals trading bot.
#
# Usage:
#   python bot/run_bot.py
#
# The bot:
#   1. Verifies that required environment variables are present.
#   2. Warms up seasonal path data for all watched symbols.
#   3. Builds an APScheduler with one cron job per top entry slot.
#   4. Runs indefinitely; handles SIGINT / SIGTERM with a graceful
#      kill-switch → flatten → shutdown sequence.
#
# DRY_RUN=True (default in config_bot.py) means NO real orders are
# placed — useful for live testing with real signals.

import os
import sys
import signal
import logging
import time

# ── Ensure the project root is on sys.path ────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bot.config_bot import (
    DRY_RUN, WATCHED_SYMBOLS, LOG_DIR,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    TS_CLIENT_ID, TS_CLIENT_SECRET, TS_REFRESH_TOKEN, TS_ACCOUNT_ID,
    MIN_COMPOSITE_SCORE, APPROVAL_TIMEOUT_SEC,
)
from bot.order_manager import OrderManager
from bot.scheduler import create_scheduler
from bot import telegram_gate


# ─────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log"), encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    # Quieten noisy third-party loggers
    for noisy in ("apscheduler.scheduler", "apscheduler.executors",
                  "urllib3", "openai", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("run_bot")


# ─────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────

def _check_env() -> list:
    """
    Verify that all required environment variables are populated.
    Returns a list of missing variable names.
    """
    required = {
        "TRADESTATION_CLIENT_ID":     TS_CLIENT_ID,
        "TRADESTATION_CLIENT_SECRET": TS_CLIENT_SECRET,
        "TRADESTATION_REFRESH_TOKEN": TS_REFRESH_TOKEN,
        "TRADESTATION_ACCOUNT_ID":    TS_ACCOUNT_ID,
        "TELEGRAM_BOT_TOKEN":         TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID":           TELEGRAM_CHAT_ID,
    }
    missing = [name for name, val in required.items() if not val]
    return missing


def _check_data_cache() -> list:
    """Warn about symbols that do not yet have a local cache file."""
    cache_dir = os.path.join(_ROOT, "data")
    missing = []
    for sym in WATCHED_SYMBOLS:
        path = os.path.join(cache_dir, f"{sym}_local_cache.csv")
        if not os.path.exists(path):
            missing.append(sym)
    return missing


# ─────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────

_scheduler = None
_order_manager: OrderManager = None


def _graceful_shutdown(signum, frame) -> None:
    """
    Signal handler for SIGINT (Ctrl-C) and SIGTERM.
    Activates the kill switch before exiting so no position is left open.
    """
    logger.warning(f"Received signal {signum} — initiating graceful shutdown")

    if _order_manager is not None:
        logger.warning("Activating kill switch...")
        try:
            result = _order_manager.kill_switch()
            telegram_gate.send_kill_switch_notification(result)
        except Exception as e:
            logger.error(f"Kill switch error during shutdown: {e}")

    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass

    logger.info("Bot shut down cleanly.")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    global _scheduler, _order_manager

    _setup_logging()

    # ── Banner ────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  INTRADAY SEASONALS TRADING BOT")
    logger.info("=" * 65)
    logger.info(f"  DRY_RUN          : {DRY_RUN}")
    logger.info(f"  Watching         : {WATCHED_SYMBOLS}")
    logger.info(f"  Min composite    : {MIN_COMPOSITE_SCORE}")
    logger.info(f"  Approval timeout : {APPROVAL_TIMEOUT_SEC}s "
                f"({APPROVAL_TIMEOUT_SEC // 60} min)")
    logger.info("=" * 65)

    if DRY_RUN:
        logger.info("  ⚠  DRY_RUN=True — no real orders will be placed")
        logger.info("=" * 65)

    # ── Environment check ─────────────────────────────────────
    missing_env = _check_env()
    if missing_env:
        logger.error(
            f"Missing required environment variables: {missing_env}\n"
            "Copy .env.example to .env and fill in all values."
        )
        sys.exit(1)

    # ── Data cache check (non-fatal warning) ─────────────────
    missing_data = _check_data_cache()
    if missing_data:
        logger.warning(
            f"No local cache for: {missing_data}. "
            "Run the Streamlit dashboard once to populate data/. "
            "Those symbols will be skipped by the scheduler."
        )

    # ── Register signal handlers ──────────────────────────────
    signal.signal(signal.SIGINT,  _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    # ── Initialise order manager ──────────────────────────────
    _order_manager = OrderManager()

    # ── Build and start scheduler ─────────────────────────────
    logger.info("Building seasonal schedule (this may take a few seconds)...")
    try:
        _scheduler = create_scheduler(_order_manager)
    except Exception as e:
        logger.critical(f"Failed to create scheduler: {e}")
        telegram_gate.send_error_notification("Bot startup failed", str(e))
        sys.exit(1)

    # Notify Telegram that the bot is live
    mode_label = "DRY RUN" if DRY_RUN else "LIVE"
    telegram_gate.send_message(
        f"🤖 *Seasonals Bot Started* [{mode_label}]\n\n"
        f"Watching: `{', '.join(WATCHED_SYMBOLS)}`\n"
        f"Min score: `{MIN_COMPOSITE_SCORE}`\n\n"
        f"Reply `/kill` at any time to activate the kill switch."
    )

    logger.info("Scheduler running. Waiting for seasonal entry times...")
    logger.info("(Press Ctrl-C to stop and activate the kill switch)")

    # ── Main loop — keep the process alive and poll for /kill ─
    # The scheduler runs its jobs in background threads.
    # We stay alive here to handle Telegram /kill commands and signals.
    try:
        while True:
            time.sleep(10)
            _poll_kill_command()
    except (KeyboardInterrupt, SystemExit):
        _graceful_shutdown(signal.SIGINT, None)


def _poll_kill_command() -> None:
    """
    Check for a /kill command sent to the Telegram bot outside of a
    normal approval window. This lets you trigger the kill switch from
    Telegram at any time, not just during the 5-minute approval window.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    # We keep a module-level offset so we don't re-process old messages
    if not hasattr(_poll_kill_command, "_offset"):
        _poll_kill_command._offset = 0

    try:
        import requests
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
            params={"offset": _poll_kill_command._offset, "limit": 20, "timeout": 0},
            timeout=5,
        )
        if resp.status_code != 200:
            return
        updates = resp.json().get("result", [])
        for upd in updates:
            _poll_kill_command._offset = upd["update_id"] + 1
            msg = upd.get("message", {})
            if not msg:
                continue
            if str(msg.get("chat", {}).get("id", "")) != str(TELEGRAM_CHAT_ID):
                continue
            text = (msg.get("text") or "").strip().lower()
            if text.startswith("/kill"):
                logger.critical("[main] Received /kill command via Telegram")
                _graceful_shutdown(signal.SIGTERM, None)
    except Exception:
        pass   # never let polling crash the main loop


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
