# bot/telegram_gate.py
# Telegram integration: send formatted trade alerts and poll for
# /approve or /reject replies from the configured chat.
#
# Protocol:
#   1. send_alert(signal, trade_details)  — sends a rich message
#   2. wait_for_approval(timeout=300)     — polls getUpdates every 5 s
#      Returns "approve" | "reject" | "timeout"
#
# Uses plain requests (no python-telegram-bot dependency needed).
#
# Independently testable:
#   python -c "from bot.telegram_gate import send_message; send_message('ping')"

import os
import sys
import time
import logging
from datetime import datetime, timezone

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bot.config_bot import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    APPROVAL_TIMEOUT_SEC, DRY_RUN,
)

logger = logging.getLogger(__name__)

_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
_POLL_INTERVAL_SEC = 5       # how often to check for replies


# ─────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────

def _post(method: str, payload: dict) -> dict:
    """POST to a Telegram Bot API method. Returns the JSON response."""
    try:
        resp = requests.post(f"{_BASE}/{method}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[telegram] {method} failed: {e}")
        return {}


def _get(method: str, params: dict = None) -> dict:
    """GET from a Telegram Bot API method."""
    try:
        resp = requests.get(f"{_BASE}/{method}", params=params, timeout=12)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[telegram] {method} failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# PUBLIC SEND FUNCTIONS
# ─────────────────────────────────────────────────────────────

def send_message(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send a plain text message to TELEGRAM_CHAT_ID.
    Returns True if the API accepted it, False otherwise.
    In DRY_RUN mode, logs the message instead of sending it.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[telegram] Bot token or chat ID not configured — cannot send")
        return False

    if DRY_RUN:
        logger.info(f"[DRY_RUN][telegram] Would send:\n{text}")
        return True

    result = _post("sendMessage", {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
    })
    ok = result.get("ok", False)
    if not ok:
        logger.error(f"[telegram] sendMessage failed: {result}")
    return ok


def send_alert(signal: dict, trade_details: dict = None) -> bool:
    """
    Send a formatted trade-alert message with full signal breakdown.

    Parameters
    ----------
    signal        : dict from signal_engine.compute_composite_signal()
    trade_details : optional dict with stop_price, quantity, etc.
                    (added after place_trade() computes the stop level)
    """
    direction_emoji = "📈 LONG" if signal.get("direction", 1) > 0 else "📉 SHORT"
    score = signal.get("composite_score", 0.0)
    score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))

    lines = [
        f"🤖 *SEASONAL BOT — TRADE ALERT*",
        f"",
        f"*Symbol:*    `{signal.get('symbol', '?')}`",
        f"*Direction:* {direction_emoji}",
        f"*Slot:*      `{signal.get('seasonal_label', '?')}`",
        f"",
        f"*Composite Score:* `{score:.3f}` [{score_bar}]",
        f"",
        f"Signal breakdown:",
        f"  Seasonal  (40%): `{signal.get('seasonal_score', 0):.3f}`  "
        f"win={signal.get('seasonal_win_rate', 0):.1f}%  "
        f"avg={signal.get('seasonal_avg', 0):.3f}%  N={signal.get('seasonal_count', 0)}",
        f"  Sentiment (25%): `{signal.get('sentiment_score', 0):.3f}`",
        f"  Gamma     (20%): `{signal.get('gamma_score', 0):.3f}`",
        f"  Volume    (15%): `{signal.get('volume_score', 0):.3f}`",
    ]

    if trade_details:
        ep = trade_details.get("entry_price", 0)
        sp = trade_details.get("stop_price", 0)
        qty = trade_details.get("quantity", 1)
        stop_dist = abs(ep - sp) if ep and sp else 0
        stop_pct  = (stop_dist / ep * 100) if ep else 0
        lines += [
            f"",
            f"*Trade details:*",
            f"  Quantity:    `{qty}` contract(s)",
            f"  Entry:       `{ep:.2f}` (market)",
            f"  Stop-loss:   `{sp:.2f}` ({stop_pct:.2f}% from entry)",
        ]

    if DRY_RUN:
        lines.insert(1, "⚠️ *DRY RUN — no real order will be placed*")

    lines += [
        f"",
        f"Reply within `{APPROVAL_TIMEOUT_SEC // 60}` min:",
        f"  /approve — place the trade",
        f"  /reject  — skip this signal",
        f"",
        f"_{signal.get('timestamp', datetime.now(timezone.utc).isoformat())}_",
    ]

    return send_message("\n".join(lines))


def send_kill_switch_notification(result: dict) -> None:
    """Notify on Telegram that the kill switch has been activated."""
    msg = (
        f"🚨 *KILL SWITCH ACTIVATED*\n\n"
        f"Orders cancelled: `{result.get('orders_cancelled', 0)}`\n"
        f"Positions flattened: `{result.get('symbols_flattened', [])}`\n"
        f"Errors: `{result.get('errors', [])}`\n\n"
        f"_All open orders and positions have been closed._"
    )
    send_message(msg)


def send_trade_placed_notification(trade: dict) -> None:
    """Confirm to the user that an order was accepted by TradeStation."""
    mode = "⚠️ DRY RUN" if DRY_RUN else "✅ LIVE"
    direction_lbl = "LONG" if trade.get("direction", 1) > 0 else "SHORT"
    msg = (
        f"{mode} — *Trade Placed*\n\n"
        f"Symbol:      `{trade.get('symbol')}`\n"
        f"Direction:   `{direction_lbl}`\n"
        f"Quantity:    `{trade.get('quantity')}` contract(s)\n"
        f"Entry ID:    `{trade.get('entry_order_id')}`\n"
        f"Stop ID:     `{trade.get('stop_order_id')}`\n"
        f"Stop price:  `{trade.get('stop_price', 0):.4f}`\n"
    )
    send_message(msg)


def send_error_notification(context: str, error: str) -> None:
    """Send an error alert so you are always aware of bot failures."""
    msg = (
        f"❌ *Bot Error*\n\n"
        f"Context: `{context}`\n"
        f"Error:   `{error[:400]}`"
    )
    send_message(msg)


# ─────────────────────────────────────────────────────────────
# APPROVAL POLLER
# ─────────────────────────────────────────────────────────────

def _get_latest_update_id() -> int:
    """
    Drain current pending updates and return the next offset to use,
    so we only react to messages sent AFTER this call.
    """
    data = _get("getUpdates", params={"limit": 100, "timeout": 0})
    updates = data.get("result", [])
    if updates:
        return updates[-1]["update_id"] + 1
    return 0


def wait_for_approval(timeout: int = APPROVAL_TIMEOUT_SEC) -> str:
    """
    Poll Telegram getUpdates for up to `timeout` seconds, looking for
    /approve or /reject from TELEGRAM_CHAT_ID.

    Returns
    -------
    "approve"  — user replied /approve
    "reject"   — user replied /reject
    "timeout"  — no reply within the timeout window
    "dry_run"  — DRY_RUN mode: immediately returns "approve" to allow
                 the full pipeline to run without requiring user input
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[approval] Telegram not configured — auto-rejecting")
        return "reject"

    # In DRY_RUN, auto-approve so the full pipeline can be exercised
    if DRY_RUN:
        logger.info("[DRY_RUN] Auto-approving trade (DRY_RUN=True)")
        return "dry_run"

    # Advance the offset past all existing messages so we only see
    # new replies that arrive AFTER the alert was sent
    offset = _get_latest_update_id()
    deadline = time.monotonic() + timeout
    chat_id_str = str(TELEGRAM_CHAT_ID)

    logger.info(
        f"[approval] Waiting up to {timeout}s for /approve or /reject "
        f"in chat {TELEGRAM_CHAT_ID} ..."
    )

    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SEC)

        data = _get("getUpdates", params={"offset": offset, "limit": 50, "timeout": 0})
        updates = data.get("result", [])

        for upd in updates:
            offset = upd["update_id"] + 1   # always advance offset
            msg = upd.get("message", {})
            if not msg:
                continue

            # Only accept commands from the configured chat
            sender_chat = str(msg.get("chat", {}).get("id", ""))
            if sender_chat != chat_id_str:
                logger.debug(f"[approval] Ignoring message from unknown chat {sender_chat}")
                continue

            text = (msg.get("text") or "").strip().lower()
            if text.startswith("/approve"):
                logger.info("[approval] Received /approve")
                return "approve"
            if text.startswith("/reject"):
                logger.info("[approval] Received /reject")
                return "reject"

    logger.info(f"[approval] Timeout after {timeout}s — skipping trade")
    return "timeout"


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("Sending test ping to Telegram...")
    ok = send_message("🤖 *Bot test ping* — if you see this, Telegram is configured correctly.")
    print("Sent:", ok)
    if ok and not DRY_RUN:
        print("Waiting 30 s for /approve or /reject...")
        decision = wait_for_approval(timeout=30)
        print("Decision:", decision)
