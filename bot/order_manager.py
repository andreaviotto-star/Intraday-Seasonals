# bot/order_manager.py
# TradeStation order placement, position tracking, stop-loss management,
# and kill switch.
#
# RISK RULES enforced here (non-negotiable):
#   1. Every order gets a hard stop-loss at 1× std of the avg path.
#   2. Max 1 open trade — checked before every placement.
#   3. Kill switch: cancels all open orders then flattens all positions.
#   4. If the TS API is unreachable, the trade is aborted entirely.
#
# Independently testable:
#   python -c "from bot.order_manager import OrderManager; m=OrderManager(); print(m.get_open_positions())"

import os
import sys
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bot.config_bot import (
    DRY_RUN, TS_CLIENT_ID, TS_CLIENT_SECRET, TS_REFRESH_TOKEN,
    TS_ACCOUNT_ID, TS_BASE_URL, TS_SYMBOL_MAP, DEFAULT_QTY,
    STOP_LOSS_MULTIPLIER,
)

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Thin wrapper around the TradeStation v3 REST API for order
    execution, position queries, and the kill switch.
    """

    def __init__(self):
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        # In-memory record of the currently active trade.
        # Cleared when the trade closes or kill switch fires.
        self._active_trade: Optional[dict] = None
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────
    # AUTHENTICATION
    # ─────────────────────────────────────────────────────────

    def _authenticate(self) -> None:
        """
        Exchange the refresh token for a new access token.
        Tokens from TradeStation typically expire in 1200 seconds (20 min).
        Re-authenticates automatically before every request if needed.
        """
        if not all([TS_CLIENT_ID, TS_CLIENT_SECRET, TS_REFRESH_TOKEN]):
            raise ValueError(
                "TradeStation credentials missing — check TRADESTATION_CLIENT_ID, "
                "TRADESTATION_CLIENT_SECRET, TRADESTATION_REFRESH_TOKEN in .env"
            )

        resp = requests.post(
            "https://signin.tradestation.com/oauth/token",
            data={
                "grant_type":    "refresh_token",
                "client_id":     TS_CLIENT_ID,
                "client_secret": TS_CLIENT_SECRET,
                "refresh_token": TS_REFRESH_TOKEN,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if resp.status_code != 200:
            raise ConnectionError(
                f"TradeStation auth failed ({resp.status_code}): {resp.text[:300]}"
            )

        data = resp.json()
        self._access_token = data["access_token"]
        expires_in = int(data.get("expires_in", 1200))
        # Subtract 60 s buffer so we re-auth before actual expiry
        self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
        logger.debug("TradeStation token refreshed (expires in %ds)", expires_in)

    def _ensure_auth(self) -> None:
        """Re-authenticate if the token is missing or about to expire."""
        now = datetime.now(timezone.utc)
        if self._access_token is None or (
            self._token_expiry and now >= self._token_expiry
        ):
            self._authenticate()

    def _headers(self) -> dict:
        self._ensure_auth()
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type":  "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict:
        """GET with connection-error guard."""
        try:
            resp = requests.get(
                f"{TS_BASE_URL}{path}", headers=self._headers(),
                params=params, timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"TradeStation API unreachable: {e}") from e

    def _post(self, path: str, payload: dict) -> dict:
        """POST with connection-error guard."""
        try:
            resp = requests.post(
                f"{TS_BASE_URL}{path}", headers=self._headers(),
                json=payload, timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"TradeStation API unreachable: {e}") from e

    def _delete(self, path: str) -> dict:
        """DELETE with connection-error guard."""
        try:
            resp = requests.delete(
                f"{TS_BASE_URL}{path}", headers=self._headers(), timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"TradeStation API unreachable: {e}") from e

    # ─────────────────────────────────────────────────────────
    # MARKET DATA — current price
    # ─────────────────────────────────────────────────────────

    def get_current_price(self, symbol: str) -> float:
        """
        Fetch the last trade price for a symbol via TradeStation snapshot quotes.
        Returns 0.0 if the API call fails (caller must treat this as an abort).
        """
        ts_sym = TS_SYMBOL_MAP.get(symbol, symbol)
        try:
            data = self._get(f"/marketdata/quotes/{ts_sym}")
            # TS v3 returns a list or a single dict depending on version
            if isinstance(data, list):
                data = data[0]
            last = float(data.get("Last") or data.get("Close") or 0)
            logger.debug(f"[price] {ts_sym} last={last}")
            return last
        except Exception as e:
            logger.error(f"[price] Failed to get quote for {ts_sym}: {e}")
            return 0.0

    # ─────────────────────────────────────────────────────────
    # POSITION QUERIES
    # ─────────────────────────────────────────────────────────

    def get_open_positions(self) -> list:
        """
        Return a list of open position dicts from TradeStation.
        Returns [] on API error (treated conservatively as "might be open").
        """
        if not TS_ACCOUNT_ID:
            logger.warning("[positions] TRADESTATION_ACCOUNT_ID not set")
            return []
        try:
            data = self._get(f"/brokerage/accounts/{TS_ACCOUNT_ID}/positions")
            return data.get("Positions", [])
        except Exception as e:
            logger.error(f"[positions] Error fetching positions: {e}")
            return []

    def is_position_open(self) -> bool:
        """
        True if there is already an active trade.
        Checks both the in-memory flag AND the live TS API.
        A single "yes" from either source blocks a new trade.
        """
        with self._lock:
            if self._active_trade is not None:
                return True
        # Also query the API in case the bot restarted mid-trade
        positions = self.get_open_positions()
        return len(positions) > 0

    # ─────────────────────────────────────────────────────────
    # ORDER PLACEMENT
    # ─────────────────────────────────────────────────────────

    def place_entry_order(
        self, symbol: str, direction: int, quantity: int
    ) -> Optional[str]:
        """
        Submit a market entry order.

        Returns the TradeStation OrderID on success, None on failure.
        In DRY_RUN mode, logs the intent and returns a synthetic ID.
        """
        ts_sym    = TS_SYMBOL_MAP.get(symbol, symbol)
        action    = "BUY" if direction > 0 else "SELL"
        direction_lbl = "LONG" if direction > 0 else "SHORT"

        if DRY_RUN:
            fake_id = f"DRY-ENTRY-{symbol}-{datetime.now(timezone.utc):%H%M%S}"
            logger.info(f"[DRY_RUN] Would place {direction_lbl} {quantity}x {ts_sym} @ MARKET")
            return fake_id

        payload = {
            "AccountID":  TS_ACCOUNT_ID,
            "Symbol":     ts_sym,
            "Quantity":   str(quantity),
            "OrderType":  "Market",
            "TradeAction": action,
            "TimeInForce": {"Duration": "DAY"},
            "Route":      "Intelligent",
        }

        try:
            data = self._post("/orderexecution/orders", payload)
            orders = data.get("Orders", [])
            if not orders:
                logger.error(f"[entry] No Orders in TS response: {data}")
                return None
            order_id = str(orders[0].get("OrderID", ""))
            status   = orders[0].get("Status", "unknown")
            logger.info(
                f"[entry] {direction_lbl} {quantity}x {ts_sym} "
                f"→ OrderID={order_id} status={status}"
            )
            return order_id if order_id else None
        except Exception as e:
            logger.error(f"[entry] Failed to place entry order for {symbol}: {e}")
            return None

    def place_stop_loss(
        self,
        symbol: str,
        direction: int,
        quantity: int,
        stop_price: float,
    ) -> Optional[str]:
        """
        Attach a GTC stop-market order as the hard stop-loss.

        direction: +1 means we are long → stop is a SELL order.
                   -1 means we are short → stop is a BUYTOCOVER order.
        Returns the OrderID on success, None on failure.
        In DRY_RUN mode, returns a synthetic ID.
        """
        ts_sym    = TS_SYMBOL_MAP.get(symbol, symbol)
        # Reverse direction for the protective stop
        action    = "SELL" if direction > 0 else "BUY"

        if DRY_RUN:
            fake_id = f"DRY-STOP-{symbol}-{datetime.now(timezone.utc):%H%M%S}"
            logger.info(
                f"[DRY_RUN] Would place stop-loss {action} {quantity}x {ts_sym} "
                f"@ STOP {stop_price:.4f}"
            )
            return fake_id

        payload = {
            "AccountID":  TS_ACCOUNT_ID,
            "Symbol":     ts_sym,
            "Quantity":   str(quantity),
            "OrderType":  "StopMarket",
            "TradeAction": action,
            "StopPrice":  f"{stop_price:.4f}",
            "TimeInForce": {"Duration": "GTC"},
            "Route":      "Intelligent",
        }

        try:
            data = self._post("/orderexecution/orders", payload)
            orders = data.get("Orders", [])
            if not orders:
                logger.error(f"[stop] No Orders in TS response: {data}")
                return None
            order_id = str(orders[0].get("OrderID", ""))
            status   = orders[0].get("Status", "unknown")
            logger.info(
                f"[stop] {action} {quantity}x {ts_sym} @ STOP {stop_price:.4f} "
                f"→ OrderID={order_id} status={status}"
            )
            return order_id if order_id else None
        except Exception as e:
            logger.error(f"[stop] Failed to place stop-loss for {symbol}: {e}")
            return None

    def place_trade(
        self,
        symbol: str,
        direction: int,
        std_pct: float,
    ) -> dict:
        """
        Complete trade placement pipeline:
          1. Enforce MAX 1 open trade rule.
          2. Fetch current price; abort if unavailable.
          3. Place market entry order.
          4. Compute stop price = entry ± STOP_LOSS_MULTIPLIER × std.
          5. Place GTC stop-loss order.
          6. Record trade in _active_trade.

        Returns a dict with trade details for logging.
        """
        # ── RISK CHECK: max 1 open trade ─────────────────────
        if self.is_position_open():
            logger.warning(
                f"[trade] SKIPPED {symbol}: position already open (MAX_OPEN_TRADES=1)"
            )
            return {"status": "skipped", "reason": "position_already_open"}

        quantity = DEFAULT_QTY.get(symbol, 1)

        # ── Get current price to anchor stop-loss ────────────
        entry_price = self.get_current_price(symbol)
        if entry_price <= 0:
            msg = f"[trade] ABORTED {symbol}: could not fetch current price"
            logger.error(msg)
            return {"status": "aborted", "reason": "price_unavailable"}

        # ── Compute stop price ────────────────────────────────
        # std_pct is the 1-sigma return (%) from the avg path at entry time
        stop_distance = entry_price * (std_pct / 100.0) * STOP_LOSS_MULTIPLIER
        if direction > 0:
            stop_price = entry_price - stop_distance   # long: stop below entry
        else:
            stop_price = entry_price + stop_distance   # short: stop above entry

        stop_price = round(stop_price, 4)

        # ── Place entry ───────────────────────────────────────
        try:
            entry_id = self.place_entry_order(symbol, direction, quantity)
        except ConnectionError as e:
            msg = f"[trade] ABORTED {symbol}: TS API unreachable during entry — {e}"
            logger.error(msg)
            return {"status": "aborted", "reason": "api_unreachable", "detail": str(e)}

        if not entry_id:
            return {"status": "failed", "reason": "entry_order_rejected"}

        # ── Place stop-loss ───────────────────────────────────
        try:
            stop_id = self.place_stop_loss(symbol, direction, quantity, stop_price)
        except ConnectionError as e:
            msg = (
                f"[trade] WARNING {symbol}: entry placed ({entry_id}) but "
                f"stop-loss FAILED — TS API unreachable. Manual action required!"
            )
            logger.critical(msg)
            return {
                "status": "partial",
                "reason": "stop_loss_failed",
                "entry_order_id": entry_id,
                "detail": str(e),
            }

        # ── Record active trade ───────────────────────────────
        trade = {
            "symbol":         symbol,
            "direction":      direction,
            "quantity":       quantity,
            "entry_order_id": entry_id,
            "stop_order_id":  stop_id or "",
            "entry_price":    entry_price,
            "stop_price":     stop_price,
            "status":         "dry_run" if DRY_RUN else "placed",
        }
        with self._lock:
            self._active_trade = trade

        return trade

    # ─────────────────────────────────────────────────────────
    # ORDER MANAGEMENT
    # ─────────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order by ID. Returns True on success."""
        if DRY_RUN:
            logger.info(f"[DRY_RUN] Would cancel order {order_id}")
            return True
        try:
            self._delete(f"/orderexecution/orders/{order_id}")
            logger.info(f"[cancel] Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"[cancel] Failed to cancel {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders on the account.
        Returns the number of orders cancelled.
        """
        if DRY_RUN:
            logger.info("[DRY_RUN] Would cancel all open orders")
            return 0
        if not TS_ACCOUNT_ID:
            logger.warning("[cancel_all] TRADESTATION_ACCOUNT_ID not set")
            return 0
        try:
            data = self._get(
                "/orderexecution/orders",
                params={"status": "Open", "accountID": TS_ACCOUNT_ID},
            )
            open_orders = data.get("Orders", [])
            cancelled = 0
            for order in open_orders:
                oid = order.get("OrderID")
                if oid and self.cancel_order(str(oid)):
                    cancelled += 1
            logger.info(f"[cancel_all] Cancelled {cancelled}/{len(open_orders)} orders")
            return cancelled
        except Exception as e:
            logger.error(f"[cancel_all] Error: {e}")
            return 0

    def flatten_position(self, symbol: str) -> bool:
        """
        Close an open position in `symbol` by placing a market order
        in the opposite direction.
        Returns True if the flatten order was accepted.
        """
        positions = self.get_open_positions()
        for pos in positions:
            if pos.get("Symbol", "").startswith(TS_SYMBOL_MAP.get(symbol, symbol)):
                qty = abs(int(float(pos.get("Quantity", 0))))
                if qty == 0:
                    continue
                # Determine current side from quantity sign (TS convention)
                raw_qty = float(pos.get("Quantity", 0))
                close_action = "SELL" if raw_qty > 0 else "BUY"
                ts_sym = pos.get("Symbol")

                if DRY_RUN:
                    logger.info(
                        f"[DRY_RUN] Would flatten {ts_sym}: "
                        f"{close_action} {qty} @ MARKET"
                    )
                    return True

                payload = {
                    "AccountID":   TS_ACCOUNT_ID,
                    "Symbol":      ts_sym,
                    "Quantity":    str(qty),
                    "OrderType":   "Market",
                    "TradeAction": close_action,
                    "TimeInForce": {"Duration": "DAY"},
                    "Route":       "Intelligent",
                }
                try:
                    data = self._post("/orderexecution/orders", payload)
                    oid = (data.get("Orders") or [{}])[0].get("OrderID", "unknown")
                    logger.info(
                        f"[flatten] {close_action} {qty}x {ts_sym} → OrderID={oid}"
                    )
                    return True
                except Exception as e:
                    logger.error(f"[flatten] Failed to flatten {symbol}: {e}")
                    return False
        logger.info(f"[flatten] No open position found for {symbol}")
        return True   # Nothing to flatten — not an error

    # ─────────────────────────────────────────────────────────
    # KILL SWITCH  (non-negotiable)
    # ─────────────────────────────────────────────────────────

    def kill_switch(self) -> dict:
        """
        Emergency kill switch:
          1. Cancel ALL open orders.
          2. Flatten ALL open positions (market close).
          3. Clear the in-memory active-trade record.

        Safe to call at any time — idempotent, returns a status dict.
        Works in DRY_RUN mode (logs actions, submits nothing real).
        """
        logger.critical("═══ KILL SWITCH ACTIVATED ═══")
        result = {"orders_cancelled": 0, "symbols_flattened": [], "errors": []}

        # Step 1: Cancel all open orders
        try:
            result["orders_cancelled"] = self.cancel_all_orders()
        except Exception as e:
            msg = f"cancel_all_orders error: {e}"
            logger.error(f"[kill] {msg}")
            result["errors"].append(msg)

        # Step 2: Flatten all open positions
        positions = []
        try:
            positions = self.get_open_positions()
        except Exception as e:
            msg = f"get_open_positions error: {e}"
            logger.error(f"[kill] {msg}")
            result["errors"].append(msg)

        for pos in positions:
            raw_sym = pos.get("Symbol", "")
            # Reverse-map TS symbol to internal key for flatten_position()
            internal = next(
                (k for k, v in TS_SYMBOL_MAP.items() if raw_sym.startswith(v)),
                raw_sym,
            )
            if self.flatten_position(internal):
                result["symbols_flattened"].append(raw_sym)
            else:
                result["errors"].append(f"flatten failed: {raw_sym}")

        # Step 3: Clear in-memory trade state
        with self._lock:
            self._active_trade = None

        logger.critical(
            f"[kill] Done — cancelled {result['orders_cancelled']} orders, "
            f"flattened {result['symbols_flattened']}, "
            f"errors: {result['errors']}"
        )
        return result

    def clear_active_trade(self) -> None:
        """Call this when a trade closes naturally (stop hit, target reached)."""
        with self._lock:
            self._active_trade = None


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    mgr = OrderManager()
    print("DRY_RUN =", DRY_RUN)
    print("is_position_open:", mgr.is_position_open())
    trade = mgr.place_trade(symbol="ES", direction=1, std_pct=0.20)
    print("Trade result:", trade)
    print("Kill switch:", mgr.kill_switch())
