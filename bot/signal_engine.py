# bot/signal_engine.py
# Composite signal calculator combining four independent components:
#   1. Seasonal edge    (40%) — from utils/seasonal_stats.py
#   2. News sentiment   (25%) — OpenAI chat-completion
#   3. 0DTE gamma expo  (20%) — Tradier option chain
#   4. Volume surge     (15%) — bar-range proxy on cached OHLC data
#
# Independently testable:
#   python -c "
#     from bot.signal_engine import compute_composite_signal
#     import pprint; pprint.pprint(compute_composite_signal('ES'))
#   "

import os
import sys
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import requests

# ── Add the project root to sys.path so sibling packages are importable ──────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bot.config_bot import (
    WEIGHTS, MIN_WIN_RATE, MIN_PROFIT_FACTOR, MAX_P_VALUE, MIN_OBS,
    LOOKBACK_YEARS, OPENAI_API_KEY, OPENAI_MODEL,
    TRADIER_API_KEY, TRADIER_BASE_URL, OPTION_CHAIN_MAP,
)
from utils.seasonal_stats import compute_avg_path, compute_edge_score

logger = logging.getLogger(__name__)

# Path to the existing local OHLC cache written by ts_data_fetcher / app.py
_CACHE_DIR = os.path.join(_ROOT, "data")

_ET = pytz.timezone("America/New_York")


# ═══════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════

def load_raw_data(symbol: str) -> pd.DataFrame:
    """
    Load 5-min OHLC bars from the local cache written by the dashboard.
    Returns a UTC-indexed DataFrame, or an empty DataFrame on failure.
    """
    local_file = os.path.join(_CACHE_DIR, f"{symbol}_local_cache.csv")
    if not os.path.exists(local_file):
        logger.warning(f"No cache file for {symbol}: {local_file}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(local_file, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception as e:
        logger.error(f"Failed to read cache for {symbol}: {e}")
        return pd.DataFrame()


def build_path_data(symbol: str) -> pd.DataFrame:
    """
    Load raw data, filter to the lookback window, convert to ET,
    and compute the average weekly path + edge metrics.

    This is the same pipeline as app.py's calculate_seasonals().
    Called once at bot startup; the result is cached by the scheduler.
    """
    df_raw = load_raw_data(symbol)
    if df_raw.empty:
        logger.error(f"Cannot build path data for {symbol}: no cache")
        return pd.DataFrame()

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * LOOKBACK_YEARS)
    df = df_raw[df_raw.index >= cutoff].copy()
    df.index = df.index.tz_convert("America/New_York")

    path = compute_avg_path(df, freq="15min", anchor="open",
                             symbol=symbol, min_obs=MIN_OBS)
    return path


# ═══════════════════════════════════════════════════════════════
# COMPONENT 1 — SEASONAL EDGE  (weight 0.40)
# ═══════════════════════════════════════════════════════════════

def compute_seasonal_score(
    symbol: str,
    entry_ct: float,
    path_data: pd.DataFrame,
) -> dict:
    """
    Look up the seasonal edge stats for the given continuous-time (CT)
    slot in path_data and return a normalised score in [0, 1].

    score = 0.5  → neutral (50% win rate)
    score = 1.0  → maximum edge (100% win rate, hypothetically)

    Also returns direction (+1 long / -1 short) based on avg return sign.
    """
    _empty = {
        "score": 0.5, "direction": 1, "label": "N/A",
        "avg": 0.0, "std": 0.0, "win_rate": 50.0,
        "profit_factor": float("nan"), "p_value": float("nan"), "count": 0,
    }

    if path_data is None or path_data.empty:
        return _empty

    # Find the CT slot closest to entry_ct (within ±30 min = ±0.5 CT units)
    delta = (path_data.index - entry_ct).abs()
    if delta.min() > 0.5:
        logger.warning(f"[{symbol}] No path_data slot within 30 min of CT={entry_ct:.2f}")
        return _empty

    slot = path_data.loc[delta.idxmin()]
    avg = float(slot.get("avg", 0.0))

    if pd.isna(avg):
        return _empty

    win_rate = float(slot.get("win_rate", 50.0))
    # Normalise: 50% → 0.5, 70% → 0.9, 80%+ → ~1.0
    seasonal_score = float(np.clip((win_rate / 100.0 - 0.5) * 2.0 + 0.5, 0.0, 1.0))

    return {
        "score":          seasonal_score,
        "direction":      1 if avg >= 0 else -1,
        "label":          str(slot.get("label", f"CT={entry_ct:.2f}")),
        "avg":            avg,
        "std":            float(slot.get("std", 0.0)),
        "win_rate":       win_rate,
        "profit_factor":  float(slot.get("profit_factor", float("nan"))),
        "p_value":        float(slot.get("p_value", float("nan"))),
        "count":          int(slot.get("count", 0)),
    }


# ═══════════════════════════════════════════════════════════════
# COMPONENT 2 — NEWS SENTIMENT  (weight 0.25)
# ═══════════════════════════════════════════════════════════════

def compute_sentiment_score(symbol: str, direction: int) -> float:
    """
    Ask GPT-4o-mini to assess current market sentiment for the symbol
    and score how strongly it aligns with the intended trade direction.

    Returns 0.0–1.0, where 1.0 = strongly confirms the direction.
    Returns 0.5 (neutral) on any API error or missing key.
    """
    if not OPENAI_API_KEY:
        logger.info("[sentiment] No OpenAI key — returning neutral 0.5")
        return 0.5

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[sentiment] openai package not installed — returning 0.5")
        return 0.5

    direction_str = "upward (bullish)" if direction > 0 else "downward (bearish)"
    prompt = (
        f"You are a professional futures trader. Based on the most recent publicly "
        f"available market news, macro data, and sentiment for {symbol} futures, "
        f"how strongly does the current environment support a {direction_str} move "
        f"in the next 1–4 hours?\n\n"
        f"Reply with a SINGLE decimal number only — no words, no explanation:\n"
        f"  0.0 = strongly OPPOSES the {direction_str} move\n"
        f"  0.5 = neutral or unclear\n"
        f"  1.0 = strongly CONFIRMS the {direction_str} move"
    )

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        score = float(raw)
        return float(np.clip(score, 0.0, 1.0))
    except (ValueError, AttributeError) as e:
        logger.warning(f"[sentiment] Could not parse OpenAI response '{raw}': {e}")
        return 0.5
    except Exception as e:
        logger.error(f"[sentiment] OpenAI API error: {e}")
        return 0.5


# ═══════════════════════════════════════════════════════════════
# COMPONENT 3 — 0DTE GAMMA EXPOSURE  (weight 0.20)
# ═══════════════════════════════════════════════════════════════

def compute_gamma_score(symbol: str, direction: int) -> float:
    """
    Fetch today's 0DTE option chain for the proxy ETF via Tradier,
    compute net Gamma Exposure (GEX), and return a score in [0, 1]
    that reflects whether gamma positioning supports the trade direction.

    Positive net GEX (more call gamma) → supports a long (bullish) trade.
    Negative net GEX (more put gamma)  → supports a short (bearish) trade.

    Returns 0.5 (neutral) on any API error or missing key.
    """
    if not TRADIER_API_KEY:
        logger.info("[gamma] No Tradier key — returning neutral 0.5")
        return 0.5

    proxy = OPTION_CHAIN_MAP.get(symbol, "SPY")
    today = datetime.now(_ET).strftime("%Y-%m-%d")
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    headers = {
        "Authorization": f"Bearer {TRADIER_API_KEY}",
        "Accept": "application/json",
    }
    params = {"symbol": proxy, "expiration": today, "greeks": "true"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            logger.warning(
                f"[gamma] Tradier returned {resp.status_code} for {proxy}: {resp.text[:200]}"
            )
            return 0.5

        options = (resp.json().get("options") or {}).get("option", [])
        if not options:
            logger.info(f"[gamma] No 0DTE options found for {proxy} on {today}")
            return 0.5

        call_gex = 0.0
        put_gex = 0.0
        for opt in options:
            greeks = opt.get("greeks") or {}
            gamma = float(greeks.get("gamma") or 0)
            oi    = float(opt.get("open_interest") or 0)
            gex   = gamma * oi * 100   # standard equity option multiplier
            if opt.get("option_type") == "call":
                call_gex += gex
            else:
                put_gex += gex

        total_gex = call_gex + put_gex
        if total_gex == 0:
            return 0.5

        # net_ratio: +1 = all call gamma, -1 = all put gamma
        net_ratio = (call_gex - put_gex) / total_gex

        # For a long trade: call-heavy GEX means bullish sentiment → high score
        # For a short trade: put-heavy GEX means bearish sentiment → high score
        if direction > 0:
            score = 0.5 + 0.5 * net_ratio
        else:
            score = 0.5 - 0.5 * net_ratio

        logger.debug(
            f"[gamma] {proxy} call_gex={call_gex:.0f} put_gex={put_gex:.0f} "
            f"net_ratio={net_ratio:.3f} score={score:.3f}"
        )
        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.error(f"[gamma] Tradier error for {proxy}: {e}")
        return 0.5


# ═══════════════════════════════════════════════════════════════
# COMPONENT 4 — VOLUME SURGE  (weight 0.15)
# ═══════════════════════════════════════════════════════════════

def compute_volume_score(symbol: str, window: int = 20) -> float:
    """
    Proxy for volume: compare the most recent bar's High-Low range
    to the trailing 20-bar average range.

    Note: the local OHLC cache does NOT store Volume (ts_data_fetcher.py
    drops it), so bar range is used as the best available surrogate.

    Returns 0.0–1.0, where 1.0 = strong range expansion (volume surge).
    Returns 0.5 (neutral) when data is insufficient.
    """
    df = load_raw_data(symbol)
    if df.empty or len(df) < window + 1:
        logger.info(f"[volume] Insufficient data for {symbol} — returning 0.5")
        return 0.5

    recent = df.tail(window + 1)
    ranges = recent["High"] - recent["Low"]

    current_range = float(ranges.iloc[-1])
    avg_range     = float(ranges.iloc[:-1].mean())

    if avg_range <= 0:
        return 0.5

    ratio = current_range / avg_range
    # Mapping: 0.5× → 0.2,  1.0× → 0.5,  2.0× → 0.8,  3.0×+ → ~1.0
    score = float(np.clip(0.5 + (ratio - 1.0) * 0.3, 0.0, 1.0))
    logger.debug(f"[volume] {symbol} range={current_range:.4f} avg={avg_range:.4f} "
                 f"ratio={ratio:.2f} score={score:.3f}")
    return score


# ═══════════════════════════════════════════════════════════════
# COMPOSITE SIGNAL
# ═══════════════════════════════════════════════════════════════

def compute_composite_signal(
    symbol: str,
    entry_ct: float,
    path_data: pd.DataFrame,
    now: datetime = None,
) -> dict:
    """
    Compute the full 4-component composite signal for a seasonal entry slot.

    Parameters
    ----------
    symbol    : futures symbol key (e.g. "ES")
    entry_ct  : continuous-time coordinate of the scheduled entry slot
    path_data : pre-computed DataFrame from build_path_data(symbol)
    now       : override current time (defaults to utcnow)

    Returns a flat dict with all scores and metadata — suitable for
    direct use in telegram_gate.send_alert() and logger.log_signal().
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # ── Component 1: Seasonal ────────────────────────────────
    seasonal = compute_seasonal_score(symbol, entry_ct, path_data)
    direction = seasonal["direction"]

    # ── Component 2: Sentiment ───────────────────────────────
    sentiment_score = compute_sentiment_score(symbol, direction)

    # ── Component 3: Gamma ───────────────────────────────────
    gamma_score = compute_gamma_score(symbol, direction)

    # ── Component 4: Volume ──────────────────────────────────
    volume_score = compute_volume_score(symbol)

    # ── Weighted composite ───────────────────────────────────
    composite = (
        WEIGHTS["seasonal"]  * seasonal["score"] +
        WEIGHTS["sentiment"] * sentiment_score   +
        WEIGHTS["gamma"]     * gamma_score       +
        WEIGHTS["volume"]    * volume_score
    )

    result = {
        "symbol":                  symbol,
        "timestamp":               now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "direction":               direction,
        "composite_score":         round(composite, 4),
        "seasonal_score":          round(seasonal["score"], 4),
        "sentiment_score":         round(sentiment_score, 4),
        "gamma_score":             round(gamma_score, 4),
        "volume_score":            round(volume_score, 4),
        "seasonal_label":          seasonal["label"],
        "seasonal_avg":            round(seasonal["avg"], 4),
        "seasonal_std":            round(seasonal["std"], 4),
        "seasonal_win_rate":       round(seasonal["win_rate"], 2),
        "seasonal_profit_factor":  seasonal["profit_factor"],
        "seasonal_count":          seasonal["count"],
    }

    logger.info(
        f"[signal] {symbol} | slot={seasonal['label']} | "
        f"composite={composite:.3f} | "
        f"S={seasonal['score']:.2f} N={sentiment_score:.2f} "
        f"G={gamma_score:.2f} V={volume_score:.2f}"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.DEBUG)

    print("Building path data for ES...")
    pd_es = build_path_data("ES")
    if pd_es.empty:
        print("No path data — make sure data/ES_local_cache.csv exists.")
    else:
        # Use the top edge slot for the test
        from utils.seasonal_stats import compute_edge_score
        edge = compute_edge_score(pd_es)
        if not edge.empty:
            top_ct    = edge.index[0]
            top_label = edge.iloc[0]["label"]
            print(f"Top slot: {top_label}  CT={top_ct:.2f}")
            sig = compute_composite_signal("ES", top_ct, pd_es)
            pprint.pprint(sig)
        else:
            print("No edge slots found.")
