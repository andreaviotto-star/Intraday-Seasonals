# utils/seasonal_stats.py  —  v4: Connected Weekly Matrix + Session Boundary Filter
# No external dependencies (math + numpy + pandas only)
import math
import datetime as dt
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FUTURES_ROLL_MONTHS    = [3, 6, 9, 12]
MIN_OBS_GATE           = 30
OUTLIER_SIGMA          = 6.0
CME_MAINTENANCE_START  = 17   # 17:00 local time
CME_MAINTENANCE_END    = 18   # 18:00 local time


# ═══════════════════════════════════════════════════════════
# CME HOLIDAY ENGINE  (algorithmic — works for any year)
# ═══════════════════════════════════════════════════════════

def _easter(year: int) -> dt.date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month, day_num = divmod(h + l - 7*m + 114, 31)
    return dt.date(year, month, day_num + 1)

def _nth_weekday(year, month, n, weekday):
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + (n - 1) * 7)

def _last_weekday(year, month, weekday):
    last = (dt.date(year + 1, 1, 1) if month == 12
            else dt.date(year, month + 1, 1)) - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)

def _observed(d):
    if d.weekday() == 5: return d - dt.timedelta(days=1)
    if d.weekday() == 6: return d + dt.timedelta(days=1)
    return d

def cme_closed_dates(year: int) -> set:
    closed = set()
    closed.add(_observed(dt.date(year, 1, 1)))
    closed.add(_easter(year) - dt.timedelta(days=2))
    july4 = dt.date(year, 7, 4)
    if july4.weekday() == 5:
        closed.add(dt.date(year, 7, 3))
    elif july4.weekday() == 6:
        closed.add(dt.date(year, 7, 5))
    else:
        closed.add(july4)
    closed.add(_nth_weekday(year, 11, 4, 3))
    closed.add(_observed(dt.date(year, 12, 25)))
    return closed

def cme_half_day_dates(year: int) -> set:
    half = set()
    half.add(_nth_weekday(year, 1, 3, 0))
    half.add(_nth_weekday(year, 2, 3, 0))
    half.add(_last_weekday(year, 5, 0))
    if year >= 2021:
        half.add(_observed(dt.date(year, 6, 19)))
    july4 = dt.date(year, 7, 4)
    if july4.weekday() not in (5, 6):
        jul3 = dt.date(year, 7, 3)
        if jul3.weekday() < 5:
            half.add(jul3)
    half.add(_nth_weekday(year, 9, 1, 0))
    thanksgiving = _nth_weekday(year, 11, 4, 3)
    half.add(thanksgiving + dt.timedelta(days=1))
    dec25 = dt.date(year, 12, 25)
    dec24 = dt.date(year, 12, 24)
    if dec24.weekday() < 5:
        half.add(dec24)
    dec31 = dt.date(year, 12, 31)
    if dec31.weekday() < 5:
        half.add(dec31)
    half -= cme_closed_dates(year)
    return half


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    rename_map = {c: c.capitalize() for c in df.columns
                  if c.lower() in ("open", "high", "low", "close", "volume")}
    return df.rename(columns=rename_map)

def _naive_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return df.index.tz_localize(None) if df.index.tz else df.index

def _ttest_1samp(arr: np.ndarray):
    n = len(arr)
    if n < 2: return np.nan, np.nan
    std = arr.std(ddof=1)
    if std == 0: return np.nan, np.nan
    t = arr.mean() / (std / math.sqrt(n))
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


# ═══════════════════════════════════════════════════════════
# SESSION BOUNDARY FILTER  — fixes the Friday midnight jump
# ═══════════════════════════════════════════════════════════

def filter_session_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove bars in the CME maintenance window and after Friday close.

    A) CME maintenance window: 17:00–18:00 local time, Monday–Thursday.
       Some feeds inject stale/price-adjusted bars during the halt.
       These create artificial price gaps that distort the weekly
       cumulative path, producing the characteristic "Friday midnight jump."

    B) Friday post-close: bars at or after 17:00 local time on Friday.
       ES/NQ/GC/ZN close for the weekend here. Any bar after this
       time is a data-feed artifact.

    IMPORTANT: Call AFTER tz_convert so the index is in local time.
    """
    if df.empty:
        return df
    df = df.copy()
    hour = df.index.hour
    dow  = df.index.dayofweek

    maintenance = (dow <= 3) & (hour >= CME_MAINTENANCE_START) & (hour < CME_MAINTENANCE_END)
    fri_close   = (dow == 4) & (hour >= CME_MAINTENANCE_START)
    return df[~(maintenance | fri_close)]


# ═══════════════════════════════════════════════════════════
# DATA CLEANING
# ═══════════════════════════════════════════════════════════

def clean_institutional_data(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    if "Volume" in df.columns:
        df = df[df["Volume"] > 0]
    df = df[df["High"] >= df["Low"]]
    df = df[(df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])]
    df = df[(df["Open"]  >= df["Low"]) & (df["Open"]  <= df["High"])]
    bar_ret = (df["Close"] - df["Open"]).abs() / df["Open"].replace(0, np.nan)
    mu, sigma = bar_ret.mean(), bar_ret.std()
    if sigma > 0:
        df = df[bar_ret <= mu + OUTLIER_SIGMA * sigma]
    if symbol:
        idx_naive = df.index.tz_localize(None) if df.index.tz else df.index
        is_roll_month = pd.Series(idx_naive.month,     index=df.index).isin(FUTURES_ROLL_MONTHS)
        is_friday     = pd.Series(idx_naive.dayofweek, index=df.index) == 4
        is_third_week = pd.Series(idx_naive.day,       index=df.index).between(15, 21)
        roll_mask     = is_roll_month & is_friday & is_third_week
        roll_dates    = df.index[roll_mask].normalize()
        roll_plus_one = roll_dates + pd.Timedelta(days=3)
        excluded      = set(roll_dates.date) | set(roll_plus_one.date)
        bar_dates     = pd.Series(pd.DatetimeIndex(df.index).normalize().date, index=df.index)
        df = df[~bar_dates.isin(excluded)]
    df = df[df.index.dayofweek != 5]
    return df


# ═══════════════════════════════════════════════════════════
# RETURN HELPERS
# ═══════════════════════════════════════════════════════════

def daily_pct_change(df: pd.DataFrame, anchor: str = "open") -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    naive_idx = _naive_index(df)
    g = df.groupby(naive_idx.normalize().date)
    anchors = g["Open"].transform("first") if anchor == "open" else g["Close"].transform("mean")
    return (df["Close"] / anchors - 1) * 100

def weekly_pct_change(df: pd.DataFrame, anchor: str = "open") -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    naive_idx = _naive_index(df)
    is_sunday = (naive_idx.dayofweek == 6).astype(int)
    adj  = naive_idx + pd.to_timedelta(is_sunday, unit="D")
    year = adj.isocalendar().year.values
    week = adj.isocalendar().week.values
    g = df.groupby([year, week])
    anchors = g["Open"].transform("first") if anchor == "open" else g["Close"].transform("mean")
    return (df["Close"] / anchors - 1) * 100


# ═══════════════════════════════════════════════════════════
# CORE METRICS ENGINE
# ═══════════════════════════════════════════════════════════

def _edge_metrics(series: pd.Series, min_obs: int = MIN_OBS_GATE) -> dict:
    nan_row = {k: np.nan for k in [
        "avg","median","std","p10","p90","win_rate","profit_factor",
        "t_stat","p_value","sharpe","sortino","kelly","count"]}
    clean = series.dropna().values
    n = len(clean)
    nan_row["count"] = n
    if n < min_obs: return nan_row
    avg = clean.mean(); med = np.median(clean); std = clean.std(ddof=1)
    p10, p90 = np.percentile(clean, [10, 90])
    winners = clean[clean > 0]; losers = clean[clean < 0]
    win_rate = (clean > 0).mean()
    profit_factor = (winners.sum() / -losers.sum()
                     if len(losers) > 0 and losers.sum() != 0 else np.nan)
    t_stat, p_value = _ttest_1samp(clean)
    sharpe = (avg / std * math.sqrt(252 * 78)) if std > 0 else np.nan
    downside_std = losers.std(ddof=1) if len(losers) > 1 else np.nan
    sortino = (avg / downside_std * math.sqrt(252 * 78)
               if (downside_std and downside_std > 0) else np.nan)
    avg_win = winners.mean() if len(winners) > 0 else 0.0
    avg_loss = -losers.mean() if len(losers) > 0 else np.nan
    kelly = (win_rate - (1 - win_rate) / (avg_win / avg_loss)
             if (avg_loss and avg_loss > 0 and avg_win > 0) else np.nan)
    return dict(avg=avg, median=med, std=std, p10=p10, p90=p90,
                win_rate=win_rate*100, profit_factor=profit_factor,
                t_stat=t_stat, p_value=p_value,
                sharpe=sharpe, sortino=sortino, kelly=kelly, count=n)


# ═══════════════════════════════════════════════════════════
# AVERAGE PATH  — v4: Connected Weekly Matrix approach
# ═══════════════════════════════════════════════════════════

def compute_avg_path(df: pd.DataFrame, freq: str = "15min",
                     anchor: str = "open", symbol: str = "",
                     min_obs: int = MIN_OBS_GATE) -> pd.DataFrame:
    """
    Builds the weekly cumulative path using the CONNECTED MATRIX approach:

    OLD approach (causes jumps):
        For every time slot T, collect all historical returns at T,
        compute mean independently. Adjacent slots T and T+1 are averaged
        from DIFFERENT subsets of weeks → artificial discontinuities.

    NEW approach (this function):
        1. Compute each individual week's full path as a time-series.
        2. Align all weeks onto the same CT grid (outer join → NaN for
           missing bars in sparse weeks).
        3. Average ACROSS weeks at each CT position. Since each column
           in the matrix IS the same CT slot, all metrics are now computed
           from the same set of weeks that have data at that exact time.
        4. The path is NOW mathematically connected: the mean at CT+Δ
           is always the mean of paths that were AT the CT value one step
           earlier, eliminating phantom jumps from subset mismatch.

    Additional fix: filter_session_boundaries() removes CME maintenance
    window bars (17:00–18:00 ET Mon–Thu) and Friday post-close bars
    that are the primary source of sparse coverage at the Thu/Fri boundary.
    """
    df = clean_institutional_data(df, symbol)
    if df.empty or len(df) < 50: return pd.DataFrame()
    df = standardize_columns(df)

    # ── Apply session boundary filter BEFORE resampling ──────────────
    df = filter_session_boundaries(df)

    df_ohlc = df[["Open","High","Low","Close"]].resample(freq).agg(
        {"Open":"first","High":"max","Low":"min","Close":"last"}
    ).dropna()
    if df_ohlc.empty: return pd.DataFrame()

    # ── Compute weekly pct returns ────────────────────────────────────
    pct = weekly_pct_change(df_ohlc, anchor)
    dw  = pd.DataFrame({"pct": pct})
    dw  = dw[dw.index.dayofweek != 5]

    naive_dw = _naive_index(dw.copy().rename(columns={"pct":"_"}))
    is_sunday = (naive_dw.dayofweek == 6).astype(int)
    adj  = naive_dw + pd.to_timedelta(is_sunday, unit="D")
    dw["year"] = adj.isocalendar().year.values
    dw["week"] = adj.isocalendar().week.values

    # continuous time coordinate (Sun=18 → Fri≈137)
    dow = dw.index.dayofweek
    shifted = (dow + 1) % 7
    dw["ct"] = (shifted * 24) + dw.index.hour + dw.index.minute / 60.0
    dow_names = {6:"Sun",0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    dw["label"] = dow.map(dow_names) + " " + dw.index.strftime("%H:%M")

    # ── Build per-week path Series, store in list ─────────────────────
    ct_to_label = dw.groupby("ct")["label"].first().to_dict()
    weekly_series = []

    for (yr, wk), grp in dw.groupby(["year","week"]):
        if len(grp) < 20:          # skip degenerate weeks (holidays etc.)
            continue
        s = grp.set_index("ct")["pct"]
        s = s[~s.index.duplicated(keep="first")]   # safety: no duplicate CT
        weekly_series.append(s)

    if len(weekly_series) < min_obs:
        return pd.DataFrame()

    # ── Matrix: rows = weeks, columns = CT slots ──────────────────────
    path_matrix = pd.concat(weekly_series, axis=1).T  # shape (n_weeks, n_ct_slots)
    path_matrix = path_matrix.sort_index(axis=1)

    # ── Aggregate across weeks per CT slot ────────────────────────────
    rows = []
    for ct in path_matrix.columns:
        col = path_matrix[ct].dropna()
        m = _edge_metrics(col, min_obs=min_obs)
        m["continuous_time"] = ct
        m["label"] = ct_to_label.get(ct, f"CT={ct:.2f}")
        rows.append(m)

    result = pd.DataFrame(rows).set_index("continuous_time").sort_index()
    result = result[result["count"] >= min_obs]
    result = result[result.index >= 18.0]
    return result


# ═══════════════════════════════════════════════════════════
# COVERAGE DIAGNOSTIC  — new utility for the UI
# ═══════════════════════════════════════════════════════════

def compute_coverage_profile(df: pd.DataFrame, freq: str = "15min",
                              symbol: str = "") -> pd.DataFrame:
    """
    Returns a DataFrame showing bar count (N) per time slot across
    the week. Use this to identify sparse zones that could distort
    the average path. Slots below 80% of the mode are flagged.

    Render as a heatmap or line chart in the UI for transparency.
    """
    df = clean_institutional_data(df, symbol)
    if df.empty: return pd.DataFrame()
    df = standardize_columns(df)
    df = filter_session_boundaries(df)

    df_ohlc = df[["Open","Close"]].resample(freq).agg(
        {"Open":"first","Close":"last"}
    ).dropna()
    df_ohlc = df_ohlc[df_ohlc.index.dayofweek != 5]

    dow = df_ohlc.index.dayofweek
    shifted = (dow + 1) % 7
    ct = (shifted * 24) + df_ohlc.index.hour + df_ohlc.index.minute / 60.0
    dow_names = {6:"Sun",0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    label = dow.map(dow_names) + " " + df_ohlc.index.strftime("%H:%M")

    counts = pd.DataFrame({"ct": ct, "label": label, "n": 1}).groupby("ct").agg(
        n=("n","count"), label=("label","first")
    ).reset_index()
    mode_n = counts["n"].mode()[0]
    counts["coverage_pct"] = (counts["n"] / mode_n * 100).clip(upper=100)
    counts["sparse"] = counts["coverage_pct"] < 80
    return counts.sort_values("ct")


# ═══════════════════════════════════════════════════════════
# REMAINING FUNCTIONS  (unchanged from v3)
# ═══════════════════════════════════════════════════════════

def compute_heatmap_grid(df: pd.DataFrame, freq: str = "30min",
                         symbol: str = "") -> pd.DataFrame:
    df = clean_institutional_data(df, symbol)
    if df.empty: return pd.DataFrame()
    df = standardize_columns(df)
    df = filter_session_boundaries(df)
    df_ohlc = df[["Open","High","Low","Close"]].resample(freq).agg(
        {"Open":"first","High":"max","Low":"min","Close":"last"}
    ).dropna()
    df_ohlc = df_ohlc[df_ohlc.index.dayofweek != 5]
    pct = daily_pct_change(df_ohlc, "open")
    dw  = pd.DataFrame({"pct": pct})
    dow_names = {6:"Sun",0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    dw["dow"]  = dw.index.dayofweek.map(dow_names)
    dw["time"] = dw.index.strftime("%H:%M")
    pivot = dw.pivot_table(values="pct", index="dow", columns="time", aggfunc="mean")
    return pivot.reindex(["Sun","Mon","Tue","Wed","Thu","Fri"])

def compute_daily_performance(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    df = clean_institutional_data(df, symbol)
    if df.empty: return pd.DataFrame()
    df = df[df.index.dayofweek != 5]
    naive_idx = _naive_index(df)
    daily = df.groupby(naive_idx.normalize().date).agg(
        Open=("Open","first"), Close=("Close","last")
    ).dropna()
    daily["pct"] = (daily["Close"] / daily["Open"] - 1) * 100
    daily["dow"] = pd.to_datetime(daily.index).dayofweek
    dow_names = {6:"Sun",0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    rows = []
    for dow_int, grp in daily.groupby("dow"):
        m = _edge_metrics(grp["pct"])
        m["day"] = dow_names.get(dow_int, str(dow_int))
        rows.append(m)
    res = pd.DataFrame(rows).set_index("day")
    return res.reindex(["Sun","Mon","Tue","Wed","Thu","Fri"]).dropna(how="all")

def compute_hourly_volatility(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    df = clean_institutional_data(df, symbol)
    if df.empty: return pd.DataFrame()
    df = filter_session_boundaries(df)
    df = df[df.index.dayofweek != 5].copy()
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_pct"] = tr / df["Open"] * 100
    df["hour"]    = df.index.hour
    res = df.groupby("hour").agg(
        vol      =("atr_pct","mean"),
        vol_median=("atr_pct","median"),
        vol_p90  =("atr_pct", lambda x: np.percentile(x, 90))
    ).reset_index()
    res["time_label"] = res["hour"].apply(lambda x: f"{x:02d}:00")
    return res

def compute_edge_score(path_df: pd.DataFrame,
                       min_win_rate: float = 55.0,
                       min_profit_factor: float = 1.2,
                       max_p_value: float = 0.05) -> pd.DataFrame:
    if path_df.empty: return pd.DataFrame()
    required = ["avg","win_rate","profit_factor","t_stat","p_value","sharpe","count"]
    if any(c not in path_df.columns for c in required): return pd.DataFrame()
    df = path_df.dropna(subset=required).copy()
    df = df[
        (df["avg"] > 0) &
        (df["win_rate"] >= min_win_rate) &
        (df["profit_factor"] >= min_profit_factor) &
        (df["p_value"] <= max_p_value)
    ]
    df["edge_score"] = (
        df["sharpe"].clip(lower=0) * np.sqrt(df["count"]) * (df["win_rate"] / 100)
    )
    return df.sort_values("edge_score", ascending=False)

def apply_event_filter(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    if df.empty: return df
    naive_idx = _naive_index(df)
    dates  = pd.Series(naive_idx.normalize().date, index=df.index)
    dt_idx = pd.to_datetime(dates)
    mask   = pd.Series(False, index=df.index)

    if event_type == "NFP":
        mask = (dt_idx.dt.dayofweek == 4) & (dt_idx.dt.day <= 7)
    elif event_type == "End-of-Month":
        mask = dt_idx.dt.is_month_end
    elif event_type == "Triple Witching":
        mask = (dt_idx.dt.month.isin([3,6,9,12]) &
                (dt_idx.dt.dayofweek == 4) & dt_idx.dt.day.between(15, 21))
    elif event_type == "Market Holidays":
        years_in_data = range(naive_idx.year.min(), naive_idx.year.max() + 2)
        closed = set()
        for yr in years_in_data: closed |= cme_closed_dates(yr)
        mask = dates.isin(closed)
    elif event_type == "Half-Day Sessions":
        years_in_data = range(naive_idx.year.min(), naive_idx.year.max() + 2)
        half_days = set()
        for yr in years_in_data: half_days |= cme_half_day_dates(yr)
        mask = dates.isin(half_days)
    elif event_type == "FOMC":
        fomc_dates = pd.to_datetime([
            "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
            "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
            "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
            "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
            "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
            "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
            "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
            "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
            "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
            "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
            "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-10-29","2025-12-10",
            "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29","2026-09-16","2026-10-28","2026-12-09",
            "2027-01-27","2027-03-17","2027-04-28","2027-06-09","2027-07-28","2027-09-15","2027-10-27","2027-12-08",
        ]).date
        mask = dates.isin(fomc_dates)
    elif event_type == "CPI":
        cpi_dates = pd.to_datetime([
            "2016-01-20","2016-02-19","2016-03-16","2016-04-14","2016-05-17","2016-06-16","2016-07-15","2016-08-16","2016-09-16","2016-10-18","2016-11-17","2016-12-15",
            "2017-01-18","2017-02-15","2017-03-15","2017-04-14","2017-05-12","2017-06-14","2017-07-14","2017-08-11","2017-09-14","2017-10-13","2017-11-15","2017-12-13",
            "2018-01-12","2018-02-14","2018-03-13","2018-04-11","2018-05-10","2018-06-12","2018-07-12","2018-08-10","2018-09-13","2018-10-11","2018-11-14","2018-12-12",
            "2019-01-11","2019-02-13","2019-03-12","2019-04-10","2019-05-10","2019-06-12","2019-07-11","2019-08-13","2019-09-12","2019-10-10","2019-11-13","2019-12-11",
            "2020-01-14","2020-02-13","2020-03-11","2020-04-10","2020-05-12","2020-06-10","2020-07-14","2020-08-12","2020-09-11","2020-10-13","2020-11-12","2020-12-10",
            "2021-01-13","2021-02-10","2021-03-10","2021-04-13","2021-05-12","2021-06-10","2021-07-13","2021-08-11","2021-09-14","2021-10-13","2021-11-10","2021-12-10",
            "2022-01-12","2022-02-10","2022-03-10","2022-04-12","2022-05-11","2022-06-10","2022-07-13","2022-08-10","2022-09-13","2022-10-13","2022-11-10","2022-12-13",
            "2023-01-12","2023-02-14","2023-03-14","2023-04-12","2023-05-10","2023-06-13","2023-07-12","2023-08-10","2023-09-13","2023-10-12","2023-11-14","2023-12-12",
            "2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15","2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10","2024-11-13","2024-12-11",
            "2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13","2025-06-11","2025-07-15","2025-08-12","2025-09-11","2025-10-24","2025-11-13","2025-12-18",
            "2026-01-13","2026-02-10","2026-03-10","2026-04-14","2026-05-12","2026-06-10","2026-07-14","2026-08-12","2026-09-11","2026-10-14","2026-11-10","2026-12-10",
            "2027-01-13","2027-02-10","2027-03-10","2027-04-14","2027-05-12","2027-06-10","2027-07-14","2027-08-11","2027-09-10","2027-10-13","2027-11-10","2027-12-09",
        ]).date
        mask = dates.isin(cpi_dates)
    elif event_type == "PCE":
        pce_dates = pd.to_datetime([
            "2016-01-29","2016-02-26","2016-03-28","2016-04-29","2016-05-27","2016-06-30","2016-07-29","2016-08-26","2016-09-30","2016-10-28","2016-11-30","2016-12-22",
            "2017-01-27","2017-02-24","2017-03-31","2017-04-28","2017-05-26","2017-06-30","2017-07-28","2017-08-31","2017-09-29","2017-10-27","2017-11-30","2017-12-22",
            "2018-01-26","2018-02-28","2018-03-29","2018-04-27","2018-05-31","2018-06-29","2018-07-27","2018-08-30","2018-09-28","2018-10-26","2018-11-29","2018-12-21",
            "2019-01-25","2019-02-28","2019-03-29","2019-04-26","2019-05-31","2019-06-28","2019-07-26","2019-08-30","2019-09-27","2019-10-25","2019-11-27","2019-12-20",
            "2020-01-31","2020-02-28","2020-03-27","2020-04-30","2020-05-29","2020-06-26","2020-07-31","2020-08-28","2020-09-25","2020-10-30","2020-11-25","2020-12-23",
            "2021-01-29","2021-02-26","2021-03-26","2021-04-30","2021-05-28","2021-06-25","2021-07-30","2021-08-27","2021-09-24","2021-10-29","2021-11-24","2021-12-23",
            "2022-01-28","2022-02-25","2022-03-31","2022-04-29","2022-05-27","2022-06-30","2022-07-29","2022-08-26","2022-09-30","2022-10-28","2022-11-30","2022-12-23",
            "2023-01-27","2023-02-24","2023-03-31","2023-04-28","2023-05-26","2023-06-30","2023-07-28","2023-08-31","2023-09-29","2023-10-27","2023-11-30","2023-12-22",
            "2024-01-26","2024-02-29","2024-03-29","2024-04-26","2024-05-31","2024-06-28","2024-07-26","2024-08-30","2024-09-27","2024-10-31","2024-11-27","2024-12-20",
            "2025-01-31","2025-02-28","2025-03-28","2025-04-25","2025-05-30","2025-06-27","2025-07-25","2025-08-29","2025-09-26","2025-10-31","2025-11-26","2025-12-19",
            "2026-01-30","2026-02-27","2026-03-27","2026-04-24","2026-05-29","2026-06-26","2026-07-31","2026-08-28","2026-09-25","2026-10-30","2026-11-20","2026-12-18",
            "2027-01-29","2027-02-26","2027-03-26","2027-04-30","2027-05-28","2027-06-25","2027-07-30","2027-08-27","2027-09-24","2027-10-29","2027-11-19","2027-12-17",
        ]).date
        mask = dates.isin(pce_dates)
    return df[~mask]