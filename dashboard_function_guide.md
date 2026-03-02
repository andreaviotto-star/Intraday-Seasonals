# Intraday Seasonals Dashboard — Complete Function Guide
# ============================================================
# A plain-English reference for every function in the codebase.
# Read top-to-bottom for a full understanding of the data pipeline.


## ═══════════════════════════════════════════════════════════
## PART 1 — HOW THE DATA FLOWS (Big Picture)
## ═══════════════════════════════════════════════════════════

The dashboard follows a strict one-way pipeline:

  [Raw OHLCV bars]
       │
       ▼
  get_raw_data()          ← fetch from TradeStation or yfinance, cache to CSV
       │
       ▼
  calculate_seasonals()   ← slice by lookback years, apply timezone, run filters
       │
       ├── clean_institutional_data()   ← 6-layer sanity filter
       │
       ├── compute_avg_path()           ← weekly cumulative path + all edge metrics
       ├── compute_heatmap_grid()       ← day-of-week × time-of-day mean return
       ├── compute_daily_performance()  ← per-weekday edge metrics
       └── compute_hourly_volatility()  ← ATR profile by hour
              │
              ▼
       compute_edge_score()             ← ranked table of statistically valid slots
              │
              ▼
       Streamlit UI charts + tables


## ═══════════════════════════════════════════════════════════
## PART 2 — utils/seasonal_stats.py
## ═══════════════════════════════════════════════════════════

─────────────────────────────────────────────────────────────
FUNCTION: standardize_columns(df)
─────────────────────────────────────────────────────────────
PURPOSE:
  Makes sure the DataFrame always has columns named exactly
  "Open", "High", "Low", "Close", "Volume" regardless of
  how the data source delivered them.

WHY IT EXISTS:
  TradeStation may return "open" (lowercase).
  yfinance may return a MultiIndex like ("Close", "ES=F").
  If columns are inconsistent, every downstream function breaks.

WHAT IT DOES:
  1. If the columns are a MultiIndex (yfinance style), it strips
     the second level so you only keep the first.
  2. Renames any lowercase/uppercase variant to Title Case.

INPUT:  Any raw DataFrame from a data fetcher.
OUTPUT: Same DataFrame with standardised column names.

EXAMPLE:
  Before: columns = ["open", "HIGH", "low", "close"]
  After:  columns = ["Open", "High", "Low", "Close"]


─────────────────────────────────────────────────────────────
FUNCTION: clean_institutional_data(df, symbol="")
─────────────────────────────────────────────────────────────
PURPOSE:
  The data quality firewall. Removes six categories of bad bars
  before any statistics are computed. Garbage in = garbage edge.

LAYER A — Duplicate timestamps
  Problem:  Some feeds re-send bars on reconnect. A bar at
            09:30 appearing twice doubles its weight in the mean.
  Fix:      Keep only the first occurrence of each timestamp.

LAYER B — Zero-volume bars
  Problem:  During pre-market, some feeds inject placeholder bars
            with no actual trades (Volume = 0). These have random
            or stale OHLC values that corrupt your averages.
  Fix:      Drop all bars where Volume == 0.
            (Only applied if a Volume column exists.)

LAYER C — Invalid OHLC geometry
  Problem:  Bad ticks or feed errors can produce bars where
            Close > High, or Low > High — physically impossible.
  Fix:      Keep only bars satisfying:
            Low <= Open <= High  AND  Low <= Close <= High

LAYER D — 6-sigma outlier gate
  Problem:  On roll dates, data-vendor errors, or fat-finger prints,
            a single bar can show a 2% move when the average is 0.02%.
            This one bar can shift the mean for an entire time slot
            by more than the edge you are trying to measure.
  Fix:      Compute the mean (μ) and standard deviation (σ) of all
            bar returns (|Close - Open| / Open). Drop any bar where
            the return exceeds μ + 6σ.
            Six sigma is chosen because legitimate large moves
            (e.g., CPI day) rarely exceed 5σ on a 5-min bar.

LAYER E — Quarterly roll date exclusion
  Problem:  ES, NQ, GC, ZN all roll contracts on the 3rd Friday
            of March, June, September, December. On these days:
            - The front-month contract loses liquidity rapidly
            - The data feed may splice front and back months
              creating an artificial price gap
            - Volume migrates to the next contract mid-session
            These effects inflate volatility and distort seasonal
            averages near the roll, creating a false "edge".
  Fix:      Identify the 3rd Friday of each roll month
            (day is between 15 and 21, dayofweek == 4).
            Exclude that day AND the following Monday
            (the first day trading fully on the new contract).

LAYER F — Saturday drift
  Problem:  Timezone conversion bugs (especially UTC → CET in
            winter/summer time transitions) can shift Sunday
            overnight bars into Saturday timestamps.
  Fix:      Drop all bars where dayofweek == 5 (Saturday).

INPUT:  Raw OHLCV DataFrame (tz-aware index).
OUTPUT: Cleaned DataFrame with bad bars removed.


─────────────────────────────────────────────────────────────
FUNCTION: _naive_index(df)
─────────────────────────────────────────────────────────────
PURPOSE:
  Internal utility. Returns the index stripped of timezone info.

WHY IT EXISTS:
  Pandas groupby operations on date do not work cleanly with
  tz-aware DatetimeIndex. This helper strips the timezone so
  we can group by calendar date without errors.

INPUT:  DataFrame with either tz-aware or tz-naive index.
OUTPUT: DatetimeIndex without timezone.


─────────────────────────────────────────────────────────────
FUNCTION: _ttest_1samp(series)
─────────────────────────────────────────────────────────────
PURPOSE:
  Answers the question: "Is this average return actually
  different from zero, or could it be random noise?"

THE MATHS:
  1. Compute t = mean / (std / sqrt(n))
     This is how many standard errors the mean is away from 0.
     A t of 2.0 means the mean is 2 standard errors above zero.

  2. Compute p = erfc(|t| / sqrt(2))
     erfc is the complementary error function from Python's
     built-in math module. It gives the two-tailed probability
     of seeing a t-value this extreme if the true mean were 0.

HOW TO READ THE OUTPUT:
  p = 0.04 → only 4% chance this pattern is random noise → KEEP
  p = 0.30 → 30% chance it's random → DISCARD
  Standard threshold: p <= 0.05

WHY NO SCIPY:
  For n >= 30 (guaranteed by MIN_OBS_GATE), the t-distribution
  converges to the standard normal. The erfc approximation has
  an error of < 0.001 vs scipy — irrelevant at p-value thresholds
  of 0.01 or 0.05.

INPUT:  numpy array of returns.
OUTPUT: (t_statistic, p_value) tuple.


─────────────────────────────────────────────────────────────
FUNCTION: _edge_metrics(series, min_obs=30)
─────────────────────────────────────────────────────────────
PURPOSE:
  The statistical engine of the entire dashboard. Given a list
  of historical returns for a specific time slot, computes every
  institutional metric in one pass.

METRICS EXPLAINED:

  avg (Mean Return %)
    The simple average of all returns for this time slot.
    Example: avg = +0.08% means on average this slot gains 0.08%
    WARNING: Mean alone is not an edge. You need all metrics below.

  median (Median Return %)
    The middle value when all returns are sorted. More robust
    than the mean because it ignores extreme outliers.
    If median >> mean, the distribution has negative skew (big losses).
    If median << mean, it has positive skew (occasional big wins).

  std (Standard Deviation %)
    How much individual returns vary around the mean.
    High std = wide range of outcomes = risky.
    Used in Sharpe calculation.

  p10, p90 (10th and 90th Percentile)
    The realistic range of outcomes. On 80% of historical occurrences,
    the return fell between p10 and p90.
    Much more honest than ±1σ bands which assume a bell curve.
    Futures returns are NOT bell-shaped (fat tails), so use these.

  win_rate (%)
    Percentage of occurrences where the return was positive.
    50% = coin flip. You need >= 55% consistently to have an edge
    after commissions. By itself, win rate says nothing about
    the size of wins vs losses (see profit_factor).

  profit_factor
    = Sum of all winning returns / Sum of all losing returns (absolute)
    Example: profit_factor = 1.5 means for every $1 lost, $1.50 won.
    Below 1.0 = strategy loses money even with >50% win rate.
    Institutional minimum: >= 1.2

  t_stat, p_value
    See _ttest_1samp above. Applied here to the full return series
    for this time slot. If p_value > 0.05, ignore the pattern.

  sharpe (Annualised Sharpe Ratio)
    = (avg / std) × sqrt(252 × 78)
    252 = trading days/year, 78 = five-min bars per trading day.
    Measures return per unit of total risk.
    > 0.5 = acceptable, > 1.0 = good, > 2.0 = excellent.
    This is annualised so it accounts for the compounding of
    many 5-min bars into a yearly return.

  sortino (Annualised Sortino Ratio)
    Like Sharpe but divides by downside deviation only (std of
    losing returns). For a seasonal long edge, you want to be
    punished only for losses, not for large upside moves.
    Sortino > Sharpe is normal and healthy for long strategies.

  kelly (Kelly Criterion fraction)
    = Win_Rate - (1 - Win_Rate) / (avg_win / avg_loss)
    The mathematically optimal fraction of capital to risk.
    Example: kelly = 0.12 → risk 12% of account on this trade.
    Negative kelly = do not trade this slot under any conditions.
    In practice, use half-Kelly (kelly / 2) to account for
    estimation error.

  count (N — number of observations)
    How many historical occurrences back this statistic.
    A win_rate of 70% over N=15 is worthless.
    A win_rate of 58% over N=520 is highly significant.
    Always check N before trusting any metric.

INPUT:  pd.Series of returns (%), min_obs threshold.
OUTPUT: dict with all 13 metrics (NaN if n < min_obs).


─────────────────────────────────────────────────────────────
FUNCTION: daily_pct_change(df, anchor="open")
─────────────────────────────────────────────────────────────
PURPOSE:
  For each bar, computes how far price has moved since the
  opening of THAT SAME CALENDAR DAY.

HOW IT WORKS:
  1. Group all bars by their calendar date.
  2. For each group, find the "anchor":
     - anchor="open"  → first Open price of the day
     - anchor="close" → mean Close price of the day
  3. For every bar: return = (Close / anchor - 1) × 100

USED BY: compute_heatmap_grid (intraday moves vs day open)

INPUT:  Cleaned OHLCV DataFrame.
OUTPUT: pd.Series of % returns, same index as input.


─────────────────────────────────────────────────────────────
FUNCTION: weekly_pct_change(df, anchor="open")
─────────────────────────────────────────────────────────────
PURPOSE:
  For each bar, computes how far price has moved since the
  opening of THAT SAME CALENDAR WEEK.
  This is what drives the main avg path chart.

THE SUNDAY PROBLEM:
  ES/NQ open Sunday at 18:00 ET. ISO week numbering puts Sunday
  in the PREVIOUS week. Without correction, Sunday evening bars
  would anchor to Friday's open — completely wrong.
  Fix: Add 1 day to any Sunday bar's date before computing
  the ISO week. This moves Sunday into Monday's week.

HOW IT WORKS:
  1. Adjust Sunday timestamps forward by 1 day.
  2. Group all bars by (ISO year, ISO week).
  3. Anchor = first Open of the week (Sunday 18:00 ET open).
  4. For every bar: return = (Close / anchor - 1) × 100

USED BY: compute_avg_path

INPUT:  Cleaned OHLCV DataFrame.
OUTPUT: pd.Series of % returns from week open.


─────────────────────────────────────────────────────────────
FUNCTION: compute_avg_path(df, freq, anchor, symbol, min_obs)
─────────────────────────────────────────────────────────────
PURPOSE:
  Builds the main weekly cumulative path chart.
  Answers: "On average, where does price go during the week,
  and at what time slots does a real edge exist?"

PARAMETERS:
  freq     = Resampling frequency. "15min" = aggregate raw 5-min
             bars into 15-min bars before computing statistics.
             Reduces noise and speeds up computation.
  anchor   = "open" to measure vs weekly open price.
  symbol   = Passed to clean_institutional_data for roll exclusion.
  min_obs  = Drop any time slot with fewer than this many weeks
             of history. Default 30. With 10 years = ~520 weeks,
             this rarely triggers except at unusual session times.

CONTINUOUS TIME INDEX:
  To plot the week as a single continuous line (Sun→Fri), each
  bar is assigned a "continuous_time" value:
    continuous_time = (shifted_dayofweek × 24) + hour + (minute/60)
  where shifted_dayofweek maps Sun=1, Mon=2, ..., Fri=6.
  This gives a number from 18 (Sun 18:00) to 144 (Fri 23:55).
  The x-axis of the chart IS this number.

OUTPUT COLUMNS:
  avg, median, std, p10, p90, win_rate, profit_factor,
  t_stat, p_value, sharpe, sortino, kelly, count, label
  One row per unique time-of-week slot.


─────────────────────────────────────────────────────────────
FUNCTION: compute_heatmap_grid(df, freq, symbol)
─────────────────────────────────────────────────────────────
PURPOSE:
  Builds the day-of-week × time-of-day heatmap.
  Answers: "Which specific hour on which specific day
  tends to be strongest or weakest?"

HOW IT WORKS:
  1. Resample to freq (default 30min for readability).
  2. Compute daily_pct_change → each bar shows % vs day open.
  3. Pivot: rows = days (Sun–Fri), columns = times (00:00–23:30).
  4. Each cell = average return across all historical occurrences
     of that day+time combination.

IMPORTANT — NaN vs Zero:
  Missing data (holiday, early close, no trading) is left as NaN.
  Plotly renders NaN as grey. The old code used fillna(0) which
  injected a fake 0% return, pulling cells toward neutral and
  hiding real patterns.

OUTPUT: DataFrame (6 rows × 48 columns), NaN for missing cells.


─────────────────────────────────────────────────────────────
FUNCTION: compute_daily_performance(df, symbol)
─────────────────────────────────────────────────────────────
PURPOSE:
  Answers: "Which day of the week has the best statistical edge
  when measured from day open to day close?"

HOW IT WORKS:
  1. Collapse all 5-min bars into one row per calendar day:
     Open = first bar's Open, Close = last bar's Close.
  2. Compute daily return = (Close / Open - 1) × 100.
  3. Group by day-of-week (Sun, Mon, ..., Fri).
  4. Run _edge_metrics on each group.

OUTPUT: DataFrame with one row per day, all 13 edge metrics.


─────────────────────────────────────────────────────────────
FUNCTION: compute_hourly_volatility(df, symbol)
─────────────────────────────────────────────────────────────
PURPOSE:
  Answers: "When during the 24-hour session is the market
  most volatile? When is it safest to trade?"

WHY TRUE RANGE (not just High-Low):
  High-Low only captures the bar's internal move. It misses
  overnight gaps. If ES closes at 4500 and opens next bar at
  4480, the gap is 20 points of risk that High-Low ignores.
  True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
  This captures gap risk, which is the dominant volatility
  source for 24/7 futures.

METRICS:
  vol        = mean ATR % per hour (mean True Range / Open × 100)
  vol_median = median ATR % (robust to outlier sessions)
  vol_p90    = 90th percentile ATR % (worst-case planning)

USE CASE:
  High vol_p90 at 14:30 ET = CPI/FOMC hour. Your stop needs to
  be wider here, or you skip trading this hour entirely.

OUTPUT: DataFrame with one row per hour (0–23), 4 columns.


─────────────────────────────────────────────────────────────
FUNCTION: compute_edge_score(path_df, min_win_rate,
                              min_profit_factor, max_p_value)
─────────────────────────────────────────────────────────────
PURPOSE:
  The final filter. Of the ~500 time slots in a week, only a
  handful will pass all three institutional gates simultaneously.
  This function surfaces only those slots and ranks them.

THE THREE GATES:
  1. win_rate >= min_win_rate (default 55%)
     The slot must win more than half the time by a margin.

  2. profit_factor >= min_profit_factor (default 1.2)
     Gross wins must outweigh gross losses by at least 20%.
     A 60% win rate with tiny wins and massive losses = losing.

  3. p_value <= max_p_value (default 0.05)
     The t-test must confirm the mean return is statistically
     distinguishable from zero. Without this gate, curve-fitting
     on historical data will always produce a "best" slot even
     from random noise.

THE EDGE SCORE FORMULA:
  edge_score = Sharpe × sqrt(N) × (win_rate / 100)

  - Sharpe:         rewards high return-per-unit-of-risk
  - sqrt(N):        rewards statistical confidence (more history)
  - win_rate/100:   rewards consistency (not just one big win)

  This is a composite ranking score, not a dollar amount.
  Use it to prioritise which slots deserve further research.

OUTPUT: Filtered, ranked DataFrame — highest edge_score first.


─────────────────────────────────────────────────────────────
FUNCTION: apply_event_filter(df, event_type)
─────────────────────────────────────────────────────────────
PURPOSE:
  Removes all bars from known high-impact macro event days,
  so you can study the "clean" seasonal without event noise.

EVENTS AND DETECTION METHOD:

  NFP (Non-Farm Payrolls)
    Algorithmic: first Friday of every month (day <= 7 AND Friday).
    No hardcoded dates needed — works for all years automatically.

  End-of-Month
    Uses pandas dt.is_month_end. Captures pension rebalancing,
    window dressing, and systematic month-end flows.

  Triple Witching
    March/June/September/December, third Friday (day 15–21).
    Simultaneous expiry of index futures, index options, and
    single-stock options creates massive artificial volume.

  FOMC / CPI / PCE
    Hardcoded release dates from 2015 to 2026.
    These are extended back to 2015 (vs 2024 in the original code)
    to cover the full 10-year lookback window.
    Without this fix, 8 years of event days passed through
    the filter silently.

  WHY EXCLUDE THESE:
    Your seasonal model assumes price movement is driven by
    time-of-day patterns. On event days, movement is driven
    by the event itself. Mixing them in pollutes your averages
    and makes the model think 08:30 ET is structurally bullish
    when it's really just "CPI day the Fed cut rates".

INPUT:  Cleaned DataFrame + event type string.
OUTPUT: Same DataFrame with all bars from event days removed.


## ═══════════════════════════════════════════════════════════
## PART 3 — app.py
## ═══════════════════════════════════════════════════════════

─────────────────────────────────────────────────────────────
FUNCTION: start_background_downloader()
─────────────────────────────────────────────────────────────
PURPOSE:
  Silently pre-fetches data for all symbols in the background
  so they are ready before the user selects them.

RACE CONDITION FIX:
  The original code wrote CSVs directly: df.to_csv(file_path)
  If the main thread reads the file while the background thread
  is still writing, you get a partial read (truncated CSV → crash).
  Fix: Write to a temporary file first (file_path + ".tmp"),
  then call os.replace() which is an atomic OS-level operation.
  The main thread will either see the complete old file or the
  complete new file — never a partial write.

STREAMLIT NOTE:
  Decorated with @st.cache_resource so Streamlit only starts
  one thread per app session, not one per UI interaction.


─────────────────────────────────────────────────────────────
FUNCTION: get_raw_data(sym)
─────────────────────────────────────────────────────────────
PURPOSE:
  Loads raw OHLCV data for a symbol. Tries sources in order:
  1. Local CSV cache (fastest — loads in <1 second)
  2. TradeStation API (full institutional-quality data)
  3. yfinance fallback (free but limited to 60 days for 5-min)

CACHING:
  @st.cache_data(ttl=4 hours) — Streamlit caches the result
  in memory. Re-runs within 4 hours reuse the cached DataFrame.
  After 4 hours, re-fetches to get today's bars.

INPUT:  Symbol string (e.g., "ES", "NQ").
OUTPUT: (DataFrame, source_string) tuple.


─────────────────────────────────────────────────────────────
FUNCTION: calculate_seasonals(df_raw, years_lookback, events,
                               tz_setting, end_reference,
                               obs_gate, sym)
─────────────────────────────────────────────────────────────
PURPOSE:
  The main computation pipeline. Slices the raw data to the
  requested lookback window, applies timezone, runs event
  filters, then calls all four seasonal functions.

CACHE KEY FIX:
  The original code captured end_dt from the module-level scope,
  meaning the cache key never included the current date.
  If the app ran across midnight, all calculations were anchored
  to yesterday's date but served from cache indefinitely.
  Fix: Pass end_dt.strftime("%Y-%m-%d") as end_reference.
  Now the cache invalidates automatically each calendar day.

PARAMETERS:
  years_lookback  → How many years of data to use (1, 3, 5, 10)
  events          → Tuple of event names to exclude
  tz_setting      → "US futures (EST)" or "Zürich (CET)"
  end_reference   → Today's date as string (cache key)
  obs_gate        → Minimum observations per slot (from slider)
  sym             → Symbol name (passed to roll exclusion)

OUTPUT: (PathData, HeatmapData, DailyPerf, HourlyVol) tuple.
        All DataFrames. Empty DataFrames if insufficient data.


─────────────────────────────────────────────────────────────
SECTION: Sidebar Controls
─────────────────────────────────────────────────────────────
  symbol           → Which futures contract to analyse
  period_label     → Lookback period (maps to years via PERIODS dict)
  event_filters    → Which macro events to strip out
  tz_mode          → All charts show times in EST or CET
  min_win_rate     → Edge Score filter: minimum win rate %
  min_profit_factor→ Edge Score filter: minimum profit factor
  max_p_value      → Edge Score filter: maximum p-value
  min_obs          → Minimum historical observations per slot


─────────────────────────────────────────────────────────────
SECTION 1: Edge Score Table
─────────────────────────────────────────────────────────────
  The primary output of the dashboard.
  Shows only time slots that simultaneously pass all three
  institutional gates. Ranked by edge_score descending.
  If the table is empty, no statistically valid edge exists
  for the current symbol/period/filter combination.
  This is correct and honest — not a bug.


─────────────────────────────────────────────────────────────
SECTION 2: Trade Recommendation
─────────────────────────────────────────────────────────────
  Driven entirely by the top row of the Edge Score table.
  Displays: best entry time, avg return, win rate (with N),
  Kelly fraction, Sharpe, Profit Factor, t-stat, p-value.
  If no slot passes the gates, shows a warning instead of
  inventing a signal (unlike the original idxmin/idxmax logic).


─────────────────────────────────────────────────────────────
SECTION 3: Average Path Chart
─────────────────────────────────────────────────────────────
  Blue line  = Mean path (avg return from weekly open)
  Orange dot = Median path (robust to outliers)
  Blue band  = 10th–90th percentile range (80% of all weeks
               fell inside this band)
  Hover      = Shows win_rate, t_stat, N, profit_factor
               for every individual time slot

  Vertical dashed lines mark day boundaries (Mon–Fri).
  X-axis is continuous_time (see compute_avg_path above).


─────────────────────────────────────────────────────────────
SECTION 4: Heatmap
─────────────────────────────────────────────────────────────
  Colour scale: Red = negative, Green = positive, Grey = NaN.
  Each cell = average daily return at that day+time combination.
  Grey cells mean no data exists — not a flat return.
  Use this to spot recurring intraday patterns by session
  (Asian open, London open, US open, US close).


─────────────────────────────────────────────────────────────
SECTION 5: Daily Performance + Volatility
─────────────────────────────────────────────────────────────
  Left panel:
    Bar chart of win rate per weekday.
    Green bars = >= 55% (institutional edge threshold).
    Red bars   = < 55% (below edge threshold).
    Table below shows all 13 metrics per day.

  Right panel:
    Orange bars = mean ATR % per hour (True Range based).
    Red dots    = 90th percentile ATR (worst-case volatility).
    Use this for position sizing: higher ATR hours require
    wider stops and smaller size.


## ═══════════════════════════════════════════════════════════
## PART 4 — QUICK REFERENCE: METRIC CHEAT SHEET
## ═══════════════════════════════════════════════════════════

  METRIC          FORMULA                   GOOD VALUE    INTERPRETATION
  ─────────────────────────────────────────────────────────────────────
  avg             mean(returns)             > 0           Positive drift
  win_rate        P(return > 0) × 100       >= 55%        Win more than lose
  profit_factor   ΣWins / |ΣLosses|         >= 1.2        Wins cover losses + margin
  t_stat          avg / (std/sqrt(N))       |t| >= 2      Signal strong vs noise
  p_value         erfc(|t|/sqrt(2))         <= 0.05       5% or less chance of randomness
  sharpe          avg/std × sqrt(252×78)    >= 0.5        Return per unit total risk
  sortino         avg/down_std × sqrt(...)  > sharpe      Penalise losses only
  kelly           W - (1-W)/(W_avg/L_avg)   > 0           Position size fraction
  count (N)       len(observations)         >= 30         Minimum for valid statistics
  p10/p90         percentiles               p10>0 ideal   Realistic outcome range
  edge_score      Sharpe×sqrt(N)×WR/100     Higher=better Composite rank


## ═══════════════════════════════════════════════════════════
## PART 5 — COMMON QUESTIONS
## ═══════════════════════════════════════════════════════════

Q: Why does the Edge Score table sometimes show zero rows?
A: No time slot passed all three gates simultaneously. This is
   correct — it means there is no statistically valid edge
   for the selected symbol/period/filter combination.
   Try relaxing min_win_rate or max_p_value in the sidebar.

Q: Why is my win_rate high but profit_factor below 1.0?
A: You are winning frequently but losing bigger amounts on
   losers than you win on winners. The strategy is not
   profitable despite high win rate. This is common in
   mean-reversion setups. Check avg_win vs avg_loss.

Q: What does a negative Kelly mean?
A: Do not trade that slot. The math says the expected value
   of betting on it is negative regardless of size.

Q: Why does the heatmap have grey cells?
A: Grey = NaN = no data for that combination of day and time.
   Could be a holiday, an early close, or outside trading hours.
   The previous version showed 0% (green/neutral) for these,
   which was misleading.

Q: The avg path looks smooth on 1-year but jagged on 10-year.
A: More data = more variance in the average path because you
   are capturing more different market regimes (bull, bear,
   2020 crash, 2022 rate hike cycle). The 10-year path is
   more statistically reliable even if visually noisier.
   Trust the t_stat and p_value, not visual smoothness.

Q: How often should I re-run / refresh data?
A: The cache TTL is 4 hours. For intraday trading decisions,
   re-fetch at the start of each session. For structural
   seasonal research, weekly is sufficient.

Q: Should I use half-Kelly or full-Kelly for sizing?
A: Always half-Kelly (kelly / 2) or less. Full-Kelly is
   theoretically optimal but assumes your statistical estimates
   are exact. In practice, estimation error is large and
   full-Kelly causes severe drawdowns.
