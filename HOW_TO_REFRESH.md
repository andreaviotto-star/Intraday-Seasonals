# How to Refresh Intraday Seasonals Data

## Dashboard URL (bookmark this!)
https://intraday-seasonals.streamlit.app/

---

## Weekly Data Refresh (every Monday, 2 min)

### Step 1: Open Codespace
1. Go to https://github.com/codespaces
2. Click "Intraday-Seasonals" codespace
3. Wait for it to load (green = ready)

### Step 2: Open Terminal
- Press Ctrl+` (backtick) to open terminal
- You should see: @andreaviotto-star /workspaces/Intraday-Seasonals

### Step 3: Fetch Fresh Data
cd /workspaces/Intraday-Seasonals
python ts_data_fetcher.py

- Browser opens → Login TradeStation → click Allow
- Wait for all symbols to download (ES NQ ZN GC HG SI DX 6E CL)
- You see: "DONE - Downloaded X total bars"

### Step 4: Push to GitHub
git add data/
git commit -m "weekly data update"
git push origin main

### Step 5: Done!
- Wait 2 min → dashboard auto-updates
- Visit https://intraday-seasonals.streamlit.app/
- Check top of dashboard for latest data date

---

## Troubleshooting

### Dashboard shows old data
→ Repeat Steps 3-4 above

### TradeStation login fails
→ Run: python get_ts_token.py
→ Follow browser login steps

### Streamlit error page
→ Go to dashboard → bottom right "Manage app" → "Reboot app"

### Codespace deleted/expired
→ github.com/andreaviotto-star/Intraday-Seasonals
→ Click green "Code" → "Codespaces" → "New codespace"
→ Run: pip install -r requirements.txt
→ Then follow Step 3 above

---

## Files Reference
| File | Purpose |
|------|---------|
| app.py | Main dashboard |
| config.py | Symbols + settings |
| ts_data_fetcher.py | Fetches TradeStation data |
| get_ts_token.py | Auth helper (first time) |
| utils/seasonal_stats.py | Calculations |
| data/*.csv | Cached price data |
| requirements.txt | Python packages |
