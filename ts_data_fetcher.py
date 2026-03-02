# ts_data_fetcher.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

TS_CLIENT_ID = os.getenv("TRADESTATION_CLIENT_ID")
TS_CLIENT_SECRET = os.getenv("TRADESTATION_CLIENT_SECRET")
TS_REFRESH_TOKEN = os.getenv("TRADESTATION_REFRESH_TOKEN") 

YFINANCE_TICKER_MAP = {
    "ES": "ES=F", "NQ": "NQ=F", "ZN": "ZN=F", "GC": "GC=F",
    "HG": "HG=F", "SI": "SI=F", "DX": "DX=F", "6E": "EURUSD=X", "CL": "CL=F"
}

# FIXED: Added the '@' prefix which TradeStation requires for continuous futures contracts
TS_SYMBOL_MAP = {
    "ES": "@ES", "NQ": "@NQ", "ZN": "@TY", # TradeStation uses TY for 10-Yr Notes
    "GC": "@GC", "HG": "@HG", "SI": "@SI", 
    "DX": "@DX", "6E": "@EC", # TradeStation uses EC for Euro FX
    "CL": "@CL"
}

class TradeStationFetcher:
    def __init__(self):
        self.base_url = "https://api.tradestation.com/v3"
        self.access_token = None

    def authenticate(self):
        if not TS_CLIENT_ID or not TS_CLIENT_SECRET or not TS_REFRESH_TOKEN:
            raise ValueError("TradeStation Credentials or Refresh Token missing from .env")

        token_url = "https://signin.tradestation.com/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "client_id": TS_CLIENT_ID,
            "client_secret": TS_CLIENT_SECRET,
            "refresh_token": TS_REFRESH_TOKEN
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(token_url, data=payload, headers=headers)
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
        else:
            raise Exception(f"Failed to authenticate TS: {response.text}")

    def get_intraday_bars(self, symbol: str, start: datetime, end: datetime, interval="5"):
        if not self.access_token:
            self.authenticate()

        ts_sym = TS_SYMBOL_MAP.get(symbol, symbol)
        
        # Calculate how many bars we need total
        total_days = (end - start).days
        bars_needed = int(total_days * 24 * (60 / int(interval)))
        
        all_bars = []
        chunk_size = 50000  
        current_lastdate = end

        while bars_needed > 0:
            fetch_count = min(chunk_size, bars_needed)
            
            url = f"{self.base_url}/marketdata/barcharts/{ts_sym}"
            params = {
                "interval": interval,
                "unit": "Minute",
                "sessiontemplate": "USEQ24Hour", # Force TS to return the overnight sessions
                "barsback": fetch_count,
                "lastdate": current_lastdate.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            resp = requests.get(url, params=params, headers=headers)
            
            if resp.status_code == 200:
                bars = resp.json().get("Bars", [])
                if not bars:
                    break 
                
                oldest_time_str = bars[0]["TimeStamp"]
                current_lastdate = datetime.strptime(oldest_time_str, "%Y-%m-%dT%H:%M:%SZ") - timedelta(minutes=1)
                
                all_bars.extend(bars)
                bars_needed -= len(bars)
            else:
                print(f"TS API Error fetching chunk for {ts_sym}: {resp.text}")
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df = df.rename(columns={
            "TimeStamp": "datetime",
            "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"
        })
        
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.drop_duplicates(subset=["datetime"])
        df = df.set_index("datetime").sort_index()
        df.index = df.index.tz_convert('UTC')
        
        # Format as float
        df = df[["Open", "High", "Low", "Close"]].astype(float)
        
        start_utc = pd.to_datetime(start, utc=True)
        df = df[df.index >= start_utc]
        
        # Resample to continuous 5-minute bars to eliminate overnight visual gaps
        df = df.resample("5min").ffill()
        
        return df


def fetch_yfinance(symbol: str, start: datetime, end: datetime, freq="5m"):
    ticker = YFINANCE_TICKER_MAP.get(symbol, f"{symbol}=F")
    all_data = []
    current_start = start
    
    while current_start < end:
        chunk_end = min(current_start + timedelta(days=7), end)
        chunk = yf.download(ticker, start=current_start, end=chunk_end, 
                           interval=freq, progress=False)
        if not chunk.empty:
            all_data.append(chunk)
        current_start = chunk_end
    
    if not all_data:
        raise ValueError(f"No data for {ticker}")
    
    df = pd.concat(all_data).dropna()
    df = df.resample(freq).ffill().dropna()
    
    return df
