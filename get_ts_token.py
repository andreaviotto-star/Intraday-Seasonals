# get_ts_token.py
import os
import urllib.parse
import requests
from dotenv import load_dotenv

# Load your existing .env file
load_dotenv()

CLIENT_ID = os.getenv("TRADESTATION_CLIENT_ID")
CLIENT_SECRET = os.getenv("TRADESTATION_CLIENT_SECRET")
REDIRECT_URI = "http://localhost"

if not CLIENT_ID or not CLIENT_SECRET:
    print("Error: TRADESTATION_CLIENT_ID and TRADESTATION_CLIENT_SECRET not found in .env")
    exit()

# 1. Generate Auth URL
auth_url = (
    f"https://signin.tradestation.com/authorize"
    f"?response_type=code"
    f"&client_id={CLIENT_ID}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
    f"&audience=https://api.tradestation.com"
    f"&scope=offline_access%20MarketData"
)

print("\n" + "="*60)
print("STEP 1: Click this link to open TradeStation login:")
print(auth_url)
print("="*60 + "\n")

# 2. Get the redirected URL from the user
print("STEP 2: Log into TradeStation in your browser.")
print("After logging in, you will be redirected to a blank 'localhost' page.")
redirected_url = input("Copy the ENTIRE URL from your browser's address bar and paste it here:\n> ")

# Extract the 'code' parameter
try:
    parsed_url = urllib.parse.urlparse(redirected_url)
    params = urllib.parse.parse_qs(parsed_url.query)
    auth_code = params['code'][0]
except Exception as e:
    print("\n❌ Error: Could not find 'code=' in the URL. Make sure you copied the exact redirected link.")
    exit()

# 3. Exchange the Code for a Refresh Token
print("\nSTEP 3: Exchanging code for tokens...")
token_url = "https://signin.tradestation.com/oauth/token"
payload = {
    "grant_type": "authorization_code",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "code": auth_code,
    "redirect_uri": REDIRECT_URI
}
headers = {"Content-Type": "application/x-www-form-urlencoded"}

response = requests.post(token_url, data=payload, headers=headers)

if response.status_code == 200:
    data = response.json()
    refresh_token = data.get("refresh_token")
    print("\n✅ SUCCESS! Add this exact line to your .env file:\n")
    print(f"TRADESTATION_REFRESH_TOKEN={refresh_token}\n")
else:
    print("\n❌ Failed to get token:", response.text)

