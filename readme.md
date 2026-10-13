# 📊 CSE Portfolio Intelligence App

This app provides advanced portfolio analytics, risk metrics, and trade insights for **Casablanca Stock Exchange (CSE)** portfolios through a user-friendly interface. It supports portfolio uploads and trade logs, analyzes sector allocation, risk-adjusted performance, and generates AI-assisted investment memos.

Historical prices are pulled from a **MongoDB** collection (`ptf_tracker.historical_data`) rather than Yahoo Finance, and all figures are denominated in **MAD**.

This tracker is kept purely for **demonstration purposes**. Every analysis run uses one fixed historical window - **2025-01-02 to 2026-09-15** (`DEMO_START_DATE` / `DEMO_END_DATE` in `data_fetcher.py`) - instead of "up to today", so there's no need to keep MongoDB continuously updated with fresh prices for every holding. "Current price" and "today" throughout the app effectively mean the end of that window, not the real current date.

---

## 🔧 How to Use

### 1. Portfolio Upload
Upload a CSV file with your current holdings. Required columns:
- `Ticker`: CSE ticker symbol (e.g. ATW, IAM, BCP), or `CASH` for a cash position
- `Quantity`: Number of shares (negative for short positions)
- `Category`: Sector (e.g. Banks, Telecommunications, Real Estate, etc.)
- `Cost_Basis_per_Share`: Purchase price per share, in MAD

**Example:**
```
Ticker,Quantity,Category,Cost_Basis_per_Share
ATW,100,Banks,450.0
IAM,50,Telecommunications,95.0
TLT,15,Fixed Income,100.0
```

Please note the following:

- **Ticker** must be the column name, and must match a `ticker` value in the MongoDB `historical_data` collection (or be `CASH`).
- **Category** is optional. If provided, it will be kept as the sector label.
- If **Category** is missing, a static Casablanca sector map (`sector_classification.py`) is used as a fallback.
- Some tickers may remain "Unclassified" if they aren't in that map yet — extend `CSE_SECTOR_MAP` in `sector_classification.py` as needed.
- Zero cost basis is not explicitly handled, so users should avoid or preprocess these cases.
- AI-generated insights require a valid `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the `.env` file; without one, detailed fallback recommendations are used instead.

---

### 2. Trade Log Upload
Upload a CSV file representing your historical transactions and cash movements. Required columns:
- `Date`: Format YYYY-MM-DD
- `Ticker`: Ticker symbol (ignored for `Deposit`/`Withdrawal`; optional for `Dividend`)
- `Action`: `Buy`, `Sell`, `Deposit`, `Withdrawal`, or `Dividend`
- `Quantity`: Number of shares (share trades), or any factor such that `Quantity × Price` equals the cash amount (cash events - e.g. `Quantity=1, Price=5000` for a 5,000 MAD deposit, or `Quantity=<shares held>, Price=<dividend per share>` for a dividend)
- `Price`: Trade price per share (MAD), or per-unit amount for cash events
- `Category`: Same as portfolio (used for consistency)
- `Notes`: Optional notes
- `Conviction_Score`: A score from 1 to 10 representing how confident you were in the trade (share trades only)

**Example:**
```
Date,Ticker,Action,Quantity,Price,Category,Notes,Conviction_Score
2024-02-01,ATW,Buy,10,420.0,Banks,Long-term belief,9
2024-04-15,ATW,Sell,10,470.0,Banks,Profit taking,9
2024-05-05,IAM,Buy,5,90.0,Telecommunications,Defensive add,6
2024-06-01,,Deposit,1,10000.0,Cash,Added funds,
2024-07-15,IAM,Dividend,5,3.2,Telecommunications,Annual dividend,
2024-08-01,,Withdrawal,1,2000.0,Cash,Personal expense,
```

**Cash handling:** Uploading a trade log reconciles the portfolio's `CASH` position for this analysis: it starts from the `CASH` row's quantity in the portfolio CSV (0 if there isn't one) and adds deposits, dividends, and sell proceeds, then subtracts withdrawals and buy costs. Portfolio Value, weights, and P&L all reflect this reconciled balance - see the Cash Flow Summary at the top of the Trade Log tab for a breakdown, and `reconcile_cash_flows()` / `calculate_realized_pnl()` in `streamlit_app.py`.

---

## 🧠 Features

- 📈 **Performance Overview**: Track NAV, returns, volatility, and compare vs the MSI20 benchmark.
- 📊 **Sector Allocation**: View portfolio structure by ticker and by Casablanca market sector (descriptive - no fixed targets).
- 📉 **Risk Metrics**: See Sharpe, Sortino, Omega, Tail ratios with visual gauges.
- 🔎 **Correlation Alerts**: Identify highly correlated asset pairs, flagged as same-sector concentration when applicable.
- 🎯 **Sector Concentration Risk**: Flags when a single sector dominates the portfolio.
- 🌡️ **Market Regime Detection**: Bull/Bear/Neutral based on MSI20 trend and its own trailing realized volatility (no Moroccan VIX-equivalent exists, so realized volatility stands in for an implied-vol fear gauge).
- 🧾 **Trade Log Summary**: Understand gains/losses and conviction impact.
- 💸 **Tax Loss Harvesting**: Flags unrealized losses worth considering for a sale, with an estimated tax saving at a configurable flat rate (default 15% capital gains / 15% dividends - see `PortfolioAnalyzer.__init__` in `analysis_engine.py`).
- 🧠 **AI-Generated Insights**: Executive investment memo with prioritized, data-driven recommendations.

Not included (removed as part of the Moroccan-market adaptation, since neither has a real equivalent on the CSE): US-ETF-based "Dragon Portfolio" target allocation, AI-sector-specific concentration scoring, and listed-options strategy suggestions.

---

## 🗄️ Data Source

`data_fetcher.py` reads historical close prices from MongoDB:
- Connection: `MONGO_URI` environment variable (falls back to `mongodb://localhost:27017/`)
- Database: `ptf_tracker`, Collection: `historical_data`
- Each document: `{"ticker": "ATW", "historical_prices": [{"date": ..., "close": ...}, ...]}`
- Benchmark: `MSI20` (primary), with `MASI` fetched alongside it by default
- `CASH` is a special ticker with a constant price of 1 and needs no MongoDB document

### Populating MongoDB: `scraper.py`

`scraper.py` fills the `historical_data` collection from two free, unauthenticated public sources - no login/API key needed for either:

- **Indices (MASI, MSI20)** - from casablanca-bourse.com's own "Gamme des indices" tool. Fully verified: confirmed JSON shape, no blocking observed.
- **Individual stocks (ATW, IAM, BCP, ...)** - from **investing.com**, not casablanca-bourse.com. Casablanca Bourse's equivalent per-stock endpoint started returning a WAF "Request Rejected" page after only a few requests, even from a normal residential connection - that (index data open, per-security data gated) lines up with per-security history being what their paid "Offre Market Data" product sells, so this scraper doesn't try to work around it. investing.com serves the same data for free through its own search and historical-data pages, but it does have real bot detection on that flow (a fresh automated browser gets a 403 there, headless or not, even with a normal User-Agent). To close that gap honestly, the script launches Chrome against a **persistent profile directory** (`chrome_profile/` by default, see `--profile-dir`) instead of a throwaway one each run, so cookies/session history build up across runs the way a real browser's would - it does not use any fingerprint-spoofing or stealth tooling. Each ticker is resolved to its investing.com numeric ID once (via search, filtered to "Casablanca equities" results) and cached in `investing_ticker_map.json`, so repeat runs skip straight to fetching data.

```bash
pip install -r requirements-scraper.txt
playwright install chromium

# See what would be fetched without touching MongoDB:
python scraper.py --portfolio sample_portfolio.csv --dry-run

# Real run (start conservative - see notes below):
python scraper.py --portfolio sample_portfolio.csv --min-delay 8 --max-delay 20
```

By default this fetches the same fixed demonstration window as the rest of the app (2025-01-02 to 2026-09-15) - override with `--start-date`/`--end-date` if you ever need a different range. Since the app only ever analyzes that one window, there's no need to re-run this scraper to keep prices "current".

Notes, also documented in `scraper.py`'s module docstring:

- If a ticker resolves to the wrong company on investing.com (ticker symbols occasionally collide across exchanges worldwide), fix it by hand-editing `investing_ticker_map.json` - delete that ticker's entry and rerun, or set the correct numeric ID directly (open the ticker's page on investing.com and look for `"instrument_id":"<id>"` in the page source).
- Run this from a normal residential connection, not a VPN/datacenter IP - generic anti-bot rules are far more likely to flag those. If either site starts persistently blocking you, treat it as a signal to increase `--min-delay`/`--max-delay` or run less often, not to route around it.
- First run(s) against investing.com: use `--no-headless` so a visible Chrome window opens with the persistent profile - this gives that profile the best chance of looking like an ordinary browsing session. If you still get a 403 in `scraper_debug/` after a real visible run, that's a stronger signal it's genuine bot detection rather than something this script's pacing/settings can fix - at that point individual-stock automation from investing.com isn't reliable and it's worth reconsidering the data source rather than pushing further.

---

## 📤 Export

All analytics and visualizations are available for export as CSV files for easy sharing and documentation.

---

## ⚠️ Notes

- Conviction scores are optional but recommended for richer insights.
- Trade logs must match the tickers in the portfolio, mismatches are flagged automatically.
- The risk-free rate (default ~2.75%), and both tax rates (default 15%/15%), are configurable constructor parameters on `PortfolioAnalyzer` — update them to the current Moroccan T-bill/BAM rate and actual tax treatment as needed.
- Realized-volatility thresholds used for regime detection (`low_vol_threshold` / `high_vol_threshold`, default 15% / 25%) are starting approximations — tune them once more MSI20 history is available.

---

## 🚀 Deployment (Streamlit Community Cloud)

This repo is ready to push to GitHub and deploy on [share.streamlit.io](https://share.streamlit.io) - it's free, and `requirements.txt` is kept minimal (no `playwright`, no unused heavy packages) so the build is fast. `.gitignore` already excludes `.env` and the scraper's local-only artifacts.

1. **Push to GitHub** (not yet connected to a remote):
   ```bash
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
   Create the (empty) repo on GitHub first if you haven't - public or private both work with Streamlit Community Cloud.

2. **MongoDB Atlas network access**: your `MONGO_URI` is an Atlas (`mongodb+srv://`) connection string, which is cloud-reachable - but Atlas's **Network Access** list must allow connections from Streamlit Cloud. Since Streamlit Cloud doesn't publish fixed outbound IPs, the simplest option is to add `0.0.0.0/0` (allow from anywhere) in Atlas's Network Access settings. Your Atlas user/password still gates access, so this only removes the IP allowlist layer, not authentication.

3. **Deploy**: go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click "New app", pick this repo/branch, and set the main file path to `streamlit_app.py`.

4. **Add secrets**: in the app's Settings → Secrets, paste (TOML format, same keys as `.env.example`):
   ```toml
   MONGO_URI = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
   OPENAI_API_KEY = ""
   ANTHROPIC_API_KEY = ""
   ```
   Streamlit exposes these as environment variables, so `os.environ.get("MONGO_URI")` in `data_fetcher.py` picks them up with no code changes. Leave the AI keys blank if you don't want to wire them up yet - the app falls back to rule-based recommendations.

5. **Populate MongoDB** (if you haven't already) by running `scraper.py` locally against the same Atlas cluster before or after deploying - the deployed app only reads from MongoDB, it never writes to it.

---

## 💬 Support

If you encounter issues uploading your data, please double-check the column names and formats. Still stuck? Reach out and I'll be happy to assist.
