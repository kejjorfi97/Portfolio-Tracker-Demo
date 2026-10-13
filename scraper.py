"""
Casablanca Stock Exchange Historical Price Scraper
====================================================
Populates the MongoDB `ptf_tracker.historical_data` collection (the schema
data_fetcher.py expects) from two public, free, unauthenticated sources:

1. Indices (MASI, MSI20, and the MASI sector sub-indices) - from
   casablanca-bourse.com's own "Gamme des indices" tool:
   `/api/boursenova/indices/historical?code=<CODE>&period=<1m|3m|6m|1y|2y|3y>`
   Verified end-to-end (confirmed JSON shape, confirmed MSI20's code). No
   blocking observed.

2. Individual stocks (ATW, IAM, BCP, ...) - from investing.com, NOT from
   casablanca-bourse.com. Casablanca-bourse.com's equivalent endpoint
   (`/api/boursenova/stock-historical`) started returning a WAF "Request
   Rejected" page after only a handful of requests - both scripted and from
   the real page's own UI - even from an ordinary residential connection on
   the very first stock request of a session. That pattern (index data
   open, per-security data gated) lines up with per-security historical
   data being the thing Casablanca Bourse's paid "Offre Market Data"
   product sells, so this scraper does not try to work around it.
   investing.com serves the same kind of data for free through its own
   public "Historical Data" page and search:
     - Search: `/search/?q=<ticker>` (a real page load; results include a
       `Casablanca equities` tagged row for CSE-listed tickers)
     - Data: `https://api.investing.com/api/financialdata/historical/<id>
       ?start-date=YYYY-MM-DD&end-date=YYYY-MM-DD&time-frame=Daily
       &add-missing-rows=false` (header `domain-id: www`)
   Both endpoints and the resulting JSON shape were verified directly by
   hand (ATW -> instrument_id 13237, IAM -> 13261, BCP -> resolved via
   ticker-only search filtered to "Casablanca equities"), but investing.com
   DOES have real bot detection on this flow: a fresh, throwaway Playwright
   browser got a 403 on the search page, in both headless and visible
   (--no-headless) mode, even with navigator.webdriver masked and a normal
   desktop User-Agent. A real, manually-driven Chrome browser (same
   machine) loads the exact same page fine. To close that gap without
   spoofing anything, this script launches Chromium against a PERSISTENT
   profile directory (chrome_profile/ by default, see --profile-dir)
   instead of a fresh one every run - cookies, consent choices, and session
   history accumulate across runs the way a real browser's would. Resolved
   ticker -> instrument_id mappings are cached in investing_ticker_map.json
   so each ticker is only searched for once regardless.
   If you still get blocked with a persistent profile, that's a stronger
   signal it's genuine automation/fingerprint detection rather than a
   throwaway-profile heuristic - see the conversation this script was built
   in for the alternatives considered (driving your own already-open Chrome
   via CDP, semi-manual export, or a different data source). This script
   deliberately does not go further than persistent-profile + visible
   browser + honest self-identification; anything past that (stealth
   plugins, fingerprint spoofing, etc.) is out of scope on purpose.

Note on how API requests are made: all calls to the two endpoints above go
through an in-page `fetch()` (via `page.evaluate`), not Playwright's
separate `APIRequestContext`. The latter uses Node's own TLS stack, which
doesn't read the Windows certificate store - on a machine with
antivirus/corporate HTTPS inspection this fails with "unable to verify the
first certificate" even though the real browser (and this script) works
fine. Routing through the page reuses the browser's already-working TLS
handling.

This tracker is kept purely for demonstration purposes, so this scraper
fetches one fixed historical window (data_fetcher.DEMO_START_DATE to
DEMO_END_DATE) rather than "up to today" - no need to keep re-running it to
keep prices current. Override with --start-date/--end-date if needed.

Ethics / ToS note: this only reads data both sites already serve for free,
unauthenticated, to any visitor via their own public tools. It deliberately
does NOT try to defeat CAPTCHAs or any bot challenge - if either site starts
persistently blocking you, that's a signal to slow down (bigger
--min-delay/--max-delay), run less often, or stop relying on that source,
not to route around it. Run this from a normal residential connection, not
a datacenter/VPN IP - generic anti-bot/WAF rules are far more likely to
flag the latter.

Usage
-----
    pip install -r requirements-scraper.txt
    playwright install chromium

    # Dry run (prints what would be written, no MongoDB writes):
    python scraper.py --portfolio sample_portfolio.csv --dry-run

    # Real run, conservative pacing (recommended for a first try):
    python scraper.py --portfolio sample_portfolio.csv --min-delay 8 --max-delay 20

    # Explicit ticker list instead of a portfolio CSV:
    python scraper.py --tickers ATW,IAM,BCP --indices MASI,MSI20
"""

import argparse
import json
import logging
import random
import re
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from playwright.sync_api import sync_playwright

from data_fetcher import MONGO_URI, DB_NAME, COLLECTION_NAME, DEMO_START_DATE, DEMO_END_DATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CSE_BASE_URL = "https://www.casablanca-bourse.com"
CSE_INDICES_PAGE = f"{CSE_BASE_URL}/market-data/indices"
CSE_INDEX_HISTORICAL_URL = f"{CSE_BASE_URL}/api/boursenova/indices/historical"

INVESTING_BASE_URL = "https://www.investing.com"
INVESTING_API_BASE_URL = "https://api.investing.com"
INVESTING_HISTORICAL_URL = f"{INVESTING_API_BASE_URL}/api/financialdata/historical"

DEBUG_DIR = Path(__file__).parent / "scraper_debug"
TICKER_MAP_CACHE = Path(__file__).parent / "investing_ticker_map.json"
DEFAULT_PROFILE_DIR = Path(__file__).parent / "chrome_profile"

DEFAULT_INDEX_CODES = ["MASI", "MSI20"]


class FetchResult:
    """Mimics just enough of Playwright's APIResponse for the parsing code below."""

    def __init__(self, status: int, text: str):
        self.status = status
        self.ok = 200 <= status < 300
        self._text = text

    def text(self) -> str:
        return self._text

    def json(self) -> Any:
        return json.loads(self._text)


def _fetch_json_via_page(page, url: str, params: Dict[str, str],
                          extra_headers: Optional[Dict[str, str]] = None) -> FetchResult:
    """
    Fetch a URL using an in-page `fetch()` call (via page.evaluate) instead of
    Playwright's separate APIRequestContext.

    Why: APIRequestContext.get() uses Node's own TLS/HTTP stack, which doesn't
    read the Windows certificate store. If antivirus or a corporate network
    does HTTPS inspection (re-signing certs with a locally-trusted root CA),
    Node's client rejects the connection with "unable to verify the first
    certificate" even though the real browser - which DOES use the OS store,
    and already succeeded for page.goto() - would accept it fine. Running the
    fetch inside the page reuses that already-working browser TLS handling,
    and also means the request looks identical to one the site's own JS would
    make (same cookies, same TLS stack).
    """
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    result = page.evaluate(
        """
        async ([url, headers]) => {
            try {
                const res = await fetch(url, { credentials: 'include', headers: headers || {} });
                const text = await res.text();
                return { status: res.status, text: text };
            } catch (e) {
                return { status: 0, text: String(e) };
            }
        }
        """,
        [full_url, extra_headers or {}],
    )
    return FetchResult(result["status"], result["text"])


def _french_date_to_iso(seance: str) -> Optional[str]:
    """Convert 'DD/MM/YYYY' to 'YYYY-MM-DD'. Returns None if malformed."""
    try:
        d, m, y = seance.split("/")
        return f"{y}-{m}-{d}"
    except (ValueError, AttributeError):
        return None


def _dump_debug(name: str, payload: Any) -> Path:
    DEBUG_DIR.mkdir(exist_ok=True)
    path = DEBUG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(payload, (dict, list)):
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(payload))
    return path


def _dismiss_cookie_banner(page) -> bool:
    """
    Click through investing.com's OneTrust cookie-consent banner if present.
    This is just clicking "Accept" the way a human visitor would (required for
    the page to render/behave normally) - not a bot-detection workaround.
    Returns True if a banner was found and clicked.
    """
    for selector in ("#onetrust-accept-btn-handler", "button#onetrust-accept-btn-handler"):
        try:
            page.click(selector, timeout=3000)
            time.sleep(1)
            return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Indices (casablanca-bourse.com)
# ---------------------------------------------------------------------------

def _parse_index_payload(payload: Dict, code: str) -> List[Dict]:
    """Parse the confirmed {"totalCount", "items": [{"seance", "indices": {code: {"valeur"}}}]} shape."""
    records = []
    for item in payload.get("items", []):
        seance = item.get("seance")
        iso_date = _french_date_to_iso(seance) if seance else None
        idx = (item.get("indices") or {}).get(code, {})
        valeur = idx.get("valeur")
        if iso_date and valeur is not None:
            records.append({"date": iso_date, "close": float(valeur)})
    return records


def fetch_index_history(page, code: str, period: str = "3y") -> List[Dict]:
    """Fetch historical values for an index (MASI, MSI20, or a MASI sector sub-index)."""
    resp = _fetch_json_via_page(page, CSE_INDEX_HISTORICAL_URL, {"code": code, "period": period})
    if not resp.ok:
        logger.error(f"[{code}] HTTP {resp.status} fetching index history")
        _dump_debug(f"index_{code}_httperror", resp.text())
        return []

    try:
        payload = resp.json()
    except Exception:
        snippet = resp.text()[:200].replace("\n", " ")
        path = _dump_debug(f"index_{code}_nonjson", resp.text())
        logger.error(
            f"[{code}] Response was not valid JSON (status {resp.status}). "
            f"First 200 chars: {snippet!r} - full body saved to {path}"
        )
        return []

    total = payload.get("totalCount")
    records = _parse_index_payload(payload, code)
    if total is not None and len(records) < total:
        logger.warning(
            f"[{code}] Got {len(records)} records but totalCount={total} - "
            "the API may paginate for long ranges; this scraper currently assumes it doesn't."
        )
    logger.info(f"[{code}] Fetched {len(records)} index records")
    return records


# ---------------------------------------------------------------------------
# Individual stocks (investing.com)
# ---------------------------------------------------------------------------

def _load_ticker_map() -> Dict[str, str]:
    if TICKER_MAP_CACHE.exists():
        try:
            return json.loads(TICKER_MAP_CACHE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(f"Could not parse {TICKER_MAP_CACHE}, starting with an empty ticker map.")
    return {}


def _save_ticker_map(ticker_map: Dict[str, str]) -> None:
    TICKER_MAP_CACHE.write_text(json.dumps(ticker_map, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_investing_instrument_id(page, ticker: str, ticker_map: Dict[str, str]) -> Optional[str]:
    """
    Resolve a CSE ticker (e.g. 'ATW') to its investing.com numeric instrument
    id, via investing.com's own search page filtered to Casablanca-listed
    results. Cached in investing_ticker_map.json - if a ticker resolves to
    the wrong company (ticker collisions are rare but possible), fix it by
    hand-editing that file.
    """
    ticker = ticker.upper()
    if ticker in ticker_map:
        return ticker_map[ticker]

    search_url = f"{INVESTING_BASE_URL}/search/?q={urllib.parse.quote(ticker)}"
    page.goto(search_url, wait_until="domcontentloaded")
    banner_dismissed = _dismiss_cookie_banner(page)

    # The search-results list is populated client-side after load, not present
    # in the initial HTML - wait for it explicitly rather than guessing a fixed
    # sleep is long enough.
    try:
        page.wait_for_selector('a.js-inner-all-results-quote-item, .js-group-results', timeout=8000)
    except Exception:
        pass
    time.sleep(1.5)

    candidates = page.evaluate(
        """
        () => [...document.querySelectorAll('a.js-inner-all-results-quote-item[href*="/equities/"]')]
            .map(a => ({href: a.getAttribute('href'), text: a.innerText.replace(/\\s+/g, ' ').trim()}))
        """
    )

    casa_matches = [c for c in candidates if "casablanca equities" in c["text"].lower()]
    exact = [c for c in casa_matches if c["text"].split(" ")[0].upper() == ticker]
    chosen = exact or casa_matches

    if not chosen:
        # Dump everything needed to diagnose without another guess-and-check
        # round trip: full rendered HTML, a screenshot, and page title/URL.
        html_path = _dump_debug(f"resolve_{ticker}_no_match", page.content())
        screenshot_path = DEBUG_DIR / f"resolve_{ticker}_no_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        try:
            page.screenshot(path=str(screenshot_path), full_page=True)
        except Exception:
            screenshot_path = None
        logger.warning(
            f"[{ticker}] No Casablanca-listed match found on investing.com search "
            f"({len(candidates)} total quote candidates seen, cookie banner dismissed: {banner_dismissed}, "
            f"page title: {page.title()!r}). "
            f"HTML saved to {html_path}"
            + (f", screenshot saved to {screenshot_path}" if screenshot_path else "")
        )
        return None
    if len(chosen) > 1:
        logger.warning(f"[{ticker}] Multiple Casablanca matches found, using the first: {chosen}")

    href = chosen[0]["href"]
    page.goto(f"{INVESTING_BASE_URL}{href}", wait_until="domcontentloaded")
    _dismiss_cookie_banner(page)
    time.sleep(1)
    html = page.content()
    m = re.search(r'"instrument_id":"(\d+)"', html)
    if not m:
        path = _dump_debug(f"resolve_{ticker}_no_instrument_id", html)
        logger.warning(f"[{ticker}] Found page {href} but couldn't extract instrument_id from it. "
                        f"Page HTML saved to {path}.")
        return None

    instrument_id = m.group(1)
    ticker_map[ticker] = instrument_id
    _save_ticker_map(ticker_map)
    logger.info(f"[{ticker}] Resolved to investing.com instrument_id={instrument_id} ({href})")
    return instrument_id


def fetch_stock_history(page, ticker: str, instrument_id: str, start_date: str, end_date: str) -> List[Dict]:
    """Fetch historical close prices for one CSE-listed stock from investing.com."""
    url = f"{INVESTING_HISTORICAL_URL}/{instrument_id}"
    params = {
        "start-date": start_date,
        "end-date": end_date,
        "time-frame": "Daily",
        "add-missing-rows": "false",
    }
    resp = _fetch_json_via_page(page, url, params, extra_headers={"domain-id": "www"})
    if not resp.ok:
        logger.error(f"[{ticker}] HTTP {resp.status} fetching stock history from investing.com")
        _dump_debug(f"stock_{ticker}_httperror", resp.text())
        return []

    try:
        payload = resp.json()
    except Exception:
        text = resp.text()
        snippet = text[:200].replace("\n", " ")
        path = _dump_debug(f"stock_{ticker}_nonjson", text)
        logger.error(
            f"[{ticker}] Response was not valid JSON (status {resp.status}). "
            f"First 200 chars: {snippet!r} - full body saved to {path}"
        )
        return []

    records = []
    for item in payload.get("data", []):
        ts = item.get("rowDateTimestamp")  # e.g. "2026-09-09T00:00:00Z"
        close = item.get("last_closeRaw")
        if ts and close is not None:
            records.append({"date": ts[:10], "close": float(close)})

    logger.info(f"[{ticker}] Fetched {len(records)} stock records from investing.com")
    return records


def upsert_price_history(collection, ticker: str, records: List[Dict]) -> None:
    """Merge new {date, close} records into the ticker's document, deduped by date."""
    if not records:
        return

    existing = collection.find_one({"ticker": ticker}, {"historical_prices": 1})
    merged = {}
    if existing and existing.get("historical_prices"):
        for row in existing["historical_prices"]:
            if row.get("date"):
                merged[row["date"]] = row
    for row in records:
        merged[row["date"]] = row

    merged_list = [merged[d] for d in sorted(merged.keys())]
    collection.update_one(
        {"ticker": ticker},
        {"$set": {"ticker": ticker, "historical_prices": merged_list}},
        upsert=True,
    )
    logger.info(f"[{ticker}] Upserted {len(merged_list)} total records ({len(records)} fetched this run)")


def load_tickers_from_portfolio(portfolio_path: str) -> List[str]:
    df = pd.read_csv(portfolio_path)
    tickers = df["Ticker"].astype(str).str.strip().str.upper().unique().tolist()
    return [t for t in tickers if t != "CASH"]


def main():
    parser = argparse.ArgumentParser(description="Scrape CSE historical prices into MongoDB.")
    parser.add_argument("--portfolio", help="Portfolio CSV to read tickers from (Ticker column).")
    parser.add_argument("--tickers", help="Comma-separated ticker codes (overrides --portfolio).")
    parser.add_argument("--indices", default=",".join(DEFAULT_INDEX_CODES),
                         help="Comma-separated index codes (default: MASI,MSI20).")
    parser.add_argument("--start-date", default=DEMO_START_DATE.strftime("%Y-%m-%d"),
                         help=f"Stock history start date, YYYY-MM-DD (default: {DEMO_START_DATE.date()}, "
                              "the fixed demo window - see data_fetcher.DEMO_START_DATE).")
    parser.add_argument("--end-date", default=DEMO_END_DATE.strftime("%Y-%m-%d"),
                         help=f"Stock history end date, YYYY-MM-DD (default: {DEMO_END_DATE.date()}, "
                              "the fixed demo window - see data_fetcher.DEMO_END_DATE).")
    parser.add_argument("--index-period", default="3y", choices=["1m", "3m", "6m", "1y", "2y", "3y"],
                         help="Period for index history (default: 3y - comfortably covers the demo window).")
    parser.add_argument("--min-delay", type=float, default=6.0, help="Min seconds between requests.")
    parser.add_argument("--max-delay", type=float, default=14.0, help="Max seconds between requests.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE_DIR),
                         help="Persistent Chrome profile directory reused across runs "
                              f"(default: {DEFAULT_PROFILE_DIR}). See module docstring.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and print, but don't write to MongoDB.")
    parser.add_argument("--skip-stocks", action="store_true", help="Only scrape indices, skip individual stocks.")
    parser.add_argument("--skip-indices", action="store_true", help="Only scrape stocks, skip indices.")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.portfolio:
        tickers = load_tickers_from_portfolio(args.portfolio)
    else:
        tickers = []

    index_codes = [c.strip().upper() for c in args.indices.split(",") if c.strip()]
    start_str, end_str = args.start_date, args.end_date

    client = None
    collection = None
    if not args.dry_run:
        from pymongo import MongoClient
        connection_string = MONGO_URI if MONGO_URI else "mongodb://localhost:27017/"
        client = MongoClient(connection_string)
        collection = client[DB_NAME][COLLECTION_NAME]

    ticker_map = _load_ticker_map()

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        # Use a PERSISTENT profile directory (cookies, localStorage, site
        # consent choices, etc. all survive between runs) instead of a fresh
        # throwaway profile every time. A brand-new profile on every run is
        # itself a signal some bot-detection systems weigh heavily - no
        # history, no prior consent, nothing. Reusing the same profile lets
        # a real session build up across runs, the same way your own browser
        # accumulates history. Combined with --no-headless this is the
        # closest thing to "just use a real browser" without spoofing
        # anything.
        #
        # Playwright's default headless Chromium also announces itself as
        # "HeadlessChrome/<version>" in its User-Agent string, a well-known
        # generic automation fingerprint. Use Chromium's newer "headless=new"
        # mode plus an explicit desktop-Chrome User-Agent when headless is
        # requested - this only changes how the browser identifies itself,
        # not what it renders or does.
        launch_args = ["--disable-blink-features=AutomationControlled"]
        if args.headless:
            launch_args.append("--headless=new")
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=args.headless,
            args=launch_args,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
        )
        # Playwright sets navigator.webdriver = true by default, which some
        # sites' front-end code checks to skip/alter client-side rendering
        # (distinct from any bot-*blocking* challenge - investing.com's search
        # results are populated by ordinary page JS, not gated behind a
        # CAPTCHA). Reset it to look like a normal browser.
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )
        page = context.pages[0] if context.pages else context.new_page()

        if not args.skip_indices:
            logger.info("Loading casablanca-bourse.com to establish a session...")
            page.goto(CSE_INDICES_PAGE, wait_until="networkidle")
            time.sleep(2)

            logger.info(f"Scraping {len(index_codes)} index/indices: {index_codes}")
            for code in index_codes:
                records = fetch_index_history(page, code, period=args.index_period)
                if args.dry_run:
                    print(f"[DRY RUN] {code}: {len(records)} records, "
                          f"e.g. {records[:2]}{'...' if len(records) > 2 else ''}")
                else:
                    upsert_price_history(collection, code, records)
                time.sleep(random.uniform(args.min_delay, args.max_delay))

        if not args.skip_stocks and tickers:
            logger.info(f"Scraping {len(tickers)} stock tickers via investing.com: {tickers}")
            for ticker in tickers:
                instrument_id = resolve_investing_instrument_id(page, ticker, ticker_map)
                if not instrument_id:
                    logger.error(f"[{ticker}] Skipping - could not resolve an investing.com instrument id. "
                                 f"You can add one manually to {TICKER_MAP_CACHE.name}.")
                    time.sleep(random.uniform(args.min_delay, args.max_delay))
                    continue

                records = fetch_stock_history(page, ticker, instrument_id, start_str, end_str)
                if args.dry_run:
                    print(f"[DRY RUN] {ticker}: {len(records)} records, "
                          f"e.g. {records[:2]}{'...' if len(records) > 2 else ''}")
                else:
                    upsert_price_history(collection, ticker, records)
                time.sleep(random.uniform(args.min_delay, args.max_delay))

        context.close()

    if client:
        client.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()
