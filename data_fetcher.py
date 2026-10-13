"""
Portfolio Data Fetcher Module (Casablanca Stock Exchange)
==========================================================
Fetches historical and current price data from a MongoDB "historical_data"
collection instead of Yahoo Finance.

Author: Dragon Portfolio Analytics Engine
Version: 2.0.0
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env here, not just in the importing script: MONGO_URI is read at
# module level below, so if some other module imports this one before
# calling load_dotenv() itself (e.g. streamlit_app.py calls it only after
# `from analysis_engine import ...`, which already pulls this module in),
# os.environ.get("MONGO_URI") would capture None permanently.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "ptf_tracker"
COLLECTION_NAME = "historical_data"

# MSI20 is the primary benchmark used across the app; MASI is fetched
# alongside it by default in case it's needed for future features.
BENCHMARK_TICKERS = ["MASI", "MSI20"]

# This tracker is kept purely for demonstration purposes: rather than always
# analyzing "up to today" (which would need continuously updated prices for
# every held ticker), every run of the app analyzes this one fixed
# historical window. "Current price" / "today" throughout the app therefore
# effectively means DEMO_END_DATE, not the real current date.
DEMO_START_DATE = datetime(2025, 1, 2)
DEMO_END_DATE = datetime(2026, 9, 15)


def get_historical_prices(tickers: List[str], start_date, benchmarks: List[str] = None,
                           end_date=None) -> pd.DataFrame:
    """
    Fetch historical close prices for the given tickers (+ benchmarks) from MongoDB.

    Args:
        tickers: Portfolio ticker symbols (as used in the MongoDB 'ticker' field).
        start_date: Start of the date range (datetime or ISO string).
        benchmarks: Benchmark tickers to fetch alongside the portfolio (default: MASI, MSI20).
        end_date: End of the date range. Defaults to today if not provided.

    Returns:
        DataFrame indexed by date, one column per ticker, forward-filled.
    """
    if benchmarks is None:
        benchmarks = BENCHMARK_TICKERS
    if end_date is None:
        end_date = datetime.now()

    connection_string = MONGO_URI if MONGO_URI else "mongodb://localhost:27017/"
    
    client = MongoClient(connection_string)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Preserve order, drop duplicates; CASH has no document in Mongo (constant price).
    all_tickers = list(dict.fromkeys(tickers + benchmarks))
    query_tickers = [t for t in all_tickers if t != 'CASH']

    result_dfs = []
    try:
        if query_tickers:
            cursor = collection.find({"ticker": {"$in": query_tickers}})
            for document in cursor:
                ticker = document["ticker"]
                prices_data = document.get("historical_prices")
                if not prices_data:
                    logger.warning(f"No historical_prices found for ticker {ticker}")
                    continue

                prices = pd.DataFrame(prices_data)[["date", "close"]]
                prices.columns = ['date', ticker]
                prices['date'] = pd.to_datetime(prices['date'])
                prices.set_index('date', inplace=True)
                prices = prices.sort_index()
                prices = prices[start_date:end_date]
                result_dfs.append(prices)
    finally:
        client.close()

    if not result_dfs:
        logger.error("No historical data returned from MongoDB for requested tickers")
        return pd.DataFrame()

    historical_prices = pd.concat(result_dfs, axis=1)#.ffill().fillna(0)
    if 'CASH' in tickers:
        historical_prices['CASH'] = [1.0] * historical_prices.shape[0]

    missing = [t for t in tickers if t != 'CASH' and t not in historical_prices.columns]
    if missing:
        logger.warning(f"Tickers with no data in MongoDB: {missing}")
    return historical_prices


def fetch_portfolio_data(portfolio_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to fetch all portfolio data, over the fixed demonstration
    window [DEMO_START_DATE, DEMO_END_DATE] rather than "up to today".

    Args:
        portfolio_path: Path to the portfolio CSV file.

    Returns:
        Tuple of (portfolio_df, historical_data_df)
        - portfolio_df: Enhanced portfolio data with current prices and market values.
        - historical_data_df: Historical close prices for all tickers + benchmarks.
    """
    logger.info(f"Loading portfolio from {portfolio_path}")

    try:
        portfolio_df = pd.read_csv(portfolio_path)
    except Exception as e:
        logger.error(f"Failed to read portfolio file: {str(e)}")
        raise

    required_columns = ['Ticker', 'Quantity', 'Category', 'Cost_Basis_per_Share']
    missing_columns = set(required_columns) - set(portfolio_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    portfolio_df['Ticker'] = portfolio_df['Ticker'].astype(str).str.strip().str.upper()
    portfolio_tickers = portfolio_df['Ticker'].unique().tolist()

    logger.info(f"Fetching data for {len(portfolio_tickers)} portfolio tickers + benchmarks")
    logger.info(f"Using fixed demonstration window: {DEMO_START_DATE.date()} to {DEMO_END_DATE.date()}")

    historical_data_df = get_historical_prices(portfolio_tickers, DEMO_START_DATE, end_date=DEMO_END_DATE)

    if historical_data_df.empty:
        raise ValueError(
            "No historical price data could be retrieved from MongoDB for this portfolio. "
            "Check MONGO_URI and that the tickers exist in the 'historical_data' collection."
        )

    # Current price = last available close for each ticker.
    current_prices = {}
    for ticker in portfolio_tickers:
        if ticker == 'CASH':
            current_prices[ticker] = 1.0
            continue
        if ticker in historical_data_df.columns:
            series = historical_data_df[ticker]
            series = series[series != 0].dropna()
            if not series.empty:
                current_prices[ticker] = float(series.iloc[-1])

    portfolio_df['Current_Price'] = portfolio_df['Ticker'].map(current_prices)

    missing_prices = portfolio_df[portfolio_df['Current_Price'].isna()]['Ticker'].unique()
    if len(missing_prices) > 0:
        logger.warning(f"Could not find current prices for: {missing_prices}")

    # Calculate market value (works for both long and short positions).
    portfolio_df['Market_Value'] = portfolio_df['Quantity'] * portfolio_df['Current_Price']

    # Add position type for clarity.
    portfolio_df['Position_Type'] = portfolio_df['Quantity'].apply(lambda x: 'LONG' if x > 0 else 'SHORT')

    # Calculate unrealized P&L.
    portfolio_df['Unrealized_PnL'] = (
        portfolio_df['Current_Price'] - portfolio_df['Cost_Basis_per_Share']
    ) * portfolio_df['Quantity']

    portfolio_df['Unrealized_PnL_Pct'] = (
        (portfolio_df['Current_Price'] - portfolio_df['Cost_Basis_per_Share']) /
        portfolio_df['Cost_Basis_per_Share'] * 100
    )

    # Adjust P&L percentage for short positions.
    short_mask = portfolio_df['Position_Type'] == 'SHORT'
    portfolio_df.loc[short_mask, 'Unrealized_PnL_Pct'] *= -1

    logger.info(f"Portfolio data fetched successfully. Total positions: {len(portfolio_df)}")
    logger.info(f"Historical data shape: {historical_data_df.shape}")

    return portfolio_df, historical_data_df


# Example usage and testing
if __name__ == "__main__":
    sample_portfolio = pd.DataFrame({
        'Ticker': ['ATW', 'IAM', 'BCP'],
        'Quantity': [100, 50, 75],
        'Category': ['Banks', 'Telecommunications', 'Banks'],
        'Cost_Basis_per_Share': [450.0, 95.0, 260.0]
    })

    sample_portfolio.to_csv('sample_portfolio.csv', index=False)

    try:
        portfolio_df, historical_df = fetch_portfolio_data('sample_portfolio.csv')

        logger.info("\n=== Portfolio Summary ===")
        logger.info(portfolio_df)

        logger.info("\n=== Historical Data Summary ===")
        logger.info(f"Shape: {historical_df.shape}")
        logger.info(f"Date range: {historical_df.index.min()} to {historical_df.index.max()}")
        logger.info(f"Tickers: {list(historical_df.columns)}")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
