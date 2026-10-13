# -*- coding: utf-8 -*-
"""
Sanity check: verifies PortfolioAnalyzer.calculate_portfolio_metrics against
a hand-computed ground truth for an MSI20-only "portfolio", using historical
prices pulled from MongoDB instead of Yahoo Finance.

@author: LENOVO
"""

import pandas as pd

from analysis_engine import PortfolioAnalyzer
from data_fetcher import get_historical_prices, DEMO_START_DATE, DEMO_END_DATE

print("🧪 Running MSI20-only backtest consistency check...")

# 1. Define MSI20-only test portfolio
BENCHMARK = "MSI20"
msi20_portfolio = pd.DataFrame({
    'Ticker': [BENCHMARK],
    'Market_Value': [10000]
})

# 2. Fetch MSI20 historical data from MongoDB, over the same fixed
# demonstration window the rest of the app uses.
historical = get_historical_prices([BENCHMARK], DEMO_START_DATE, benchmarks=[], end_date=DEMO_END_DATE)
msi20_data = historical[[BENCHMARK]].replace(0, pd.NA).dropna()

# 3. Instantiate the analysis engine (same default risk-free rate it uses internally)
engine = PortfolioAnalyzer()

# 4. Compute metrics using the engine
metrics = engine.calculate_portfolio_metrics(msi20_portfolio, msi20_data)

# 5. Ground truth using direct calculation
msi20_returns = msi20_data[BENCHMARK].pct_change().dropna()
msi20_nav = (1 + msi20_returns).cumprod()

msi20_cumulative_return = (1 + msi20_returns).prod() - 1
msi20_annual_return = (1 + msi20_cumulative_return) ** (252 / len(msi20_returns)) - 1

msi20_total_return = msi20_nav.iloc[-1] - 1
msi20_volatility = msi20_returns.std() * (252 ** 0.5)

risk_free_rate = engine.risk_free_rate
daily_rf = risk_free_rate / 252
excess_returns = msi20_returns - daily_rf
msi20_sharpe = (excess_returns.mean() / msi20_returns.std()) * (252 ** 0.5)

# 6. Print comparison
print("\n📊 Comparison with Ground Truth:")
print(f"  MSI20 Actual Total Return:  {msi20_total_return:.2%}")
print(f"  Engine Cumulative Return: {metrics['annual_return']:.2%}")

print(f"  MSI20 Volatility:           {msi20_volatility:.2%}")
print(f"  Engine Volatility:        {metrics['annual_volatility']:.2%}")

print(f"  MSI20 Sharpe:               {msi20_sharpe:.2f}")
print(f"  Engine Sharpe:            {metrics['sharpe_ratio']:.2f}")

# 7. Optional: assert close values
assert abs(metrics['annual_return'] - msi20_annual_return * 100) < 1, "Return mismatch"
assert abs(metrics['annual_volatility'] - msi20_volatility * 100) < 0.5, "Volatility mismatch"
assert abs(metrics['sharpe_ratio'] - msi20_sharpe) < 0.2, "Sharpe mismatch"

print("\n✅ MSI20 backtest is consistent, engine logic is sound.\n")
