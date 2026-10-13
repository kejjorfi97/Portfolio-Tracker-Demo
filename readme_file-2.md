# 🇲🇦 CSE Portfolio Analytics System

Professional portfolio analysis and risk management system for the **Casablanca Stock Exchange**, inspired by institutional quantitative methods.

## Features

- **Market Regime Detection**: Identifies Bull Complacent, Crisis/Bear, or Neutral market conditions using the MSI20 benchmark's trend and its own trailing realized volatility (no Moroccan VIX-equivalent exists)
- **Advanced Risk Metrics**: Sharpe, Sortino, and Calmar ratios with institutional-grade calculations
- **Sector Concentration Risk Analysis**: Flags when a single CSE sector dominates the portfolio
- **Tax Loss Harvesting**: Flags unrealized losses and estimates the tax value of realizing them, at a configurable flat rate
- **Conviction Scoring**: Three-factor model to identify underperforming holdings
- **Interactive Visualizations**: Professional Plotly charts
- **Giuseppe Memo**: Executive-style investment memo with prioritized actions
- **Backtesting Framework**: Historical performance analysis vs the MSI20 benchmark

## Installation

1. Clone the repository or download all files to a folder:
```bash
mkdir cse_portfolio
cd cse_portfolio
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set the MongoDB connection string (falls back to `mongodb://localhost:27017/` if unset):
```bash
export MONGO_URI='your-mongodb-connection-string'
```

## Usage

### Quick Start

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your browser to `http://localhost:8501`

3. Either:
   - Upload your portfolio CSV file
   - Use the sample portfolio for demonstration

4. Click "🚀 Run Analysis" to generate comprehensive insights

### Running the Test Suite

To see the full system in action with sample data:

```bash
python test_script.py
```

This will:
- Create a test portfolio of CSE-listed banks, telecom, real estate, insurance, mining, and other sectors
- Run comprehensive analysis including sector concentration warnings
- Generate a Giuseppe memo
- Save results to multiple files
- Display backtest results

### Portfolio CSV Format

Your portfolio CSV must have these columns:
- `Ticker`: CSE ticker symbol (e.g., ATW, IAM), or `CASH` for a cash position
- `Quantity`: Number of shares (negative for short positions)
- `Category`: Sector (e.g., Banks, Telecommunications, Real Estate)
- `Cost_Basis_per_Share`: Purchase price per share, in MAD

Example:
```csv
Ticker,Quantity,Category,Cost_Basis_per_Share
ATW,100,Banks,450.00
ARD,-50,Real Estate,450.00
CASH,10000,Cash,1.00
```

## Key Components

### 1. Data Fetcher (`data_fetcher.py`)
- Pulls historical close prices from MongoDB (`ptf_tracker.historical_data`) via `MONGO_URI`
- Automatic benchmark inclusion (MSI20, MASI)
- Robust error handling with fallbacks (last known price, logging of missing tickers)

### 2. Analysis Engine (`analysis_engine.py`)
- Institutional-grade calculations
- Market regime detection (MSI20 trend + realized volatility)
- Sector concentration risk analysis
- Tax loss harvesting with configurable Moroccan tax rates
- Basic backtesting capabilities

### 3. Web Dashboard (`streamlit_app.py`)
- Interactive Streamlit interface
- Real-time portfolio analysis
- Professional visualizations
- Export capabilities

### 4. Test Suite (`test_script.py`)
- Comprehensive testing framework
- Sample CSE portfolio generation
- Sector concentration warning demonstration
- Automated report generation

### 5. Advanced Backtesting (`backtest.py`)
- Multiple rebalancing strategies
- Monte Carlo simulations
- Regime-aware backtesting (benchmark trend + realized volatility)
- Transaction cost modeling

## Sector Concentration Risk Detection

The system flags concentration risk in the portfolio's single largest CSE sector:

- **Concentration Scoring**: The percentage of the portfolio in its largest sector
- **Concentration Warnings**: Alerts when that percentage exceeds safe thresholds (20% / 30% / 40%)
- **Correlation Enhancement**: Identifies same-sector correlations in your portfolio
- **Recommendations**: Specific guidance for diversifying away from the dominant sector

## Metrics Explained

- **Sharpe Ratio**: Risk-adjusted return (>1.0 good, >2.0 excellent)
- **Sortino Ratio**: Downside risk-adjusted return (higher is better)
- **Calmar Ratio**: Annual return / Max drawdown (>1.0 is strong)
- **Conviction Score**: 0-1 scale combining performance, risk, and allocation
- **Sector Concentration**: Percentage of the portfolio exposed to its single largest sector

## Sector Allocation

The system reports current portfolio exposure by Casablanca market sector (Banks, Telecommunications, Real Estate, Insurance, Building Materials, Mining, Food & Beverage, Distribution, Utilities, etc.). This is purely descriptive — there's no liquid Moroccan equivalent to the US ETFs (commodity-trend, long-volatility, gold) that a fixed-target allocation model like Chris Cole's "Dragon Portfolio" relies on, so no target percentages or rebalancing alerts are generated.

## Professional Features

1. **Tax Loss Harvesting**
   - Identifies losing positions
   - Estimates the tax value of realizing each loss at a configurable flat rate (default 15% capital gains)

2. **Correlation Analysis**
   - Identifies hidden concentration risks
   - Alerts for correlations > 0.75
   - Flags same-sector concentration
   - Suggests diversification opportunities

3. **Conviction Scoring**
   - Three-factor model: Performance, Risk, Allocation
   - Identifies underperforming holdings
   - Data-driven position sizing

4. **Market Regime Detection**
   - Uses MSI20 trend and its own trailing realized volatility
   - Adapts recommendations to market conditions
   - Behavioral coaching for each regime

## AI-Powered Recommendations

The system integrates with AI APIs to provide personalized, current market recommendations:

### Setup
1. **OpenAI (Recommended)**:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```

2. **Anthropic (Alternative)**:
   ```bash
   export ANTHROPIC_API_KEY='your-anthropic-api-key'
   ```

Without an API key, the system uses enhanced programmatic recommendations based on quantitative analysis.

### Features
- **Personalized Advice**: AI analyzes your specific portfolio composition
- **Current Market Context**: Recommendations adapt to recent market conditions
- **Specific Actions**: Includes ticker symbols, amounts (in MAD), and rationale
- **Fallback System**: Works without API with detailed rule-based recommendations

## Troubleshooting

### Common Issues

1. **"No historical data returned from MongoDB"**
   - Check `MONGO_URI` is set and reachable
   - Confirm the ticker exists as a document in `ptf_tracker.historical_data`
   - Check ticker symbol spelling/casing (tickers are upper-cased before lookup)

2. **Slow data fetching**
   - This tracker is demonstration-only and always fetches the same fixed window (`DEMO_START_DATE`/`DEMO_END_DATE` in `data_fetcher.py`, currently 2025-01-02 to 2026-09-15) - narrow that range if it's slow

3. **Import errors**
   - Ensure all packages installed: `pip install -r requirements.txt`
   - Use Python 3.8 or higher

## Advanced Usage

### Running Analysis Programmatically

```python
from analysis_engine import run_full_analysis

# Run analysis
results = run_full_analysis('portfolio.csv')

# Access results
print(f"Portfolio Value: {results['portfolio_value']:,.0f} MAD")
print(f"Sharpe Ratio: {results['portfolio_metrics']['sharpe_ratio']}")
print(results['giuseppe_memo'])
```

### Customizing Parameters

Construct `PortfolioAnalyzer(...)` in `analysis_engine.py` to modify:
- Risk-free rate (default: ~2.75%)
- Capital gains / dividend tax rates (default: 15% / 15%)
- Benchmark ticker (default: MSI20)
- Realized-volatility regime thresholds (default: 15% / 25%)

Edit `DEMO_START_DATE` / `DEMO_END_DATE` in `data_fetcher.py` to change the fixed demonstration window every analysis run uses.

Edit `sector_classification.py` to extend the CSE ticker → sector map used as a fallback when a portfolio CSV doesn't supply a `Category`.

## Export Options

The dashboard provides multiple export formats:
- Enhanced portfolio CSV with P&L
- Correlation matrix
- Performance metrics summary
- Executive memo (text format)

## Contributing

Feel free to extend the system with:
- Additional risk metrics
- A broader CSE sector/ticker map
- Alternative data sources
- Machine learning predictions

## License

This project is provided as-is for educational and analytical purposes.

## Changelog

### Version 2.0.0 - Casablanca Stock Exchange Adaptation
- **MongoDB Data Source**: Replaced Yahoo Finance with a MongoDB-backed historical price feed (`ptf_tracker.historical_data`)
- **MSI20 Benchmark**: Replaced SPY with the MSI20 index across regime detection, performance comparison, and conviction scoring
- **Realized-Volatility Regime Detection**: Replaced VIX-based regime detection with the benchmark's own trailing realized volatility
- **Sector Allocation Model**: Replaced the US-ETF-based "Dragon Portfolio" target allocation with a descriptive Casablanca sector breakdown
- **Sector Concentration Risk**: Repurposed AI-sector-specific scoring into a generic single-sector concentration check
- **Moroccan Tax Handling**: Tax-loss harvesting now uses configurable flat Moroccan capital-gains/dividend rates, with no US-ETF replacement suggestions
- **Removed Options Strategies**: Dropped listed-options recommendations (no standardized CSE equity options market)
- **MAD Currency**: All figures now denominated in Moroccan dirhams

### Version 1.0.0 - Initial Release (US Market / Dragon Portfolio)
- Core analysis engine with Sharpe, Sortino, Calmar ratios
- Chris Cole "Dragon Portfolio" target allocation (24/18/21/19/18)
- AI sector risk detection
- Tax loss harvesting with wash-sale-aware ETF replacements
- Options strategy recommendations
- Basic backtesting
- Streamlit dashboard

---

Built with institutional-grade quantitative methods, adapted for the Casablanca Stock Exchange.
