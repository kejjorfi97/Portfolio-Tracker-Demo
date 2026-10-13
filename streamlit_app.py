"""
Casablanca Stock Exchange Portfolio Analytics Dashboard
========================================================
Professional portfolio analysis and risk management system for Moroccan
equities. Built with Streamlit for real-time interactive analysis.

Author: Dragon Portfolio Analytics
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import io
from typing import Dict, Any
import base64
import plotly.express as px
from io import BytesIO
# Import our modules
from analysis_engine import run_full_analysis
from sector_classification import classify_sector
from dotenv import load_dotenv

load_dotenv()


 # Page configuration
st.set_page_config(
    page_title="CSE Portfolio Analytics",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        background-color: #f0f2f6;
        border: 1px solid #e0e2e6;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .giuseppe-memo {
        background-color: #f8f9fa;
        border: 2px solid #343a40;
        border-radius: 0.5rem;
        padding: 2rem;
        margin-top: 2rem;
        font-family: 'Georgia', serif;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    h1 {
        color: #1976d2;
        font-family: 'Arial', sans-serif;
        font-weight: 600;
    }
    h2 {
        color: #424242;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format value as MAD currency."""
    return f"{value:,.2f} MAD"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.2f}%"


def get_risk_class(value: float, thresholds: tuple) -> str:
    """Get CSS class based on risk level."""
    if value <= thresholds[0]:
        return "risk-low"
    elif value <= thresholds[1]:
        return "risk-medium"
    else:
        return "risk-high"


def interpret_risk_metrics(metrics: dict) -> list:
    """Short, rule-of-thumb interpretation for each risk-adjusted performance metric."""
    notes = []

    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe < 0:
        notes.append(f"**Sharpe Ratio ({sharpe:.2f})** — negative: returns didn't even clear the risk-free rate on a risk-adjusted basis.")
    elif sharpe < 1:
        notes.append(f"**Sharpe Ratio ({sharpe:.2f})** — below 1: sub-par risk-adjusted return.")
    elif sharpe < 2:
        notes.append(f"**Sharpe Ratio ({sharpe:.2f})** — between 1 and 2: a good risk-adjusted return.")
    else:
        notes.append(f"**Sharpe Ratio ({sharpe:.2f})** — above 2: an excellent risk-adjusted return.")

    sortino = metrics.get('sortino_ratio', 0)
    if sortino < 0:
        notes.append(f"**Sortino Ratio ({sortino:.2f})** — negative: poor return relative to downside risk alone.")
    elif sortino < 1:
        notes.append(f"**Sortino Ratio ({sortino:.2f})** — below 1: modest reward for the downside risk taken.")
    elif sortino < 2:
        notes.append(f"**Sortino Ratio ({sortino:.2f})** — between 1 and 2: solid downside-risk-adjusted performance.")
    else:
        notes.append(f"**Sortino Ratio ({sortino:.2f})** — above 2: excellent downside-risk-adjusted performance.")

    calmar = metrics.get('calmar_ratio', 0)
    if calmar < 0:
        notes.append(f"**Calmar Ratio ({calmar:.2f})** — negative: annual return doesn't compensate for the drawdown experienced.")
    elif calmar < 1:
        notes.append(f"**Calmar Ratio ({calmar:.2f})** — below 1: weak return relative to max drawdown.")
    elif calmar < 3:
        notes.append(f"**Calmar Ratio ({calmar:.2f})** — between 1 and 3: a healthy return-to-drawdown balance.")
    else:
        notes.append(f"**Calmar Ratio ({calmar:.2f})** — above 3: an exceptionally strong return relative to drawdown.")

    omega = metrics.get('omega_ratio', 0)
    if omega < 1:
        notes.append(f"**Omega Ratio ({omega:.2f})** — below 1: losses have outweighed gains in aggregate.")
    elif omega < 1.5:
        notes.append(f"**Omega Ratio ({omega:.2f})** — between 1 and 1.5: a modest edge of gains over losses.")
    elif omega < 2:
        notes.append(f"**Omega Ratio ({omega:.2f})** — between 1.5 and 2: a good balance of gains over losses.")
    else:
        notes.append(f"**Omega Ratio ({omega:.2f})** — above 2: gains have strongly outweighed losses.")

    tail = metrics.get('tail_ratio', 0)
    if tail < 1:
        notes.append(f"**Tail Ratio ({tail:.2f})** — below 1: worst-case losses have been larger than best-case gains.")
    elif tail < 1.5:
        notes.append(f"**Tail Ratio ({tail:.2f})** — around 1–1.5: gains and losses are roughly balanced at the extremes.")
    else:
        notes.append(f"**Tail Ratio ({tail:.2f})** — above 1.5: best-case gains have meaningfully outsized worst-case losses.")

    return notes


def create_sample_portfolio() -> pd.DataFrame:
    """Create a sample Casablanca Stock Exchange portfolio for demonstration."""
    return pd.DataFrame({
        'Ticker': ['ATW', 'IAM', 'BCP', 'LHM', 'CIH', 'WAA', 'ADH', 'MNG', 'CSR', 'LBV'],
        'Quantity': [100, 200, 150, 50, 120, 80, 100, 60, 90, 40],
        'Category': ['Banks', 'Telecommunications', 'Banks', 'Building Materials', 'Banks',
                    'Insurance', 'Real Estate', 'Mining', 'Food & Beverage', 'Distribution'],
        'Cost_Basis_per_Share': [450.0, 95.0, 260.0, 2000.0, 280.0, 4200.0, 30.0, 2200.0, 260.0, 2100.0]
    })


def render_giuseppe_memo(memo: str) -> None:
    """Render the Giuseppe memo with proper formatting."""
    memo_html = memo.replace('**', '<strong>').replace('**', '</strong>')
    memo_html = memo_html.replace('\n', '<br>')

    st.markdown(
        f'<div class="giuseppe-memo">{memo_html}</div>',
        unsafe_allow_html=True
    )

def validate_trade_log(trade_log: pd.DataFrame, portfolio: pd.DataFrame) -> dict:
    """
    Validate a trade log DataFrame against a portfolio and optional price data.

    Args:
        trade_log: DataFrame with columns ['Date', 'Ticker', 'Action', 'Quantity', 'Price'].
            Action is one of Buy, Sell (share trades) or Deposit, Withdrawal,
            Dividend (cash-only events - Quantity * Price is the cash amount,
            Ticker is ignored for Deposit/Withdrawal and optional for Dividend).
        portfolio: DataFrame with current holdings ['Ticker', 'Quantity']

    Returns:
        dict with 'errors' and 'warnings'
    """
    errors = []
    warnings = []

    required_cols = {'Date', 'Ticker', 'Action', 'Quantity', 'Price'}
    if not required_cols.issubset(trade_log.columns):
        errors.append(f"Missing required columns: {required_cols - set(trade_log.columns)}")
        return {'errors': errors, 'warnings': warnings}

    valid_actions = {'Buy', 'Sell', 'Deposit', 'Withdrawal', 'Dividend'}
    invalid_actions = set(trade_log['Action'].unique()) - valid_actions
    if invalid_actions:
        errors.append(f"Invalid action types: {invalid_actions}")

    # Check for negative or zero quantities/prices
    if (trade_log['Quantity'] <= 0).any():
        errors.append("Found trades with non-positive quantities.")
    if (trade_log['Price'] <= 0).any():
        errors.append("Found trades with non-positive prices.")

    # Oversell detection (share trades only - cash events have no holdings to oversell)
    holdings = portfolio.set_index('Ticker')['Quantity'].to_dict()
    share_trades = trade_log[trade_log['Action'].isin(['Buy', 'Sell'])]
    trade_log_sorted = share_trades.sort_values('Date')
    temp_holdings = holdings.copy()

    for _, row in trade_log_sorted.iterrows():
        ticker = row['Ticker']
        qty = row['Quantity']
        action = row['Action']

        if ticker not in temp_holdings:
            temp_holdings[ticker] = 0

        if action == 'Sell':
            if qty > temp_holdings[ticker]:
                errors.append(f"Oversell detected: Tried to sell {qty} of {ticker}, only {temp_holdings[ticker]} available on {row['Date']}.")
            else:
                temp_holdings[ticker] -= qty
        elif action == 'Buy':
            temp_holdings[ticker] += qty

    # Ticker consistency (share trades only - Deposit/Withdrawal/Dividend rows
    # legitimately reference CASH or no specific holding)
    tickers_in_trade_log = set(share_trades['Ticker'])
    tickers_in_portfolio = set(portfolio['Ticker'])
    unmatched_tickers = tickers_in_trade_log - tickers_in_portfolio
    if unmatched_tickers:
        warnings.append(f"Tickers in trade log but not in portfolio: {unmatched_tickers}")

    return {'errors': errors, 'warnings': warnings}


def calculate_realized_pnl(trade_log: pd.DataFrame) -> float:
    """
    Total realized P&L from round-trip (Buy + Sell) trades in the trade log.
    Matches the first Buy against the first Sell per ticker, mirroring the
    Trade Log tab's own matching logic - not full FIFO lot accounting.
    Returns 0.0 if no trade log or no matched round trips.
    """
    if trade_log is None or trade_log.empty:
        return 0.0

    log = trade_log.copy()
    log['Action'] = log['Action'].astype(str).str.lower().str.strip()
    log['Ticker'] = log['Ticker'].astype(str).str.upper()
    log['Quantity'] = pd.to_numeric(log['Quantity'], errors='coerce')
    log['Price'] = pd.to_numeric(log['Price'], errors='coerce')

    buy_prices = log[log['Action'] == 'buy'].set_index('Ticker')
    sell_prices = log[log['Action'] == 'sell'].set_index('Ticker')
    common_tickers = buy_prices.index.intersection(sell_prices.index)

    total_pnl = 0.0
    for ticker in common_tickers:
        buy = buy_prices.loc[ticker]
        sell = sell_prices.loc[ticker]
        if isinstance(buy, pd.DataFrame):
            buy = buy.iloc[0]
        if isinstance(sell, pd.DataFrame):
            sell = sell.iloc[0]
        total_pnl += (sell['Quantity'] * sell['Price']) - (buy['Quantity'] * buy['Price'])

    return total_pnl


def reconcile_cash_flows(trade_log: pd.DataFrame) -> dict:
    """
    Break down every cash-moving event in the trade log: Buy/Sell debit/credit
    cash at Quantity * Price, and Deposit/Withdrawal/Dividend rows do the same
    (Ticker is irrelevant for those - the cash amount is just Quantity * Price).

    Returns a dict with each component plus 'net_cash_flow', the total change
    in cash implied by the log (deposits + dividends + sell proceeds minus
    withdrawals and buy costs).
    """
    empty = {'deposits': 0.0, 'withdrawals': 0.0, 'dividends': 0.0,
             'buy_cost': 0.0, 'sell_proceeds': 0.0, 'net_cash_flow': 0.0}
    if trade_log is None or trade_log.empty:
        return empty

    log = trade_log.copy()
    log['Action'] = log['Action'].astype(str).str.strip().str.lower()
    log['Amount'] = pd.to_numeric(log['Quantity'], errors='coerce') * pd.to_numeric(log['Price'], errors='coerce')

    def total_for(action: str) -> float:
        return float(log.loc[log['Action'] == action, 'Amount'].sum())

    deposits = total_for('deposit')
    withdrawals = total_for('withdrawal')
    dividends = total_for('dividend')
    buy_cost = total_for('buy')
    sell_proceeds = total_for('sell')

    return {
        'deposits': deposits,
        'withdrawals': withdrawals,
        'dividends': dividends,
        'buy_cost': buy_cost,
        'sell_proceeds': sell_proceeds,
        'net_cash_flow': deposits + dividends + sell_proceeds - withdrawals - buy_cost,
    }


def get_classified_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in blank `Category` values using the Casablanca Stock Exchange sector
    map (sector_classification.py). Unlike the old Yahoo-Finance-based lookup,
    this is a static, offline mapping — extend CSE_SECTOR_MAP for tickers that
    come back 'Unclassified'.
    """
    df = df.copy()
    for col in ["Ticker", "Category"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("").str.strip()
    df["Ticker"] = df["Ticker"].str.upper()

    df["Category"] = df.apply(lambda r: classify_sector(r["Ticker"], r["Category"]), axis=1)
    return df


def main():
    # Header
    st.title("🇲🇦 CSE Portfolio Analytics System")
    st.markdown("**Casablanca Stock Exchange Portfolio Analysis & Risk Management**")

    # Sidebar
    with st.sidebar:
        st.header("Portfolio Configuration")

        # File upload or sample portfolio
        upload_option = st.radio(
            "Choose data source:",
            ["Upload Portfolio CSV", "Use Sample Portfolio"]
        )

        portfolio_df = None

        if upload_option == "Upload Portfolio CSV":
            uploaded_file = st.file_uploader(
                "Upload your portfolio CSV",
                type=['csv'],
                help="CSV should contain: Ticker, Quantity, Category, Cost_Basis_per_Share"
            )

            if uploaded_file is not None:
                try:
                    portfolio_df = pd.read_csv(uploaded_file)
                    portfolio_df = get_classified_portfolio(portfolio_df)
                    st.success("✅ Portfolio uploaded successfully!")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        else:
            portfolio_df = create_sample_portfolio()
            st.info("Using sample portfolio for demonstration")

        with st.expander("📥 Upload Trade Log (optional)", expanded=False):
            trade_file = st.file_uploader("Upload CSV containing your trade history", type=["csv"], key="trade_log_upload")

            if trade_file is not None:
                trade_log = pd.read_csv(trade_file, parse_dates=["Date"])

                st.success("Trade log uploaded successfully.")
                # st.dataframe(trade_log.head())
            else:
                trade_log = None

        # Analysis options
        st.header("Analysis Options")

        show_all_metrics = st.checkbox("Show all performance metrics", value=True)
        show_tax_analysis = st.checkbox("Show tax loss harvesting", value=True)
        show_correlations = st.checkbox("Show correlation analysis", value=True)

        # Run analysis button
        analyze_button = st.button(
            "🚀 Run Analysis",
            use_container_width=True,
            disabled=(portfolio_df is None)
        )

    # Main content area
    if portfolio_df is not None:
        if analyze_button:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Save portfolio to temp file
            with st.spinner("Analyzing portfolio..."):
                # Save to temporary CSV
                temp_file = "temp_portfolio.csv"

                # Reconcile cash: starting CASH balance (if any) plus the net
                # cash flow implied by the trade log (deposits, withdrawals,
                # dividends, and buy/sell proceeds). This is what makes
                # Portfolio Value, weights, and P&L downstream reflect actual
                # cash movements instead of a static CASH row.
                portfolio_to_analyze = portfolio_df.copy()
                cash_flows = reconcile_cash_flows(trade_log)
                if cash_flows['net_cash_flow'] != 0:
                    cash_mask = portfolio_to_analyze['Ticker'].astype(str).str.upper() == 'CASH'
                    if cash_mask.any():
                        portfolio_to_analyze.loc[cash_mask, 'Quantity'] += cash_flows['net_cash_flow']
                    else:
                        new_cash_row = pd.DataFrame([{
                            'Ticker': 'CASH',
                            'Quantity': cash_flows['net_cash_flow'],
                            'Category': 'Cash',
                            'Cost_Basis_per_Share': 1.0,
                        }])
                        portfolio_to_analyze = pd.concat([portfolio_to_analyze, new_cash_row], ignore_index=True)

                portfolio_to_analyze.to_csv(temp_file, index=False)

                # Update progress
                progress_bar.progress(20)
                status_text.text("Fetching market data...")

                # Run analysis
                try:
                    results = run_full_analysis(temp_file)
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                    # Store results in session state
                    st.session_state['analysis_results'] = results

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.stop()

            total_cost = (portfolio_df["Quantity"] * portfolio_df["Cost_Basis_per_Share"]).sum()
            perfTab, riskTab, regimeTab, aimemoTab, exportTab, logsTab = st.tabs([
                "📈 Performance",
                "⚠️ Risk Metrics",
                "📉 Market Regime",
                "🧠 AI Insights",
                "💾 Export Reports",
                "📒 Trade Log"
            ])


            with perfTab:
                if 'analysis_results' in st.session_state:
                    results = st.session_state['analysis_results']
                    metrics = results.get('portfolio_metrics', {})
                    graphs = results.get('graphs', {})
                    overall_return = results.get('portfolio_value', 0) / total_cost - 1
                    pnl_df = results.get("portfolio_df", pd.DataFrame())

                    st.markdown("#### Portfolio Performance")

                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                        st.metric(
                            "Initial Investment",
                            format_currency(total_cost)
                        )

                    with col2:
                        portfolio_value = results.get('portfolio_value', 0)
                        st.metric(
                            "Portfolio Value",
                            format_currency(portfolio_value),
                            delta=f"{(overall_return * 100):.2f}%",
                        )

                    with col3:
                        total_unrealized_pnl = pnl_df['Unrealized_PnL'].sum() if not pnl_df.empty else 0.0
                        unrealized_pct = (total_unrealized_pnl / total_cost * 100) if total_cost else 0.0
                        st.metric(
                            "Unrealized P&L",
                            format_currency(total_unrealized_pnl),
                            delta=f"{unrealized_pct:.2f}%",
                        )

                    with col4:
                        total_realized_pnl = calculate_realized_pnl(trade_log)
                        st.metric(
                            "Realized P&L",
                            format_currency(total_realized_pnl) if trade_log is not None else "N/A"
                        )

                    with col5:
                            st.metric(
                                "Annual Return",
                                format_percentage(metrics.get('annual_return', 0))
                            )

                    with col6:
                        st.metric(
                            "Annual Volatility",
                            format_percentage(metrics.get('annual_volatility', 0))
                        )
                    if 'performance_chart' in graphs:
                            st.plotly_chart(graphs['performance_chart'], use_container_width=True)

                    st.markdown("#### P&L per Ticker")
                    if not pnl_df.empty:
                        pnl_df = pnl_df.copy()
                        pnl_df['Investment_Amount'] = pnl_df['Quantity'] * pnl_df['Cost_Basis_per_Share']
                        total_abs_market_value = pnl_df['Market_Value'].abs().sum()
                        pnl_df['Weight_Pct'] = (
                            pnl_df['Market_Value'].abs() / total_abs_market_value * 100
                            if total_abs_market_value > 0 else 0.0
                        )
                        pnl_df = pnl_df[['Ticker', 'Quantity', 'Cost_Basis_per_Share', 'Investment_Amount',
                                          'Current_Price', 'Market_Value', 'Weight_Pct',
                                          'Unrealized_PnL', 'Unrealized_PnL_Pct']]
                        pnl_df = pnl_df.rename(columns={
                            'Cost_Basis_per_Share': 'Cost Basis',
                            'Investment_Amount': 'Investment',
                            'Current_Price': 'Price',
                            'Market_Value': 'Market Value',
                            'Weight_Pct': 'Weight %',
                            'Unrealized_PnL': 'P&L',
                            'Unrealized_PnL_Pct': 'P&L %',
                        })
                        pnl_df.set_index('Ticker', inplace=True)
                        num_rows = len(pnl_df)
                        st.dataframe(
                            pnl_df.style
                            .format({
                                'Cost Basis': '{:,.2f}',
                                'Investment': '{:,.0f}',
                                'Price': '{:,.2f}',
                                'Market Value': '{:,.0f}',
                                'Weight %': '{:.1f}%',
                                'P&L': '{:,.0f}',
                                'P&L %': '{:.1f}%'})
                            .applymap(lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red', subset=['P&L', 'P&L %']),
                            use_container_width=True,
                            height=min(num_rows, 12) * 35 + 38
                        )
                        st.caption("Amounts in MAD.")

                    # Allocation pie charts
                    df = portfolio_df.copy()

                    # Signed total cost (longs > 0, shorts < 0)
                    df["Total Cost"] = df["Quantity"] * df["Cost_Basis_per_Share"]

                    # Minimal: show allocation by absolute exposure so shorts appear in the pies
                    df["Abs_Exposure"] = df["Total Cost"].abs()

                    # Label shorts clearly without changing colors/structure
                    df["Ticker_Label"] = np.where(df["Total Cost"] < 0, df["Ticker"] + " (Short)", df["Ticker"])

                    # Weights based on absolute exposure
                    total_abs = df["Abs_Exposure"].sum()
                    df["Weight"] = df["Abs_Exposure"] / total_abs if total_abs > 0 else 0.0

                    st.markdown("#### Allocation by Ticker")
                    fig = px.pie(
                        df,
                        names="Ticker_Label",
                        values="Weight",
                        color="Category",  # keeps your original color logic
                        title="📊 Portfolio Allocation (by Absolute Exposure)",
                        hole=0.4
                    )
                    fig.update_traces(textinfo="percent+label", hovertemplate="<b>%{label}</b><br>Weight: %{percent:.1%}<extra></extra>")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### Allocation by Sector")
                    category_df = (
                        df.groupby("Category", as_index=False)["Abs_Exposure"].sum()
                        .rename(columns={"Abs_Exposure": "Total_Abs_Exposure"})
                    )
                    total_cat_abs = category_df["Total_Abs_Exposure"].sum()
                    category_df["Weight"] = category_df["Total_Abs_Exposure"] / total_cat_abs if total_cat_abs > 0 else 0.0

                    fig2 = px.pie(
                        category_df,
                        names="Category",
                        values="Weight",
                        title="Portfolio Allocation by Sector (Absolute Exposure)",
                        hole=0.4
                    )
                    fig2.update_traces(textinfo="percent+label", hovertemplate="<b>%{label}</b><br>Weight: %{percent:.1%}<extra></extra>")
                    st.plotly_chart(fig2, use_container_width=True)

            with riskTab:
                if 'analysis_results' in st.session_state:
                    results = st.session_state['analysis_results']
                    metrics = results.get('portfolio_metrics', {})
                    graphs = results.get('graphs', {})

                    # Additional metrics
                    if show_all_metrics:
                        st.subheader("Advanced Risk Metrics")
                        if 'performance_metrics' in graphs:
                            st.plotly_chart(graphs['performance_metrics'], use_container_width=True)
                            for note in interpret_risk_metrics(metrics):
                                st.markdown(f"- {note}")
                        if 'drawdown_chart' in graphs:
                            st.plotly_chart(graphs['drawdown_chart'], use_container_width=True)
                        if show_correlations and 'correlation_heatmap' in graphs:
                            st.subheader("Asset Correlation Analysis")
                            st.plotly_chart(graphs['correlation_heatmap'], use_container_width=True)

                            # High correlation alerts
                            correlations = results.get('correlation_alerts', [])
                            if correlations:
                                st.subheader("⚠️ High Correlation Alerts")
                                for alert in correlations[:5]:  # Show top 5
                                    pair = alert['pair']
                                    corr = alert['correlation']
                                    st.markdown(f"""
                                        <div style='border:1px solid #FFCCBC; background-color:#FBE9E7; padding:0.5rem; border-radius:6px; margin-bottom:0.5rem;'>
                                            <strong>{pair[0]}</strong> ↔️ <strong>{pair[1]}</strong>:
                                            <span style='color:#D84315; font-weight:bold;'>{corr:.3f}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div style='border:1px solid #EF9A9A; background-color:#FFEBEE; padding:0.5rem; border-radius:6px; margin-bottom:0.5rem;'>
                                        ❌ <strong>No high correlation results to display</strong>
                                    </div>
                                    """, unsafe_allow_html=True)

            with regimeTab:
                if 'analysis_results' in st.session_state:
                    regime = results.get('market_regime', 'Unknown')
                    regime_color = {
                        'Bull Complacent': '🟢',
                        'Crisis / Bear': '🔴',
                        'Neutral / Transition': '🟡'
                    }.get(regime, '⚪')
                    st.metric(
                        "Market Regime",
                        f"{regime_color} {regime}",
                        delta=None
                    )
                    if 'regime_timeline' in graphs:
                        st.plotly_chart(graphs['regime_timeline'], use_container_width=True)

            with aimemoTab:
                if 'analysis_results' in st.session_state:
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        if 'allocation_pie' in graphs:
                            st.plotly_chart(graphs['allocation_pie'], use_container_width=True)

                    with col2:
                        sector_allocation = results.get('sector_allocation', {})
                        st.subheader("Sector Allocation")

                        allocations = sector_allocation.get('allocations', {})

                        if allocations:
                            sectors = list(allocations.keys())
                            weights = [allocations[s] for s in sectors]

                            fig_allocation = go.Figure()
                            fig_allocation.add_trace(go.Bar(x=sectors, y=weights, name='Current', marker_color='#1976D2'))

                            fig_allocation.update_layout(
                                title='Current Allocation by Sector',
                                height=300,
                                margin=dict(t=40, b=20),
                                yaxis_title='% of Portfolio'
                            )

                            st.plotly_chart(fig_allocation, use_container_width=True)

                            largest_sector = max(allocations.items(), key=lambda x: x[1])
                            st.markdown(f"""
                            <div style='padding: 1rem; border-radius: 0.5rem; border: 1px solid #ddd;'>
                            <strong>Largest Sector:</strong> {largest_sector[0]}
                            <br>
                            <span style='font-size: 1.5rem;'>{largest_sector[1]:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

                        for sector, pct in allocations.items():
                            st.write(f"**{sector}**: {pct:.1f}%")

                    if 'conviction_bars' in graphs:
                        st.subheader("📊 Conviction Score Breakdown")
                        st.plotly_chart(graphs['conviction_bars'], use_container_width=True)

                        # Low conviction alerts
                        conviction = results.get('conviction_analysis', {})
                        low_conviction = conviction.get('low_conviction_alerts', [])

                        if low_conviction:
                            st.subheader("🚨 Low Conviction Holdings")

                            for alert in low_conviction:
                                ticker = alert['ticker']
                                score = alert['score']
                                reason = alert['reason']

                                st.markdown(f"""
                                <div style='border:1px solid #FFCDD2; background-color:#FFEBEE; padding:0.5rem; border-radius:6px; margin-bottom:0.5rem;'>
                                    <strong>{ticker}</strong> &mdash; Conviction Score: <strong style='color:#D32F2F'>{score}</strong><br>
                                    <em>Reason:</em> {reason}
                                </div>
                                """, unsafe_allow_html=True)

                    st.header("🎯 Sector Concentration Risk")
                    concentration = results.get('sector_concentration_analysis', {})
                    if concentration.get('top_sector'):
                        risk_level = concentration.get('risk_level', 'Unknown')
                        risk_class = {
                            'Critical': 'risk-high', 'High': 'risk-high',
                            'Moderate': 'risk-medium', 'Low': 'risk-low'
                        }.get(risk_level, '')
                        st.markdown(f"""
                        <div style='border:1px solid #ddd; padding:0.75rem; border-radius:6px; margin-bottom:0.5rem;'>
                            Largest sector: <strong>{concentration.get('top_sector')}</strong>
                            ({concentration.get('concentration_percentage', 0):.1f}% of portfolio) —
                            Risk level: <span class="{risk_class}">{risk_level}</span><br>
                            {concentration.get('recommendation', '')}
                        </div>
                        """, unsafe_allow_html=True)

                    if show_tax_analysis:
                            st.header("💰 Tax Loss Harvesting Opportunities")
                            tax_losses = results.get('tax_loss_harvesting', [])

                            if tax_losses and tax_losses[0] != "No tax loss harvesting opportunities currently available.":
                                for i, recommendation in enumerate(tax_losses[:5], 1):
                                    st.markdown(f"""
                                    <div style='border:1px solid #BBDEFB; background-color:#E3F2FD; padding:0.5rem; border-radius:6px; margin-bottom:0.5rem;'>
                                        {i}. {recommendation}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style='border:1px solid #C8E6C9; background-color:#E8F5E9; padding:0.5rem; border-radius:6px; margin-bottom:0.5rem;'>
                                    ✅ <strong>No tax loss harvesting opportunities at this time</strong>
                                </div>
                                """, unsafe_allow_html=True)

                    # 📝 Executive Investment Memo
                    st.header("📝 Executive Investment Memo")
                    memo = results.get('giuseppe_memo', '')

                    if memo:
                        # Download Button
                        memo_bytes = memo.encode()
                        b64 = base64.b64encode(memo_bytes).decode()
                        href = f'<a href="data:text/plain;base64,{b64}" download="investment_memo_{datetime.now().strftime("%Y%m%d")}.txt">📥 Download Memo</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Display Memo
                        with st.container():
                            render_giuseppe_memo(memo)

            with logsTab:
                if trade_log is not None:
                    results = validate_trade_log(trade_log, portfolio_df.copy())
                    if results['errors']:
                        st.error("Errors:\n" + "\n".join(results['errors']))
                    if results['warnings']:
                        st.warning("Warnings:\n" + "\n".join(results['warnings']))

                    # Cash Flow Summary - Deposit/Withdrawal/Dividend rows plus
                    # Buy/Sell proceeds, i.e. everything reconcile_cash_flows()
                    # already folded into the CASH position for this analysis.
                    st.subheader("💵 Cash Flow Summary")
                    cash_flows = reconcile_cash_flows(trade_log)
                    cf_col1, cf_col2, cf_col3, cf_col4, cf_col5 = st.columns(5)
                    cf_col1.metric("Deposits", format_currency(cash_flows['deposits']))
                    cf_col2.metric("Withdrawals", format_currency(cash_flows['withdrawals']))
                    cf_col3.metric("Dividends", format_currency(cash_flows['dividends']))
                    cf_col4.metric("Net Trading Flow", format_currency(cash_flows['sell_proceeds'] - cash_flows['buy_cost']))
                    cf_col5.metric("Net Cash Change", format_currency(cash_flows['net_cash_flow']))
                    st.caption("Net Cash Change is added to (or subtracted from) the portfolio's CASH position for this analysis.")

                    trade_log['Date'] = pd.to_datetime(trade_log['Date'])
                    trade_log = trade_log.sort_values('Date')
                    trade_log['Action'] = trade_log['Action'].str.lower().str.strip()
                    trade_log['Ticker'] = trade_log['Ticker'].str.upper()
                    trade_log['Quantity'] = pd.to_numeric(trade_log['Quantity'])
                    trade_log['Price'] = pd.to_numeric(trade_log['Price'])
                    trade_log['Value'] = trade_log['Quantity'] * trade_log['Price']
                    trade_log["Date"] = pd.to_datetime(trade_log["Date"])
                    # Compute realized PnL
                    buy_prices = trade_log[trade_log["Action"] == "buy"].set_index("Ticker")
                    sell_prices = trade_log[trade_log["Action"] == "sell"].set_index("Ticker")
                    common_tickers = buy_prices.index.intersection(sell_prices.index)
                    summary = []
                    for ticker in common_tickers:
                        buy = buy_prices.loc[ticker]
                        sell = sell_prices.loc[ticker]

                        if isinstance(buy, pd.DataFrame):
                            buy = buy.iloc[0]
                        if isinstance(sell, pd.DataFrame):
                            sell = sell.iloc[0]

                        entry_value = buy["Quantity"] * buy["Price"]
                        exit_value = sell["Quantity"] * sell["Price"]
                        pnl = exit_value - entry_value
                        pct_pnl = (exit_value / entry_value - 1) * 100
                        avg_conv = (buy["Conviction_Score"] + sell["Conviction_Score"]) / 2

                        summary.append({
                            "Ticker": ticker,
                            "Entry (MAD)": entry_value,
                            "Exit (MAD)": exit_value,
                            "PnL (MAD)": pnl,
                            "PnL (%)": pct_pnl,
                            "Conviction": avg_conv
                        })

                    summary_df = pd.DataFrame(summary)

                    insights = []

                    for _, row in summary_df.iterrows():
                        if row["Conviction"] > 0.7 and row["PnL (MAD)"] < 0:
                            insights.append(f"❌ High conviction on {row['Ticker']} led to a loss — review your thesis.")
                        elif row["Conviction"] < 0.5 and row["PnL (MAD)"] > 0:
                            insights.append(f"🤔 Low conviction on {row['Ticker']} turned into profit — consider trusting your gut more.")
                        elif abs(row["PnL (%)"]) > 10:
                            direction = "gain" if row["PnL (MAD)"] > 0 else "loss"
                            insights.append(f"📈 Significant {direction} on {row['Ticker']}: {row['PnL (%)']:.2f}%")

                    with st.expander("Show Full Trade Log"):
                        st.dataframe(trade_log)

                    # Summary table
                    st.subheader("📊 Trade Performance Summary")
                    st.dataframe(summary_df.style.format({"PnL (MAD)": "{:.2f}", "PnL (%)": "{:.2f}"}))

                    # Insights
                    st.subheader("🧠 Behavioral Insights")
                    if insights:
                        for note in insights:
                            with st.container():
                                st.markdown(f"""
                                <div style="background-color:#E3F2FD; border-left: 6px solid #2196F3; padding: 1em; border-radius: 5px; margin-bottom: 10px;">
                                    <strong>💡 Insight :</strong><br>{note}
                                </div>
                                """, unsafe_allow_html=True)
                            # st.info(note)
                    else:
                        st.success("No behavioral patterns detected.")

                    fig_pnl = go.Figure()
                    if "Ticker" in summary_df.columns:
                        fig_pnl.add_trace(go.Bar(
                            x=summary_df["Ticker"],
                            y=summary_df["PnL (MAD)"],
                            marker_color=summary_df["PnL (MAD)"].apply(lambda x: "#4CAF50" if x >= 0 else "#F44336"),
                            text=summary_df["PnL (MAD)"].map("{:,.2f} MAD".format),
                            textposition="auto"
                        ))

                        fig_pnl.update_layout(
                            title="💵 Realized PnL per Ticker",
                            yaxis_title="PnL (MAD)",
                            xaxis_title="Ticker",
                            height=400
                        )

                        st.plotly_chart(fig_pnl, use_container_width=True)

                        fig_conv = go.Figure(data=go.Scatter(
                        x=summary_df["Conviction"],
                        y=summary_df["PnL (%)"],
                        mode='markers+text',
                        text=summary_df["Ticker"],
                        marker=dict(
                            size=10,
                            color=summary_df["PnL (%)"],
                            colorscale="RdYlGn",
                            colorbar=dict(title="PnL (%)")
                        )
                        ))

                        fig_conv.update_layout(
                            title="🎯 Conviction vs. PnL",
                            xaxis_title="Conviction Score",
                            yaxis_title="PnL (%)",
                            height=400
                        )

                        st.plotly_chart(fig_conv, use_container_width=True)

            with exportTab:
                results = st.session_state["analysis_results"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Export portfolio with analysis
                    enhanced_portfolio = results.get('portfolio_df', pd.DataFrame())
                    if not enhanced_portfolio.empty:
                        csv = enhanced_portfolio.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:text/csv;base64,{b64}" download="portfolio_analysis_{datetime.now().strftime("%Y%m%d")}.csv">📊 Download Enhanced Portfolio</a>'
                        st.markdown(href, unsafe_allow_html=True)

                with col2:
                    # Export correlation matrix
                    corr_matrix = results.get('correlation_matrix', pd.DataFrame())
                    if not corr_matrix.empty:
                        csv = corr_matrix.to_csv()
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:text/csv;base64,{b64}" download="correlation_matrix_{datetime.now().strftime("%Y%m%d")}.csv">🔗 Download Correlation Matrix</a>'
                        st.markdown(href, unsafe_allow_html=True)

                with col3:
                    # Export all metrics
                    metrics_data = {
                        'Metric': ['Portfolio Value', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',  'Tail Ratio', 'Omega Ratio',
                                'Max Drawdown', 'Annual Return', 'Annual Volatility'],
                        'Value': [
                            portfolio_value,
                            metrics.get('sharpe_ratio', 0),
                            metrics.get('sortino_ratio', 0),
                            metrics.get('calmar_ratio', 0),
                            metrics.get('tail_ratio', 0),
                            metrics.get('omega_ratio', 0),
                            metrics.get('max_drawdown', 0),
                            metrics.get('annual_return', 0),
                            metrics.get('annual_volatility', 0)
                        ]
                    }
                    metrics_df = pd.DataFrame(metrics_data)
                    csv = metrics_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:text/csv;base64,{b64}" download="portfolio_metrics_{datetime.now().strftime("%Y%m%d")}.csv">📈 Download Metrics</a>'
                    st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
