"""
Portfolio Analysis Engine (Casablanca Stock Exchange)
======================================================
Core analytical engine for portfolio risk management, adapted for the
Moroccan stock market. Historical prices come from MongoDB (see
data_fetcher.py), the benchmark is the MSI20 index, and sector exposure
replaces the old US-ETF-based "Dragon Portfolio" allocation targets.

Author: Dragon Portfolio Analytics Engine
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings
import logging
from scipy import stats
from data_fetcher import fetch_portfolio_data
from sector_classification import classify_sector
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class PortfolioAnalyzer:
    """Portfolio analytics engine for Casablanca Stock Exchange (CSE) portfolios."""

    def __init__(self, risk_free_rate: float = 0.0275, tax_rate_gains: float = 0.15,
                 tax_rate_dividends: float = 0.15, benchmark_ticker: str = 'MSI20'):
        """
        Initialize the analyzer with configuration parameters.

        Args:
            risk_free_rate: Annualized risk-free rate used in Sharpe/Sortino/Calmar.
                Defaults to an approximate Bank Al-Maghrib policy-rate level (2.75%) -
                update this to the actual current Moroccan T-bill/BAM rate as needed.
            tax_rate_gains: Flat capital-gains tax rate applied in tax-loss-harvesting estimates.
            tax_rate_dividends: Flat dividend withholding tax rate (currently informational only).
            benchmark_ticker: Ticker used as the market benchmark (MSI20 by default).
        """
        self.risk_free_rate = risk_free_rate
        self.tax_rate_gains = tax_rate_gains
        self.tax_rate_dividends = tax_rate_dividends
        self.benchmark_ticker = benchmark_ticker
        self.trading_days = 252
        self.omega_ratio_threeshold = 0

        # Realized-volatility thresholds used for regime detection, in place of
        # VIX levels (Morocco has no widely-quoted implied-volatility index).
        # These are starting approximations for MSI20 annualized realized vol -
        # tune them once enough live history is available.
        self.low_vol_threshold = 15.0
        self.high_vol_threshold = 25.0

    def detect_market_regime(self, historical_data: pd.DataFrame) -> str:
        """
        Detect current market regime using the benchmark's trend (200-day SMA)
        and its own trailing realized volatility as a VIX-style fear gauge.

        Args:
            historical_data: DataFrame with historical price data.

        Returns:
            Market regime string: 'Bull Complacent', 'Crisis / Bear', or 'Neutral / Transition'.
        """
        try:
            if self.benchmark_ticker not in historical_data.columns:
                logger.warning(f"{self.benchmark_ticker} data not available for regime detection")
                return "Neutral / Transition"

            benchmark_prices = historical_data[self.benchmark_ticker].replace(0, np.nan).dropna()
            if len(benchmark_prices) < 200:
                logger.warning("Insufficient benchmark data for 200-day SMA")
                return "Neutral / Transition"

            sma_200 = benchmark_prices.rolling(window=200).mean()
            current_price = benchmark_prices.iloc[-1]
            current_sma = sma_200.iloc[-1]

            # Realized volatility proxy: trailing 20-day annualized volatility.
            returns = benchmark_prices.pct_change().dropna()
            realized_vol = returns.iloc[-20:].std() * np.sqrt(self.trading_days) * 100

            above_sma = current_price > current_sma

            if above_sma and realized_vol < self.low_vol_threshold:
                return "Bull Complacent"
            elif not above_sma or realized_vol > self.high_vol_threshold:
                return "Crisis / Bear"
            else:
                return "Neutral / Transition"

        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return "Neutral / Transition"

    def calculate_portfolio_metrics(self, portfolio_df: pd.DataFrame, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio-level risk-adjusted performance metrics.

        Args:
            portfolio_df: DataFrame with portfolio positions.
            historical_data: DataFrame with historical price data.

        Returns:
            Dictionary with Sharpe, Sortino, Calmar, Omega, Tail ratios, drawdown, etc.
        """
        try:
            logger.info("Starting portfolio metrics calculation")
            portfolio_tickers = portfolio_df['Ticker'].unique()
            logger.info(f"Portfolio tickers: {portfolio_tickers.tolist()}")
            available_tickers = [t for t in portfolio_tickers if t in historical_data.columns]
            logger.info(f"Available tickers in historical data: {available_tickers}")

            empty_metrics = {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }

            if not available_tickers:
                logger.error("No portfolio tickers found in historical data")
                return empty_metrics

            total_value = portfolio_df['Market_Value'].sum()
            logger.info(f"Total market Value: {total_value:,.2f}")

            filtered_portfolio = portfolio_df[portfolio_df['Ticker'].isin(available_tickers)].copy()
            filtered_portfolio['signed_mv'] = filtered_portfolio['Market_Value']
            filtered_portfolio['gross_mv'] = filtered_portfolio['Market_Value'].abs()

            gross_total = filtered_portfolio['gross_mv'].sum()
            if gross_total == 0 or np.isnan(gross_total):
                logger.error("Gross exposure is zero/NaN; cannot compute weights")
                return empty_metrics

            weights = (filtered_portfolio.set_index('Ticker')['signed_mv'] / gross_total)
            logger.info(f"Signed, gross-normalized weights:\n{weights}")

            returns = historical_data[weights.index].pct_change().dropna()
            logger.info(f"Returns shape: {returns.shape}")

            portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
            logger.info(f"Sample Portfolio returns:\n{portfolio_returns.head()}")

            # Sharpe Ratio
            excess_returns = portfolio_returns - self.risk_free_rate / self.trading_days
            sharpe_ratio = np.sqrt(self.trading_days) * excess_returns.mean() / portfolio_returns.std()
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            logger.info(f"Max Drawdown: {max_drawdown:.4%}")

            cumulative_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + cumulative_return) ** (self.trading_days / len(portfolio_returns)) - 1
            annual_return *= 100
            logger.info(f"Annual Return: {annual_return:.4%}")

            annual_volatility = portfolio_returns.std() * np.sqrt(self.trading_days) * 100
            logger.info(f"Annual Volatility: {annual_volatility:.4%}")

            # Sortino Ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(self.trading_days)
            sortino_ratio = (annual_return / 100 - self.risk_free_rate) / downside_std if downside_std > 0 else 0
            logger.info(f"Sortino Ratio: {sortino_ratio:.4f}")

            # Calmar Ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            logger.info(f"Calmar Ratio: {calmar_ratio:.4f}")

            # Omega Ratio
            excess_returns = portfolio_returns - self.omega_ratio_threeshold
            positive_sum = excess_returns[excess_returns > 0].sum()
            negative_sum = -excess_returns[excess_returns < 0].sum()
            omega_ratio = positive_sum / negative_sum if negative_sum != 0 else np.nan
            logger.info(f"Omega Ratio: {omega_ratio:.4f}")

            # Tail Ratio
            tail_ratio = np.percentile(portfolio_returns, 95) / abs(np.percentile(portfolio_returns, 5))
            logger.info(f"Tail Ratio (P95 / |P5|): {tail_ratio:.4f}")

            # Max Drawdown Duration (days)
            nav_series = (1 + portfolio_returns).cumprod()
            peak_series = nav_series.cummax()
            drawdown_flags = (nav_series < peak_series).astype(int)

            max_drawdown_duration = 0
            current_duration = 0
            for flag in drawdown_flags:
                if flag:
                    current_duration += 1
                    max_drawdown_duration = max(max_drawdown_duration, current_duration)
                else:
                    current_duration = 0

            logger.info(f"Max Drawdown Duration: {max_drawdown_duration} days")

            return {
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown': round(max_drawdown, 2),
                'annual_return': round(annual_return, 2),
                'annual_volatility': round(annual_volatility, 2),
                'sortino_ratio': round(sortino_ratio, 3),
                'calmar_ratio': round(calmar_ratio, 3),
                'omega_ratio': round(omega_ratio, 3),
                'tail_ratio': round(tail_ratio, 3),
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }

    def mean_variance_optimizer(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate optimal portfolio weights using Mean-Variance Optimization (max Sharpe).
        """
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_assets = len(mean_returns)

        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate / self.trading_days) / portfolio_vol
            return -sharpe_ratio

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        init_guess = np.array([1.0 / num_assets] * num_assets)

        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = pd.Series(result.x, index=returns.columns)

        return optimal_weights

    def run_sector_allocation(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Descriptive breakdown of portfolio exposure by Casablanca market sector.

        This replaces the old Chris-Cole "Dragon Portfolio" target-allocation
        analysis: there's no liquid Moroccan equivalent for commodity-trend,
        long-volatility, or gold ETFs, so this is purely descriptive (current
        weights only - no targets, deviations, or rebalancing recommendations).

        Args:
            portfolio_df: DataFrame with portfolio positions.

        Returns:
            Dictionary with sector allocation percentages.
        """
        try:
            df = portfolio_df.copy()
            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            if 'Category' not in df.columns:
                df['Category'] = ''
            df['Category'] = df['Category'].astype(str).str.strip()
            df['Market_Value'] = pd.to_numeric(df['Market_Value'], errors='coerce').fillna(0)

            df['Sector'] = df.apply(lambda r: classify_sector(r['Ticker'], r['Category']), axis=1)
            df['Abs_Exposure'] = df['Market_Value'].abs()

            total_value = float(df['Abs_Exposure'].sum())
            by_sector = df.groupby('Sector', dropna=False)['Abs_Exposure'].sum().sort_values(ascending=False)

            allocations = (by_sector / total_value * 100).to_dict() if total_value > 0 else {}

            return {
                'allocations': allocations,
                'total_value': total_value,
                'unclassified_pct': allocations.get('Unclassified', 0.0)
            }

        except Exception:
            logger.exception("Error in sector allocation analysis")
            return {
                'allocations': {},
                'total_value': 0.0,
                'unclassified_pct': 0.0
            }

    def run_correlation_analysis(self, historical_data: pd.DataFrame,
                                  portfolio_df: pd.DataFrame = None) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Calculate correlation matrix and identify highly correlated pairs.

        Args:
            historical_data: DataFrame with historical price data (portfolio
                tickers plus benchmarks like MASI/MSI20).
            portfolio_df: DataFrame with portfolio positions. Used both to
                flag same-sector concentration risk on top of raw
                correlation, and to restrict the analysis to actual
                holdings - benchmarks aren't positions and shouldn't be
                treated as correlation candidates.

        Returns:
            Tuple of (high correlation alerts, full correlation matrix).
        """
        try:
            if portfolio_df is not None:
                portfolio_tickers = portfolio_df['Ticker'].astype(str).str.strip().str.upper().unique().tolist()
                price_data = historical_data[[t for t in portfolio_tickers if t in historical_data.columns]]
            else:
                price_data = historical_data

            returns = price_data.pct_change().dropna()
            correlation_matrix = returns.corr()

            ticker_sector = {}
            if portfolio_df is not None:
                for _, row in portfolio_df.iterrows():
                    ticker_sector[str(row['Ticker']).strip().upper()] = classify_sector(
                        row['Ticker'], row.get('Category', '')
                    )

            alerts = []
            n = len(correlation_matrix.columns)

            for i in range(n):
                for j in range(i + 1, n):
                    ticker1 = correlation_matrix.columns[i]
                    ticker2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]

                    if corr_value > 0.75 and ticker1 != ticker2:
                        sector1 = ticker_sector.get(ticker1)
                        sector2 = ticker_sector.get(ticker2)
                        same_sector = sector1 is not None and sector1 == sector2 and sector1 != 'Unclassified'

                        alert = {
                            'pair': (ticker1, ticker2),
                            'correlation': round(corr_value, 3),
                            'risk_type': f'Same-Sector Concentration ({sector1})' if same_sector else 'General Correlation'
                        }

                        alerts.append(alert)

            alerts.sort(key=lambda x: x['correlation'], reverse=True)

            return alerts, correlation_matrix

        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return [], pd.DataFrame()

    def calculate_conviction_scores(self, portfolio_df: pd.DataFrame, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate 3-part conviction score for each asset.

        Components:
        1. Performance: 1-year return vs the benchmark (MSI20)
        2. Risk: Volatility relative to return
        3. Allocation: Position size relative to portfolio

        Args:
            portfolio_df: DataFrame with portfolio positions.
            historical_data: DataFrame with historical price data.

        Returns:
            Dictionary with conviction scores and low conviction alerts.
        """
        try:
            scores = {}
            low_conviction = []

            benchmark_return = 0.0
            if self.benchmark_ticker in historical_data.columns:
                benchmark_1y = historical_data[self.benchmark_ticker].iloc[-252:].pct_change().dropna()
                benchmark_return = (benchmark_1y + 1).prod() - 1

            total_value = abs(portfolio_df['Market_Value']).sum()

            for _, position in portfolio_df.iterrows():
                ticker = position['Ticker']

                if ticker not in historical_data.columns:
                    continue

                # Component 1: Performance (0-0.4)
                ticker_1y = historical_data[ticker].iloc[-252:].pct_change().dropna()
                ticker_return = (ticker_1y + 1).prod() - 1
                outperformance = ticker_return - benchmark_return
                perf_score = min(0.4, max(0, 0.2 + outperformance))

                # Component 2: Risk-adjusted return (0-0.3)
                volatility = ticker_1y.std() * np.sqrt(252)
                sharpe = (ticker_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                risk_score = min(0.3, max(0, sharpe * 0.15))

                # Component 3: Allocation confidence (0-0.3)
                position_weight = abs(position['Market_Value']) / total_value
                if 0.02 <= position_weight <= 0.08:
                    alloc_score = 0.3
                elif position_weight < 0.02:
                    alloc_score = position_weight * 15
                else:
                    alloc_score = 0.3 - (position_weight - 0.08) * 2
                alloc_score = max(0, min(0.3, alloc_score))

                total_score = perf_score + risk_score + alloc_score

                scores[ticker] = {
                    'total_score': round(total_score, 3),
                    'performance_score': round(perf_score, 3),
                    'risk_score': round(risk_score, 3),
                    'allocation_score': round(alloc_score, 3),
                    'details': {
                        '1y_return': f"{ticker_return * 100:.1f}%",
                        'volatility': f"{volatility * 100:.1f}%",
                        'weight': f"{position_weight * 100:.1f}%"
                    }
                }

                if total_score < 0.3:
                    low_conviction.append({
                        'ticker': ticker,
                        'score': round(total_score, 3),
                        'reason': self._get_low_conviction_reason(perf_score, risk_score, alloc_score)
                    })

            return {
                'scores': scores,
                'low_conviction_alerts': sorted(low_conviction, key=lambda x: x['score'])
            }

        except Exception as e:
            logger.error(f"Error calculating conviction scores: {str(e)}")
            return {'scores': {}, 'low_conviction_alerts': []}

    def _get_low_conviction_reason(self, perf_score: float, risk_score: float, alloc_score: float) -> str:
        """Generate explanation for low conviction score."""
        reasons = []
        if perf_score < 0.15:
            reasons.append("underperforming")
        if risk_score < 0.1:
            reasons.append("poor risk-adjusted returns")
        if alloc_score < 0.1:
            reasons.append("suboptimal position size")
        return ", ".join(reasons) if reasons else "multiple factors"

    def identify_tax_loss_harvesting(self, portfolio_df: pd.DataFrame) -> List[str]:
        """
        Identify tax loss harvesting opportunities.

        Uses flat Moroccan capital-gains/dividend tax rates (self.tax_rate_gains,
        self.tax_rate_dividends). Unlike the US version, this does NOT suggest a
        replacement ticker (no wash-sale rule to route around, and no equivalent
        Moroccan ETF ecosystem to swap into) - it simply flags the loss and the
        estimated tax value of realizing it.

        Args:
            portfolio_df: DataFrame with portfolio positions.

        Returns:
            List of tax loss harvesting recommendations.
        """
        try:
            recommendations = []

            loss_positions = portfolio_df[portfolio_df['Unrealized_PnL'] < 0].copy()

            if loss_positions.empty:
                return ["No tax loss harvesting opportunities currently available."]

            loss_positions = loss_positions.sort_values('Unrealized_PnL')

            for _, position in loss_positions.iterrows():
                ticker = position['Ticker']
                loss_amount = abs(position['Unrealized_PnL'])
                tax_value = loss_amount * self.tax_rate_gains

                recommendation = (
                    f"Consider selling {ticker} to realize a loss of {loss_amount:,.0f} MAD, "
                    f"offsetting capital gains taxed at {self.tax_rate_gains * 100:.0f}% "
                    f"(potential tax saving of ~{tax_value:,.0f} MAD)."
                )

                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"Error in tax loss harvesting: {str(e)}")
            return ["Unable to analyze tax loss harvesting opportunities."]

    def create_portfolio_graphs(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create interactive Plotly graphs for portfolio visualization.

        Args:
            analysis_results: Complete analysis results dictionary.

        Returns:
            Dictionary of Plotly figure objects.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        from itertools import groupby

        graphs = {}

        try:
            # 0. Portfolio NAV vs Benchmark performance chart
            historical_data = analysis_results.get('historical_data', pd.DataFrame())
            portfolio_df = analysis_results.get('portfolio_df', pd.DataFrame())
            backtest_results = analysis_results.get('backtest_results', {})
            portfolio_nav = backtest_results.get('portfolio_value_series', pd.Series())
            if not portfolio_nav.empty:
                series_to_plot = {"Portfolio": portfolio_nav}
                if self.benchmark_ticker in historical_data.columns:
                    series_to_plot[self.benchmark_ticker] = historical_data[self.benchmark_ticker]
                if "MASI" in historical_data.columns and "MASI" != self.benchmark_ticker:
                    series_to_plot["MASI"] = historical_data["MASI"]

                if len(series_to_plot) > 1:
                    df = pd.concat({name: s.rename(name) for name, s in series_to_plot.items()}, axis=1)
                    df = df.dropna()
                    df_rebased = df / df.iloc[0] * 100

                    line_styles = {
                        "Portfolio": dict(color='red', width=2),
                        self.benchmark_ticker: dict(color='#1976D2', width=2, dash='dash'),
                        "MASI": dict(color='#FF9800', width=2, dash='dot'),
                    }

                    fig_perf = go.Figure()
                    for name in df_rebased.columns:
                        fig_perf.add_trace(go.Scatter(
                            x=df_rebased.index, y=df_rebased[name],
                            name=name, line=line_styles.get(name, dict(color='#888', width=2, dash='dashdot'))
                        ))

                    benchmark_names = [n for n in series_to_plot if n != "Portfolio"]
                    fig_perf.update_layout(
                        title=f"Portfolio vs {' & '.join(benchmark_names)} – Rebased to 100",
                        xaxis_title='Date', yaxis_title='Value (Rebased)',
                        hovermode='x unified', height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    graphs["performance_chart"] = fig_perf

            # 1. Sector Allocation Pie Chart
            sector_allocation = analysis_results.get('sector_allocation', {})
            allocations = sector_allocation.get('allocations', {})

            if allocations:
                labels = list(allocations.keys())
                values = list(allocations.values())

                fig_allocation = go.Figure()
                fig_allocation.add_trace(go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='outside',
                    textfont_size=12
                ))
                fig_allocation.add_annotation(
                    text="Sector<br>Allocation",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )
                fig_allocation.update_layout(
                    title="Portfolio Allocation by Casablanca Market Sector",
                    height=500,
                    showlegend=True,
                    font=dict(size=14)
                )
                graphs['allocation_pie'] = fig_allocation

            # 2. Correlation Heatmap
            correlation_matrix = analysis_results.get('correlation_matrix', pd.DataFrame())

            if not correlation_matrix.empty:
                fig_correlation = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=correlation_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                fig_correlation.update_layout(
                    title="Asset Correlation Matrix (Alert: Correlations > 0.75)",
                    height=600,
                    xaxis_title="Assets",
                    yaxis_title="Assets"
                )
                graphs['correlation_heatmap'] = fig_correlation

            # 3. Conviction Scores Bar Chart
            conviction_analysis = analysis_results.get('conviction_analysis', {})
            scores = conviction_analysis.get('scores', {})

            if scores:
                tickers = list(scores.keys())
                total_scores = [scores[t]['total_score'] for t in tickers]
                perf_scores = [scores[t]['performance_score'] for t in tickers]
                risk_scores = [scores[t]['risk_score'] for t in tickers]
                alloc_scores = [scores[t]['allocation_score'] for t in tickers]

                sorted_indices = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)
                tickers = [tickers[i] for i in sorted_indices]
                perf_scores = [perf_scores[i] for i in sorted_indices]
                risk_scores = [risk_scores[i] for i in sorted_indices]
                alloc_scores = [alloc_scores[i] for i in sorted_indices]

                fig_conviction = go.Figure()
                fig_conviction.add_trace(go.Bar(name='Performance', x=tickers, y=perf_scores, marker_color='#4CAF50'))
                fig_conviction.add_trace(go.Bar(name='Risk-Adjusted', x=tickers, y=risk_scores, marker_color='#2196F3'))
                fig_conviction.add_trace(go.Bar(name='Allocation', x=tickers, y=alloc_scores, marker_color='#FF9800'))
                fig_conviction.add_hline(
                    y=0.3, line_dash="dash", line_color="red",
                    annotation_text="Low Conviction Threshold (0.3)"
                )
                fig_conviction.update_layout(
                    title="Conviction Scores by Asset (Max: 1.0)",
                    barmode='stack', height=500,
                    xaxis_title="Asset", yaxis_title="Conviction Score",
                    showlegend=True, yaxis=dict(range=[0, 1])
                )
                graphs['conviction_bars'] = fig_conviction

            # 4. Regime Timeline Chart (benchmark price + realized volatility)
            if self.benchmark_ticker in historical_data.columns:
                benchmark_prices = historical_data[self.benchmark_ticker].replace(0, np.nan).dropna()
                benchmark_returns = benchmark_prices.pct_change()
                realized_vol = benchmark_returns.rolling(20).std() * np.sqrt(self.trading_days) * 100
                realized_vol = realized_vol.dropna()

                common_index = benchmark_prices.index.intersection(realized_vol.index)
                benchmark_prices = benchmark_prices.loc[common_index]
                realized_vol = realized_vol.loc[common_index]
                benchmark_sma_200 = benchmark_prices.rolling(window=200).mean()

                regime = []
                for date in common_index:
                    price = benchmark_prices.loc[date]
                    sma = benchmark_sma_200.loc[date]
                    vol = realized_vol.loc[date]
                    if pd.isna(sma) or pd.isna(vol):
                        regime.append('Neutral')
                    elif price > sma and vol < self.low_vol_threshold:
                        regime.append('Bull')
                    elif price < sma and vol > self.high_vol_threshold:
                        regime.append('Bear')
                    else:
                        regime.append('Neutral')
                regime_colors = {
                    'Bull': 'rgba(76, 175, 80, 0.1)',
                    'Bear': 'rgba(244, 67, 54, 0.1)',
                    'Neutral': 'rgba(255, 235, 59, 0.1)'
                }

                fig_regime = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.7, 0.3],
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(f'{self.benchmark_ticker} Price with 200-Day SMA & Regime Overlay',
                                    'Realized Volatility (20-day, annualized)'),
                    specs=[[{"type": "xy"}], [{"type": "xy"}]]
                )

                regime_series = pd.Series(regime, index=common_index)
                shapes = []
                for label, group in groupby(regime_series.items(), lambda x: x[1]):
                    group = list(group)
                    start_date = group[0][0]
                    end_date = group[-1][0]
                    shapes.append(dict(
                        type='rect', xref='x', yref='paper',
                        x0=start_date, x1=end_date, y0=0, y1=1,
                        fillcolor=regime_colors[label], opacity=1, layer='below', line_width=0
                    ))
                fig_regime.update_layout(shapes=shapes)

                fig_regime.add_trace(
                    go.Scatter(x=benchmark_prices.index, y=benchmark_prices.values,
                               name=f'{self.benchmark_ticker} Price', line=dict(color='#1976D2', width=1)),
                    row=1, col=1
                )
                fig_regime.add_trace(
                    go.Scatter(x=benchmark_sma_200.index, y=benchmark_sma_200.values,
                               name='200-Day SMA', line=dict(color='#D32F2F', width=2, dash='dash')),
                    row=1, col=1
                )
                fig_regime.add_trace(
                    go.Scatter(x=realized_vol.index, y=realized_vol.values,
                               name='Realized Volatility', fill='tozeroy', line=dict(color='#FF5722', width=1)),
                    row=2, col=1
                )

                fig_regime.add_hline(y=self.low_vol_threshold, row=2, col=1, line_dash="dash",
                                      line_color="orange", annotation_text=f"Low Vol {self.low_vol_threshold:.0f}%")
                fig_regime.add_hline(y=self.high_vol_threshold, row=2, col=1, line_dash="dash",
                                      line_color="red", annotation_text=f"High Vol {self.high_vol_threshold:.0f}%")

                fig_regime.update_layout(
                    title="Market Regime Indicators",
                    height=700, showlegend=True, hovermode='x unified'
                )
                fig_regime.update_xaxes(title_text="Date", row=2, col=1)
                fig_regime.update_yaxes(title_text="Price (MAD)", row=1, col=1)
                fig_regime.update_yaxes(title_text="Realized Volatility (%)", row=2, col=1)

                graphs['regime_timeline'] = fig_regime

            # 5. Drawdown Chart
            metrics = analysis_results.get('portfolio_metrics', {})

            if not historical_data.empty and not portfolio_df.empty:
                portfolio_tickers = portfolio_df['Ticker'].unique()
                available_tickers = [t for t in portfolio_tickers if t in historical_data.columns]

                if available_tickers:
                    total_value = portfolio_df['Market_Value'].sum()
                    weights = {}
                    for _, row in portfolio_df.iterrows():
                        if row['Ticker'] in available_tickers:
                            weights[row['Ticker']] = row['Market_Value'] / total_value

                    returns = historical_data[available_tickers].pct_change().dropna()
                    portfolio_returns = pd.Series(0.0, index=returns.index)
                    for ticker, weight in weights.items():
                        portfolio_returns += returns[ticker] * weight

                    cumulative_returns = (1 + portfolio_returns).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = ((cumulative_returns - running_max) / running_max) * 100

                    fig_drawdown = go.Figure()
                    fig_drawdown.add_trace(go.Scatter(
                        x=drawdown.index, y=drawdown.values, fill='tozeroy', name='Drawdown',
                        line=dict(color='#D32F2F', width=1), fillcolor='rgba(211, 47, 47, 0.3)'
                    ))

                    recovery_periods = []
                    in_drawdown = False
                    start_date = None

                    for date, dd in drawdown.items():
                        if dd < -5 and not in_drawdown:
                            in_drawdown = True
                            start_date = date
                        elif dd > -1 and in_drawdown:
                            in_drawdown = False
                            if start_date:
                                recovery_periods.append({
                                    'start': start_date, 'end': date,
                                    'duration': (date - start_date).days
                                })

                    for period in recovery_periods[-3:]:
                        fig_drawdown.add_annotation(
                            x=period['end'], y=0,
                            text=f"Recovery: {period['duration']} days",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#4CAF50"
                        )

                    fig_drawdown.update_layout(
                        title=f"Portfolio Drawdown Analysis (Max: {metrics.get('max_drawdown', 0):.1f}%)",
                        height=400, xaxis_title="Date", yaxis_title="Drawdown (%)", hovermode='x unified'
                    )
                    graphs['drawdown_chart'] = fig_drawdown

            # 6. Performance Metrics Dashboard
            if metrics:
                fig_metrics = make_subplots(
                    rows=2, cols=5,
                    specs=[
                        [{"type": "indicator"}, None, {"type": "indicator"}, None, {"type": "indicator"}],
                        [None, {"type": "indicator"}, None, {"type": "indicator"}, None]
                    ],
                    subplot_titles=('Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 'Tail Ratio'),
                    vertical_spacing=0.05,
                    horizontal_spacing=0
                )

                fig_metrics.add_trace(
                    go.Indicator(
                        mode="gauge+number", value=metrics.get('sharpe_ratio', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [-1, 3]}, 'bar': {'color': "#4CAF50"},
                            'steps': [
                                {'range': [-1, 0], 'color': "#FFEBEE"},
                                {'range': [0, 1], 'color': "#FFF3E0"},
                                {'range': [1, 2], 'color': "#E8F5E9"},
                                {'range': [2, 3], 'color': "#C8E6C9"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1}
                        }
                    ), row=1, col=1
                )
                fig_metrics.add_trace(
                    go.Indicator(
                        mode="gauge+number", value=metrics.get('sortino_ratio', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [-1, 4]}, 'bar': {'color': "#2196F3"},
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1.5}
                        }
                    ), row=1, col=3
                )
                fig_metrics.add_trace(
                    go.Indicator(
                        mode="gauge+number", value=metrics.get('calmar_ratio', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 5]}, 'bar': {'color': "#FF9800"},
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1}
                        }
                    ), row=1, col=5
                )
                fig_metrics.add_trace(
                    go.Indicator(
                        mode="gauge+number", value=metrics.get('omega_ratio', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 3]}, 'bar': {'color': "#4CAF50"},
                            'steps': [
                                {'range': [0, 1], 'color': "#FFEBEE"},
                                {'range': [1, 1.5], 'color': "#FFF3E0"},
                                {'range': [1.5, 2.0], 'color': "#E8F5E9"},
                                {'range': [2.0, 3.0], 'color': "#C8E6C9"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1}
                        }
                    ), row=2, col=2
                )
                fig_metrics.add_trace(
                    go.Indicator(
                        mode="gauge+number", value=metrics.get('tail_ratio', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 4]}, 'bar': {'color': "#4CAF50"},
                            'steps': [
                                {'range': [0, 1], 'color': "#FFEBEE"},
                                {'range': [1, 2], 'color': "#FFF3E0"},
                                {'range': [2, 3], 'color': "#E8F5E9"},
                                {'range': [3, 4], 'color': "#C8E6C9"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1}
                        }
                    ), row=2, col=4
                )

                fig_metrics.update_layout(height=600, margin=dict(t=60, l=30, r=30, b=30))
                graphs['performance_metrics'] = fig_metrics

            return graphs

        except Exception as e:
            logger.error(f"Error creating graphs: {str(e)}")
            return {}

    def analyze_sector_concentration_risk(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Flags concentration risk in the portfolio's single largest sector.

        Repurposed from the original US-market "AI sector exposure" scoring
        (built around NVDA/META-style stocks), which has no equivalent among
        CSE listings. This generalizes the same warning mechanism to whatever
        sector happens to dominate the portfolio.

        Args:
            portfolio_df: DataFrame with portfolio positions.

        Returns:
            Dictionary with sector concentration analysis.
        """
        try:
            df = portfolio_df.copy()
            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            if 'Category' not in df.columns:
                df['Category'] = ''
            df['Category'] = df['Category'].astype(str).str.strip()
            df['Sector'] = df.apply(lambda r: classify_sector(r['Ticker'], r['Category']), axis=1)
            df['Abs_Exposure'] = pd.to_numeric(df['Market_Value'], errors='coerce').fillna(0).abs()

            total_value = df['Abs_Exposure'].sum()
            empty_result = {
                'concentration_percentage': 0,
                'top_sector': None,
                'risk_level': 'Unknown',
                'high_concentration_positions': [],
                'recommendation': 'No exposure to analyze.'
            }
            if total_value == 0:
                return empty_result

            by_sector = df.groupby('Sector')['Abs_Exposure'].sum().sort_values(ascending=False)
            top_sector = by_sector.index[0]
            top_value = by_sector.iloc[0]
            concentration_pct = top_value / total_value * 100

            positions = df[df['Sector'] == top_sector].copy()
            positions['weight'] = positions['Abs_Exposure'] / total_value * 100
            high_positions = positions.sort_values('weight', ascending=False)[
                ['Ticker', 'weight', 'Abs_Exposure']
            ].rename(columns={'Abs_Exposure': 'market_value'}).to_dict('records')

            if concentration_pct > 40:
                risk_level = "Critical"
                recommendation = (
                    f"Over 40% of the portfolio sits in a single sector ({top_sector}). "
                    "Consider diversifying into other CSE sectors to reduce concentration risk."
                )
            elif concentration_pct > 30:
                risk_level = "High"
                recommendation = (
                    f"Significant concentration in {top_sector}. Consider trimming positions "
                    "or diversifying into uncorrelated sectors."
                )
            elif concentration_pct > 20:
                risk_level = "Moderate"
                recommendation = f"Moderate concentration in {top_sector}; within a generally acceptable range."
            else:
                risk_level = "Low"
                recommendation = "Sector exposure is well diversified."

            return {
                'concentration_percentage': round(concentration_pct, 1),
                'top_sector': top_sector,
                'risk_level': risk_level,
                'high_concentration_positions': high_positions,
                'recommendation': recommendation
            }

        except Exception as e:
            logger.error(f"Error in sector concentration analysis: {str(e)}")
            return {
                'concentration_percentage': 0,
                'top_sector': None,
                'risk_level': 'Unknown',
                'high_concentration_positions': [],
                'recommendation': 'Unable to analyze sector concentration.'
            }

    def run_backtest(self, portfolio_df: pd.DataFrame, historical_data: pd.DataFrame,
                    start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Run a simple buy-and-hold backtest of the portfolio.

        Args:
            portfolio_df: DataFrame with portfolio positions.
            historical_data: DataFrame with historical price data.
            start_date: Backtest start date (YYYY-MM-DD).
            end_date: Backtest end date (YYYY-MM-DD).

        Returns:
            Dictionary with backtest results.
        """
        try:
            if start_date:
                historical_data = historical_data[historical_data.index >= start_date]
            if end_date:
                historical_data = historical_data[historical_data.index <= end_date]

            portfolio_tickers = portfolio_df['Ticker'].unique()
            available_tickers = [t for t in portfolio_tickers if t in historical_data.columns]

            if not available_tickers:
                logger.error("No portfolio tickers found in historical data for backtesting")
                return {}

            total_value = abs(portfolio_df['Market_Value']).sum()
            initial_weights = {}
            initial_capital = 1000000  # 1M MAD initial capital for backtest

            for _, row in portfolio_df.iterrows():
                if row['Ticker'] in available_tickers:
                    initial_weights[row['Ticker']] = row['Market_Value'] / total_value

            returns = historical_data[available_tickers].pct_change().dropna()

            portfolio_value = pd.Series(initial_capital, index=returns.index)

            for i in range(1, len(returns)):
                daily_return = 0
                for ticker, weight in initial_weights.items():
                    if ticker in returns.columns:
                        daily_return += returns[ticker].iloc[i] * weight

                portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + daily_return)

            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100

            years = len(returns) / self.trading_days
            annualized_return = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1/years) - 1) * 100

            running_max = portfolio_value.expanding().max()
            drawdown = (portfolio_value - running_max) / running_max * 100
            max_drawdown = drawdown.min()

            portfolio_returns = portfolio_value.pct_change().dropna()
            volatility = portfolio_returns.std() * np.sqrt(self.trading_days) * 100

            excess_returns = portfolio_returns - self.risk_free_rate / self.trading_days
            sharpe_ratio = np.sqrt(self.trading_days) * excess_returns.mean() / portfolio_returns.std()

            benchmark_return = 0
            if self.benchmark_ticker in historical_data.columns:
                benchmark_data = historical_data[self.benchmark_ticker].dropna()
                benchmark_data = benchmark_data[benchmark_data.index.isin(portfolio_value.index)]
                if len(benchmark_data) > 1:
                    benchmark_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[0] - 1) * 100

            return {
                'backtest_period': {
                    'start': portfolio_value.index[0].strftime('%Y-%m-%d'),
                    'end': portfolio_value.index[-1].strftime('%Y-%m-%d'),
                    'days': len(portfolio_value)
                },
                'performance': {
                    'total_return': round(total_return, 2),
                    'annualized_return': round(annualized_return, 2),
                    'volatility': round(volatility, 2),
                    'sharpe_ratio': round(sharpe_ratio, 3),
                    'max_drawdown': round(max_drawdown, 2),
                    'benchmark_return': round(benchmark_return, 2),
                    'excess_return_vs_benchmark': round(total_return - benchmark_return, 2)
                },
                'portfolio_value_series': portfolio_value,
                'drawdown_series': drawdown
            }

        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return {}

    def generate_ai_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate personalized AI recommendations based on analysis results.

        Args:
            analysis_results: Complete analysis results dictionary.

        Returns:
            AI-generated recommendations or fallback if API fails.
        """
        try:
            api_key = os.environ.get('OPENAI_API_KEY')

            if not api_key:
                logger.warning("No OpenAI API key found. Using fallback recommendations.")
                return self._generate_fallback_recommendations(analysis_results)

            context = self._prepare_ai_context(analysis_results)

            try:
                import openai
                openai.api_key = api_key

                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are Giuseppe Paleologo, a world-class institutional quantitative analyst
                            specializing in the Moroccan (Casablanca Stock Exchange) market.
                            Analyze the portfolio data and provide specific, actionable recommendations.
                            Focus on: 1) Immediate risk mitigation, 2) Sector diversification,
                            3) Tax optimization under Moroccan capital gains/dividend tax rules, 4) Current market conditions.
                            Be direct, data-driven, and specific with tickers and amounts in MAD."""
                        },
                        {
                            "role": "user",
                            "content": context
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )

                return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"OpenAI API call failed: {str(e)}")

                anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
                if anthropic_key:
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=anthropic_key)

                        response = client.messages.create(
                            model="claude-3-opus-20240229",
                            max_tokens=1500,
                            temperature=0.7,
                            system="""You are Giuseppe Paleologo, a world-class institutional quantitative analyst
                            specializing in the Moroccan (Casablanca Stock Exchange) market.
                            Analyze the portfolio data and provide specific, actionable recommendations.""",
                            messages=[
                                {"role": "user", "content": context}
                            ]
                        )

                        return response.content[0].text

                    except Exception as e:
                        logger.warning(f"Anthropic API call failed: {str(e)}")

            return self._generate_fallback_recommendations(analysis_results)

        except Exception as e:
            logger.error(f"Error generating AI recommendations: {str(e)}")
            return self._generate_fallback_recommendations(analysis_results)

    def _prepare_ai_context(self, analysis_results: Dict[str, Any]) -> str:
        """Prepare context for AI recommendation generation."""

        portfolio_value = analysis_results.get('portfolio_value', 0)
        regime = analysis_results.get('market_regime', 'Unknown')
        metrics = analysis_results.get('portfolio_metrics', {})
        sector_allocation = analysis_results.get('sector_allocation', {})
        concentration = analysis_results.get('sector_concentration_analysis', {})
        correlations = analysis_results.get('correlation_alerts', [])
        portfolio_df = analysis_results.get('portfolio_df', pd.DataFrame())

        context = f"""
        PORTFOLIO ANALYSIS CONTEXT (Casablanca Stock Exchange)
        Date: {datetime.now().strftime('%Y-%m-%d')}
        Portfolio Value: {portfolio_value:,.2f} MAD
        Market Regime: {regime}

        PERFORMANCE METRICS:
        - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
        - Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}
        - Tail Ratio: {metrics.get('tail_ratio', 0):.3f}
        - Omega Ratio: {metrics.get('omega_ratio', 0):.3f}
        - Max Drawdown: {metrics.get('max_drawdown', 0):.1f}%
        - Annual Return: {metrics.get('annual_return', 0):.1f}%

        SECTOR ALLOCATION:
        """

        for sector, pct in sector_allocation.get('allocations', {}).items():
            context += f"- {sector}: {pct:.1f}%\n"

        context += f"""
        SECTOR CONCENTRATION RISK:
        - Largest Sector: {concentration.get('top_sector', 'N/A')} ({concentration.get('concentration_percentage', 0):.1f}%)
        - Risk Level: {concentration.get('risk_level', 'Unknown')}

        TOP POSITIONS:
        """

        if not portfolio_df.empty:
            top_positions = portfolio_df.nlargest(5, 'Market_Value')[['Ticker', 'Market_Value', 'Unrealized_PnL_Pct']]
            for _, pos in top_positions.iterrows():
                context += f"- {pos['Ticker']}: {pos['Market_Value']:,.0f} MAD ({pos['Unrealized_PnL_Pct']:+.1f}%)\n"

        context += f"""
        HIGH CORRELATIONS: {len(correlations)}
        TAX LOSS OPPORTUNITIES: {len([p for p in portfolio_df.itertuples() if p.Unrealized_PnL < 0])}
        TAX RATES: {self.tax_rate_gains * 100:.0f}% capital gains, {self.tax_rate_dividends * 100:.0f}% dividends

        Based on this analysis, provide specific recommendations for:
        1. Immediate actions (next 24-48 hours)
        2. Sector diversification to reduce concentration risk
        3. Tax-loss harvesting opportunities under Moroccan capital gains rules
        4. General positioning appropriate for the {regime} regime

        Be specific with ticker symbols, amounts (in MAD), and rationale.
        """

        return context

    def _generate_fallback_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate detailed fallback recommendations when AI API is unavailable."""

        portfolio_value = analysis_results.get('portfolio_value', 0)
        regime = analysis_results.get('market_regime', 'Unknown')
        concentration = analysis_results.get('sector_concentration_analysis', {})
        correlations = analysis_results.get('correlation_alerts', [])
        portfolio_df = analysis_results.get('portfolio_df', pd.DataFrame())

        recommendations = f"""
Based on quantitative analysis, here are my specific recommendations:

**IMMEDIATE ACTIONS (Next 48 Hours):**
"""

        # Priority 1: Sector concentration risk
        if concentration.get('risk_level') in ['High', 'Critical']:
            recommendations += f"""
1. **Sector Concentration Risk**:
   - Largest sector exposure: {concentration.get('top_sector', 'N/A')} at {concentration.get('concentration_percentage', 0):.1f}%
   - Action: Trim or diversify positions in:
"""
            for pos in concentration.get('high_concentration_positions', [])[:3]:
                sell_amount = pos['market_value'] * 0.3
                recommendations += f"     * {pos['Ticker']}: Consider trimming ~{sell_amount:,.0f} MAD worth\n"

        # Priority 2: Tax Loss Harvesting
        loss_positions = portfolio_df[portfolio_df['Unrealized_PnL'] < -1000] if not portfolio_df.empty else pd.DataFrame()
        if len(loss_positions) > 0:
            recommendations += f"""
2. **Tax Loss Harvesting Opportunities** (at a {self.tax_rate_gains * 100:.0f}% capital gains rate):
"""
            for _, pos in loss_positions.head(3).iterrows():
                tax_value = abs(pos['Unrealized_PnL']) * self.tax_rate_gains
                recommendations += (
                    f"   * {pos['Ticker']}: Realize {abs(pos['Unrealized_PnL']):,.0f} MAD loss "
                    f"(~{tax_value:,.0f} MAD potential tax saving)\n"
                )

        recommendations += f"""
**MARKET REGIME GUIDANCE ({regime}):**
"""
        if regime == "Bull Complacent":
            recommendations += """
- Markets are calm but complacency is dangerous
- This is an ideal time for rebalancing and harvesting tax losses while volatility is low
- Consider trimming positions that have grown into an outsized sector weight
"""
        elif regime == "Crisis / Bear":
            recommendations += """
- Maintain discipline, don't panic sell
- Consider tax loss harvesting in underperforming equities
- Favor higher-quality, defensive names within your sector mix
"""
        else:
            recommendations += """
- Market is in transition, stay nimble
- Focus on sector diversification rather than chasing recent performance
- Reassess position sizing across your largest holdings
"""

        return recommendations

    def generate_giuseppe_prompt(self, analysis_results: Dict[str, Any]) -> str:
        """
        Format analysis results into a professional memo from Giuseppe Paleologo.

        Args:
            analysis_results: Complete analysis results dictionary.

        Returns:
            Formatted memo string.
        """
        current_date = datetime.now().strftime("%Y-%m-%d")

        portfolio_value = analysis_results.get('portfolio_value', 0)
        regime = analysis_results.get('market_regime', 'Unknown')
        metrics = analysis_results.get('portfolio_metrics', {})
        sector_allocation = analysis_results.get('sector_allocation', {})
        correlations = analysis_results.get('correlation_alerts', [])
        tax_loss = analysis_results.get('tax_loss_harvesting', [])
        concentration = analysis_results.get('sector_concentration_analysis', {})

        ai_recommendations = analysis_results.get('ai_recommendations', '')

        memo = f"""**Portfolio Analysis & Strategic Recommendations**

**TO:** Portfolio Owner
**FROM:** G. Paleologo
**DATE:** {current_date}
**SUBJECT:** Casablanca Stock Exchange Portfolio Review

**1. Executive Summary**

I have completed a quantitative review of your portfolio, valued at **{portfolio_value:,.0f} MAD**. My analysis indicates the current market regime is **{regime}**.

Your portfolio currently has a Sharpe Ratio of **{metrics.get('sharpe_ratio', 0):.3f}** and has experienced a maximum drawdown of **{metrics.get('max_drawdown', 0):.1f}%**. """

        primary_risks = []

        if len(correlations) > 5:
            primary_risks.append("high asset correlation")

        if concentration.get('risk_level') in ['High', 'Critical']:
            primary_risks.append(
                f"concentrated exposure to {concentration.get('top_sector', 'a single sector')} "
                f"({concentration.get('concentration_percentage', 0):.1f}%)"
            )

        if primary_risks:
            memo += f"The primary risks I've identified are **{', '.join(primary_risks)}**."

        memo += """

This memo outlines a prioritized action plan to improve diversification and risk-adjusted returns.

**2. Sector Allocation**

Your current allocation by Casablanca market sector:
"""

        for sector, pct in sector_allocation.get('allocations', {}).items():
            memo += f"- {sector}: {pct:.1f}%\n"

        memo += """

**3. AI-Generated Strategic Recommendations**

"""
        if ai_recommendations:
            memo += ai_recommendations
        else:
            memo += """
**IMMEDIATE ACTIONS (Tax Alpha & Risk Mitigation):**

* **Tax-Loss Harvesting:** Your portfolio has the following opportunities to realize losses, which can provide meaningful tax alpha.
"""
            for recommendation in tax_loss[:3]:
                memo += f"\n   * {recommendation}"

            memo += """

* **Correlation Hotspots:** Your portfolio has assets that are moving in lockstep, which is a hidden risk. Reduce exposure to one of each pair.
"""
            for alert in correlations[:3]:
                pair = alert['pair']
                corr = alert['correlation']
                risk_type = alert.get('risk_type', 'General Correlation')

                if 'Same-Sector' in risk_type:
                    memo += f"\n   * ⚠️ {pair[0]} and {pair[1]} (correlation: {corr}, **{risk_type}**)"
                else:
                    memo += f"\n   * {pair[0]} and {pair[1]} (correlation: {corr})"

        memo += f"""

**4. Market Environment & Behavioral Coaching**

We are in a **{regime}** regime.
"""

        if regime == "Bull Complacent":
            memo += """
Markets are calm and trending upward, but this is precisely when discipline matters most. This is the time to rebalance into more defensive names and harvest tax losses while volatility is low."""

        elif regime == "Crisis / Bear":
            memo += """
Fear dominates the market. Resist the urge to abandon quality positions at the bottom. Use this period to review your sector exposure and tax-loss harvest where appropriate."""

        else:
            memo += """
Markets are at an inflection point. Balance across uncorrelated sectors is most valuable now. Neither fully risk-on nor risk-off is appropriate. Maintain diversification and let it work."""

        memo += f"""

**5. Sector Concentration Risk**
"""

        if concentration.get('risk_level') in ['High', 'Critical']:
            memo += f"""
* **Concentration Risk Level: {concentration.get('risk_level')}** ({concentration.get('concentration_percentage', 0):.1f}% in {concentration.get('top_sector', 'N/A')})
   * {concentration.get('recommendation')}
   * Largest positions in this sector:"""

            for position in concentration.get('high_concentration_positions', [])[:3]:
                memo += f"\n     - {position['Ticker']}: {position['weight']:.1f}% of portfolio"
        else:
            memo += f"\n* Sector exposure is within acceptable limits (largest sector: {concentration.get('top_sector', 'N/A')} at {concentration.get('concentration_percentage', 0):.1f}%)"

        backtest = analysis_results.get('backtest_results', {})
        if backtest and 'performance' in backtest:
            perf = backtest['performance']
            memo += f"""

**6. Historical Backtest Results**
* Period: {backtest['backtest_period']['start']} to {backtest['backtest_period']['end']}
* Total Return: {perf['total_return']:.1f}% ({self.benchmark_ticker}: {perf['benchmark_return']:.1f}%)
* Annualized Return: {perf['annualized_return']:.1f}%
* Max Drawdown: {perf['max_drawdown']:.1f}%
* Sharpe Ratio: {perf['sharpe_ratio']:.3f}
"""

        memo += f"""

**END MEMO**

Risk management is an active, not a passive process.

**Performance Metrics Summary:**
- Annual Return: {metrics.get('annual_return', 0):.1f}%
- Volatility: {metrics.get('annual_volatility', 0):.1f}%
- Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}
- Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}
- Tail Ratio: {metrics.get('tail_ratio', 0):.3f}
- Omega Ratio: {metrics.get('omega_ratio', 0):.3f}
"""

        return memo


def run_full_analysis(portfolio_path: str) -> Dict[str, Any]:
    """
    Main wrapper function that runs complete portfolio analysis.

    Args:
        portfolio_path: Path to portfolio CSV file.

    Returns:
        Dictionary containing all analysis results.
    """
    logger.info("Starting full portfolio analysis...")

    try:
        analyzer = PortfolioAnalyzer()

        logger.info("Fetching portfolio data...")
        portfolio_df, historical_data = fetch_portfolio_data(portfolio_path)
        portfolio_value = abs(portfolio_df['Market_Value']).sum()

        logger.info("Detecting market regime...")
        market_regime = analyzer.detect_market_regime(historical_data)

        logger.info("Calculating portfolio metrics...")
        portfolio_metrics = analyzer.calculate_portfolio_metrics(portfolio_df, historical_data)

        logger.info("Analyzing sector allocation...")
        sector_allocation = analyzer.run_sector_allocation(portfolio_df)

        logger.info("Running correlation analysis...")
        correlation_alerts, correlation_matrix = analyzer.run_correlation_analysis(historical_data, portfolio_df)

        logger.info("Calculating conviction scores...")
        conviction_analysis = analyzer.calculate_conviction_scores(portfolio_df, historical_data)

        logger.info("Identifying tax loss harvesting opportunities...")
        tax_loss_harvesting = analyzer.identify_tax_loss_harvesting(portfolio_df)

        logger.info("Analyzing sector concentration risk...")
        sector_concentration_analysis = analyzer.analyze_sector_concentration_risk(portfolio_df)

        logger.info("Running portfolio backtest...")
        backtest_results = analyzer.run_backtest(portfolio_df, historical_data)

        analysis_results = {
            'portfolio_value': portfolio_value,
            'market_regime': market_regime,
            'portfolio_metrics': portfolio_metrics,
            'sector_allocation': sector_allocation,
            'correlation_alerts': correlation_alerts,
            'correlation_matrix': correlation_matrix,
            'conviction_analysis': conviction_analysis,
            'tax_loss_harvesting': tax_loss_harvesting,
            'sector_concentration_analysis': sector_concentration_analysis,
            'backtest_results': backtest_results,
            'portfolio_df': portfolio_df,
            'historical_data': historical_data
        }

        logger.info("Generating AI-powered recommendations...")
        ai_recommendations = analyzer.generate_ai_recommendations(analysis_results)
        analysis_results['ai_recommendations'] = ai_recommendations

        logger.info("Generating investment memo...")
        giuseppe_memo = analyzer.generate_giuseppe_prompt(analysis_results)
        analysis_results['giuseppe_memo'] = giuseppe_memo

        logger.info("Creating portfolio visualizations...")
        graphs = analyzer.create_portfolio_graphs(analysis_results)
        analysis_results['graphs'] = graphs

        logger.info("Analysis complete!")

        return analysis_results

    except Exception as e:
        logger.error(f"Error in full analysis: {str(e)}")
        return {
            'error': str(e),
            'portfolio_value': 0,
            'market_regime': 'Unknown',
            'giuseppe_memo': 'Analysis failed. Please check your portfolio data and try again.'
        }


# Testing and example usage
if __name__ == "__main__":
    results = run_full_analysis('sample_portfolio.csv')

    logger.info("\n" + "="*80)
    logger.info(results['giuseppe_memo'])
    logger.info("="*80 + "\n")

    logger.info("\nDetailed Metrics:")
    logger.info(f"Portfolio Value: {results['portfolio_value']:,.0f} MAD")
    logger.info(f"Market Regime: {results['market_regime']}")
    logger.info(f"Sharpe Ratio: {results['portfolio_metrics']['sharpe_ratio']}")
    logger.info(f"Max Drawdown: {results['portfolio_metrics']['max_drawdown']}%")
    logger.info(f"Sortino Ratio: {results['portfolio_metrics']['sortino_ratio']}")
    logger.info(f"Calmar Ratio: {results['portfolio_metrics']['calmar_ratio']}")
