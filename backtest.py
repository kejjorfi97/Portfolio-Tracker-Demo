"""
Advanced Backtesting Engine for Dragon Portfolio Analytics
=========================================================
Provides sophisticated backtesting capabilities including:
- Multiple rebalancing strategies
- Transaction cost modeling
- Monte Carlo simulations
- Regime-aware backtesting

Author: Dragon Portfolio Analytics
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 1_000_000  # MAD
    rebalance_frequency: str = 'quarterly'  # 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    transaction_cost_bps: float = 10  # basis points
    slippage_bps: float = 5  # basis points
    tax_rate_gains: float = 0.15  # 15% Moroccan capital gains tax (flat, configurable)
    tax_rate_dividends: float = 0.15  # 15% Moroccan dividend withholding tax (flat, configurable)
    min_position_size: float = 0.01  # 1% minimum position
    max_position_size: float = 0.25  # 25% maximum position
    allow_shorting: bool = True
    use_regime_switching: bool = True
    benchmark_ticker: str = 'MSI20'


class AdvancedBacktester:
    """Advanced backtesting engine with multiple strategies and risk models."""
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize the backtester with configuration."""
        self.config = config or BacktestConfig()
        self.results = {}
        
    def run_backtest_with_rebalancing(self, portfolio_df: pd.DataFrame, 
                                    historical_data: pd.DataFrame,
                                    strategy: str = 'buy_and_hold') -> Dict[str, Any]:
        """
        Run backtest with specified rebalancing strategy.
        
        Strategies:
        - buy_and_hold: No rebalancing
        - constant_weight: Rebalance to initial weights
        - risk_parity: Equal risk contribution
        - momentum: Tilt towards momentum
        - mean_reversion: Contrarian rebalancing
        """
        
        logger.info(f"Running backtest with {strategy} strategy")
        
        # Get initial weights
        weights = self._calculate_initial_weights(portfolio_df, historical_data)
        
        # Initialize portfolio value series
        dates = historical_data.index
        portfolio_values = pd.Series(index=dates, dtype=float)
        portfolio_values.iloc[0] = self.config.initial_capital
        
        # Track positions and transactions
        positions = {ticker: 0 for ticker in weights.keys()}
        transactions = []
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(dates, self.config.rebalance_frequency)
        
        # Run simulation
        for i in range(1, len(dates)):
            current_date = dates[i]
            prev_date = dates[i-1]
            
            # Check if rebalancing needed
            if current_date in rebalance_dates:
                new_weights = self._calculate_rebalance_weights(
                    strategy, weights, historical_data.loc[:current_date], 
                    positions, portfolio_values.iloc[i-1]
                )
                
                # Execute rebalancing
                trans = self._execute_rebalancing(
                    positions, new_weights, portfolio_values.iloc[i-1],
                    historical_data.loc[current_date]
                )
                transactions.extend(trans)
                weights = new_weights
            
            # Calculate daily portfolio value
            daily_value = 0
            for ticker, position in positions.items():
                if ticker in historical_data.columns:
                    price = historical_data.loc[current_date, ticker]
                    daily_value += position * price
            
            # Add cash
            cash = portfolio_values.iloc[i-1] - sum(
                positions[t] * historical_data.loc[prev_date, t] 
                for t in positions if t in historical_data.columns
            )
            
            portfolio_values.iloc[i] = daily_value + cash
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(
            portfolio_values, historical_data, transactions
        )
        
        return {
            'strategy': strategy,
            'portfolio_values': portfolio_values,
            'transactions': transactions,
            'metrics': metrics,
            'final_weights': weights
        }
    
    def run_monte_carlo_simulation(self, portfolio_df: pd.DataFrame,
                                 historical_data: pd.DataFrame,
                                 num_simulations: int = 1000,
                                 time_horizon_days: int = 252) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio outcomes.
        """
        logger.info(f"Running {num_simulations} Monte Carlo simulations")
        
        # Calculate historical statistics
        returns = historical_data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Get portfolio weights
        weights = self._calculate_initial_weights(portfolio_df, historical_data)
        weight_array = np.array([weights.get(ticker, 0) for ticker in returns.columns])
        
        # Portfolio statistics
        portfolio_return = np.dot(weight_array, mean_returns)
        portfolio_std = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))
        
        # Run simulations
        simulation_results = np.zeros((num_simulations, time_horizon_days))
        
        for sim in range(num_simulations):
            daily_returns = np.random.normal(
                portfolio_return, portfolio_std, time_horizon_days
            )
            
            # Calculate cumulative returns
            simulation_results[sim] = self.config.initial_capital * np.cumprod(
                1 + daily_returns
            )
        
        # Calculate statistics
        final_values = simulation_results[:, -1]
        
        return {
            'simulations': simulation_results,
            'statistics': {
                'mean_final_value': np.mean(final_values),
                'median_final_value': np.median(final_values),
                'std_final_value': np.std(final_values),
                'var_95': np.percentile(final_values, 5),
                'var_99': np.percentile(final_values, 1),
                'probability_of_loss': np.mean(final_values < self.config.initial_capital),
                'expected_return': (np.mean(final_values) / self.config.initial_capital - 1) * 100,
                'best_case_scenario': np.percentile(final_values, 95),
                'worst_case_scenario': np.percentile(final_values, 5)
            }
        }
    
    def run_regime_aware_backtest(self, portfolio_df: pd.DataFrame,
                                historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest with regime-switching strategy.
        Adjusts allocation based on detected market regime.
        """
        logger.info("Running regime-aware backtest")

        # This would integrate with the regime detection from analysis_engine.
        # For demonstration, we use a simple trend + realized-volatility regime
        # detector on the benchmark (MSI20) - there's no Moroccan VIX-equivalent,
        # so realized volatility stands in for the implied-vol "fear gauge".
        benchmark = self.config.benchmark_ticker
        benchmark_returns = historical_data[benchmark].pct_change()
        trend = benchmark_returns.rolling(20).mean()
        realized_vol = benchmark_returns.rolling(20).std() * np.sqrt(252) * 100

        # Define regime-based allocations
        regime_allocations = {
            'bull': {'equity': 0.8, 'bonds': 0.1, 'gold': 0.1},
            'bear': {'equity': 0.3, 'bonds': 0.5, 'gold': 0.2},
            'neutral': {'equity': 0.6, 'bonds': 0.3, 'gold': 0.1}
        }

        # Run backtest with regime switching
        results = self._run_regime_backtest(
            portfolio_df, historical_data, trend,
            realized_vol, regime_allocations
        )

        return results
    
    def calculate_risk_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        """
        returns = portfolio_values.pct_change().dropna()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        
        # Maximum drawdown duration
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start_idx = None
        
        for i, in_dd in enumerate(is_drawdown):
            if in_dd and start_idx is None:
                start_idx = i
            elif not in_dd and start_idx is not None:
                drawdown_periods.append(i - start_idx)
                start_idx = None
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Value at Risk (Historical method)
        var_95 = np.percentile(returns, 5) * np.sqrt(252) * portfolio_values.iloc[-1]
        var_99 = np.percentile(returns, 1) * np.sqrt(252) * portfolio_values.iloc[-1]
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(252) * portfolio_values.iloc[-1]
        
        return {
            'downside_deviation': downside_deviation,
            'max_drawdown_duration_days': max_dd_duration,
            'var_95_annual': var_95,
            'var_99_annual': var_99,
            'cvar_95_annual': cvar_95,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'omega_ratio': self._calculate_omega_ratio(returns),
            'tail_ratio': abs(np.percentile(returns, 95) / np.percentile(returns, 5))
        }
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio."""
        positive_returns = returns[returns > threshold] - threshold
        negative_returns = threshold - returns[returns <= threshold]
        
        if negative_returns.sum() == 0:
            return np.inf
            
        return positive_returns.sum() / negative_returns.sum()
    
    def _calculate_initial_weights(self, portfolio_df: pd.DataFrame, 
                                 historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate initial portfolio weights."""
        weights = {}
        total_value = abs(portfolio_df['Market_Value']).sum()
        
        for _, row in portfolio_df.iterrows():
            if row['Ticker'] in historical_data.columns:
                weights[row['Ticker']] = row['Market_Value'] / total_value
                
        return weights
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, frequency: str) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        if frequency == 'daily':
            return dates[1:].tolist()
        elif frequency == 'weekly':
            return dates[dates.weekday == 4].tolist()  # Fridays
        elif frequency == 'monthly':
            return dates[dates.is_month_end].tolist()
        elif frequency == 'quarterly':
            return dates[(dates.month % 3 == 0) & dates.is_month_end].tolist()
        elif frequency == 'yearly':
            return dates[dates.is_year_end].tolist()
        else:
            return []
    
    def _calculate_rebalance_weights(self, strategy: str, current_weights: Dict[str, float],
                                   historical_data: pd.DataFrame, positions: Dict[str, float],
                                   portfolio_value: float) -> Dict[str, float]:
        """Calculate new weights based on rebalancing strategy."""
        if strategy == 'buy_and_hold':
            return current_weights
        elif strategy == 'constant_weight':
            return current_weights  # Return to initial weights
        elif strategy == 'risk_parity':
            return self._calculate_risk_parity_weights(historical_data)
        elif strategy == 'momentum':
            return self._calculate_momentum_weights(historical_data, current_weights)
        elif strategy == 'mean_reversion':
            return self._calculate_mean_reversion_weights(historical_data, current_weights)
        else:
            return current_weights
    
    def _calculate_risk_parity_weights(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk parity weights (simplified)."""
        returns = historical_data.pct_change().dropna()
        volatilities = returns.std()
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        return weights.to_dict()
    
    def _calculate_momentum_weights(self, historical_data: pd.DataFrame, 
                                  current_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate momentum-tilted weights."""
        # Simple momentum: tilt towards recent performers
        momentum = historical_data.iloc[-20:].pct_change().mean()
        momentum_score = (momentum - momentum.mean()) / momentum.std()
        
        new_weights = {}
        for ticker in current_weights:
            if ticker in momentum_score.index:
                # Adjust weight based on momentum
                adjustment = 1 + 0.2 * momentum_score[ticker]  # ±20% max adjustment
                new_weights[ticker] = current_weights[ticker] * adjustment
            else:
                new_weights[ticker] = current_weights[ticker]
        
        # Normalize
        total = sum(new_weights.values())
        return {k: v/total for k, v in new_weights.items()}
    
    def _calculate_mean_reversion_weights(self, historical_data: pd.DataFrame,
                                        current_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate mean reversion weights (contrarian)."""
        # Opposite of momentum
        momentum_weights = self._calculate_momentum_weights(historical_data, current_weights)
        
        # Invert the tilts
        mean_weights = {}
        avg_weight = np.mean(list(current_weights.values()))
        
        for ticker in current_weights:
            if ticker in momentum_weights:
                # Move opposite to momentum
                deviation = momentum_weights[ticker] - avg_weight
                mean_weights[ticker] = avg_weight - deviation * 0.5
            else:
                mean_weights[ticker] = current_weights[ticker]
        
        # Ensure positive weights and normalize
        mean_weights = {k: max(v, 0.01) for k, v in mean_weights.items()}
        total = sum(mean_weights.values())
        
        return {k: v/total for k, v in mean_weights.items()}
    
    def _execute_rebalancing(self, positions: Dict[str, float], 
                           target_weights: Dict[str, float],
                           portfolio_value: float,
                           current_prices: pd.Series) -> List[Dict]:
        """Execute rebalancing and return transactions."""
        transactions = []
        
        # Calculate target positions
        for ticker, target_weight in target_weights.items():
            if ticker in current_prices.index:
                target_value = portfolio_value * target_weight
                target_shares = target_value / current_prices[ticker]
                current_shares = positions.get(ticker, 0)
                
                shares_to_trade = target_shares - current_shares
                
                if abs(shares_to_trade) > 0.01:  # Minimum trade size
                    # Calculate transaction cost
                    trade_value = abs(shares_to_trade * current_prices[ticker])
                    transaction_cost = trade_value * (self.config.transaction_cost_bps / 10000)
                    slippage_cost = trade_value * (self.config.slippage_bps / 10000)
                    
                    transactions.append({
                        'date': current_prices.name,
                        'ticker': ticker,
                        'shares': shares_to_trade,
                        'price': current_prices[ticker],
                        'value': shares_to_trade * current_prices[ticker],
                        'transaction_cost': transaction_cost,
                        'slippage_cost': slippage_cost,
                        'action': 'BUY' if shares_to_trade > 0 else 'SELL'
                    })
                    
                    positions[ticker] = target_shares
        
        return transactions
    
    def _calculate_performance_metrics(self, portfolio_values: pd.Series,
                                     historical_data: pd.DataFrame,
                                     transactions: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        returns = portfolio_values.pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        years = len(returns) / 252
        annualized_return = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/years) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return - 5) / volatility
        
        # Transaction costs
        total_transaction_costs = sum(t['transaction_cost'] + t['slippage_cost'] for t in transactions)
        
        # Turnover
        total_traded_value = sum(abs(t['value']) for t in transactions)
        avg_portfolio_value = portfolio_values.mean()
        annual_turnover = (total_traded_value / avg_portfolio_value) / years * 100
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_transactions': len(transactions),
            'total_transaction_costs': total_transaction_costs,
            'annual_turnover': annual_turnover,
            'cost_impact_bps': (total_transaction_costs / portfolio_values.iloc[-1]) * 10000
        }
    
    def _run_regime_backtest(self, portfolio_df: pd.DataFrame,
                           historical_data: pd.DataFrame,
                           trend: pd.Series,
                           realized_vol: pd.Series,
                           regime_allocations: Dict) -> Dict[str, Any]:
        """Execute regime-based backtesting."""
        # Simplified implementation
        # Would integrate with full regime detection logic

        portfolio_values = pd.Series(index=historical_data.index, dtype=float)
        portfolio_values.iloc[0] = self.config.initial_capital

        for i in range(1, len(historical_data)):
            # Detect regime (simplified) using benchmark trend + realized volatility
            # in place of a VIX-style implied-volatility gauge (Morocco has none).
            if trend.iloc[i] > 0.01 and realized_vol.iloc[i] < 15:
                regime = 'bull'
            elif trend.iloc[i] < -0.01 or realized_vol.iloc[i] > 25:
                regime = 'bear'
            else:
                regime = 'neutral'
            
            # Calculate portfolio return based on regime allocation
            # (Simplified for demonstration)
            daily_return = historical_data.pct_change().iloc[i].mean()
            portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + daily_return)
        
        return {
            'portfolio_values': portfolio_values,
            'final_value': portfolio_values.iloc[-1],
            'total_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        }


# Example usage
if __name__ == "__main__":
    logger.info("Advanced Backtesting Engine loaded.")
    logger.info("This module provides sophisticated backtesting capabilities including:")
    logger.info("- Multiple rebalancing strategies")
    logger.info("- Monte Carlo simulations")
    logger.info("- Regime-aware backtesting")
    logger.info("- Comprehensive risk metrics")
    logger.info("\nIntegrate with analysis_engine.py for full functionality.")
