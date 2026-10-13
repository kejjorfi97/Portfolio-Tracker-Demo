"""
Test Script for Casablanca Stock Exchange Portfolio Analytics
================================================================
Demonstrates full functionality including sector concentration risk
detection and backtesting capabilities.

Author: Dragon Portfolio Analytics
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from analysis_engine import run_full_analysis, PortfolioAnalyzer
import plotly.graph_objects as go
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def create_test_portfolio():
    """
    Create a diverse test portfolio of Casablanca Stock Exchange listings,
    spanning several sectors plus a cash position and a short position.
    """
    print("📊 Creating test portfolio with CSE assets...")

    portfolio_data = {
        'Ticker': [
            # Banks
            'ATW',    # Attijariwafa Bank
            'BCP',    # Banque Centrale Populaire
            'CIH',    # CIH Bank
            'BOA',    # Bank of Africa

            # Telecommunications
            'IAM',    # Maroc Telecom

            # Building Materials
            'LHM',    # LafargeHolcim Maroc
            'CMA',    # Ciments du Maroc

            # Insurance
            'WAA',    # Wafa Assurance
            'SAH',    # Saham Assurance

            # Real Estate
            'ADH',    # Douja Prom Addoha
            'ARD',    # Aradei Capital

            # Mining
            'MNG',    # Managem

            # Food & Beverage
            'CSR',    # Cosumar
            'LES',    # Lesieur Cristal

            # Distribution
            'LBV',    # Label Vie

            # Utilities
            'TQM',    # Taqa Morocco

            # Cash
            'CASH',
        ],
        'Quantity': [
            100,      # ATW
            150,      # BCP
            120,      # CIH
            80,       # BOA
            200,      # IAM
            50,       # LHM
            80,       # CMA
            80,       # WAA
            60,       # SAH
            300,      # ADH
            -100,     # ARD (short)
            60,       # MNG
            90,       # CSR
            70,       # LES
            40,       # LBV
            150,      # TQM
            50000,    # CASH
        ],
        'Category': [
            'Banks', 'Banks', 'Banks', 'Banks',
            'Telecommunications',
            'Building Materials', 'Building Materials',
            'Insurance', 'Insurance',
            'Real Estate', 'Real Estate',
            'Mining',
            'Food & Beverage', 'Food & Beverage',
            'Distribution',
            'Utilities',
            'Cash',
        ],
        'Cost_Basis_per_Share': [
            450.00, 260.00, 280.00, 180.00,
            95.00,
            2000.00, 1500.00,
            4200.00, 600.00,
            30.00, 450.00,
            2200.00,
            260.00, 220.00,
            2100.00,
            110.00,
            1.00,
        ]
    }

    portfolio_df = pd.DataFrame(portfolio_data)

    # Save to CSV
    portfolio_df.to_csv('test_portfolio.csv', index=False)
    print(f"✅ Created CSE test portfolio with {len(portfolio_df)} positions")
    print(f"   - Long positions: {(portfolio_df['Quantity'] > 0).sum()}")
    print(f"   - Short positions: {(portfolio_df['Quantity'] < 0).sum()}")
    print("   - Sectors: Banks, Telecom, Building Materials, Insurance, Real Estate, Mining, Food & Beverage, Distribution, Utilities")

    return portfolio_df


def display_portfolio_summary(portfolio_df, analysis_results):
    """Display a summary of the portfolio analysis."""
    print("\n" + "="*80)
    print("📈 PORTFOLIO ANALYSIS SUMMARY")
    print("="*80)

    # Basic metrics
    portfolio_value = analysis_results['portfolio_value']
    metrics = analysis_results['portfolio_metrics']
    regime = analysis_results['market_regime']

    print(f"\n💰 Portfolio Value: {portfolio_value:,.2f} MAD")
    print(f"🌡️  Market Regime: {regime}")

    print("\n📊 Risk-Adjusted Performance Metrics:")
    print(f"   - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"   - Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"   - Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"   - Max Drawdown: {metrics['max_drawdown']:.1f}%")
    print(f"   - Annual Return: {metrics['annual_return']:.1f}%")
    print(f"   - Annual Volatility: {metrics['annual_volatility']:.1f}%")

    # Sector Concentration Analysis
    concentration = analysis_results['sector_concentration_analysis']
    print(f"\n🎯 Sector Concentration Risk:")
    print(f"   - Largest Sector: {concentration.get('top_sector')}")
    print(f"   - Concentration: {concentration.get('concentration_percentage', 0):.1f}%")
    print(f"   - Risk Level: {concentration['risk_level']}")

    if concentration.get('high_concentration_positions'):
        print("   - Positions in largest sector:")
        for pos in concentration['high_concentration_positions'][:5]:
            print(f"     • {pos['Ticker']}: {pos['weight']:.1f}%")

    # Sector Allocation
    sector_allocation = analysis_results['sector_allocation']
    print(f"\n🏢 Portfolio Allocation by Sector:")

    for sector, pct in sector_allocation.get('allocations', {}).items():
        print(f"   - {sector}: {pct:.1f}%")

    # Correlation Alerts
    correlations = analysis_results['correlation_alerts']
    if correlations:
        print(f"\n⚠️  High Correlation Alerts (>{0.75}):")
        for alert in correlations[:5]:
            pair = alert['pair']
            corr = alert['correlation']
            risk_type = alert.get('risk_type', 'General')
            print(f"   - {pair[0]} ↔ {pair[1]}: {corr:.3f} ({risk_type})")

    # Tax Loss Harvesting
    tax_losses = analysis_results['tax_loss_harvesting']
    if tax_losses and tax_losses[0] != "No tax loss harvesting opportunities currently available.":
        print(f"\n💸 Tax Loss Harvesting Opportunities:")
        for i, rec in enumerate(tax_losses[:3], 1):
            print(f"   {i}. {rec[:100]}...")  # Truncate for readability

    # Backtest Results
    backtest = analysis_results.get('backtest_results', {})
    if backtest and 'performance' in backtest:
        perf = backtest['performance']
        period = backtest['backtest_period']
        print(f"\n📈 Backtest Results ({period['start']} to {period['end']}):")
        print(f"   - Total Return: {perf['total_return']:.1f}%")
        print(f"   - MSI20 Return: {perf['benchmark_return']:.1f}%")
        print(f"   - Excess Return vs MSI20: {perf['excess_return_vs_benchmark']:.1f}%")
        print(f"   - Backtest Sharpe: {perf['sharpe_ratio']:.3f}")


def save_analysis_results(analysis_results):
    """Save analysis results to files."""
    print("\n💾 Saving analysis results...")

    # Save Giuseppe memo
    with open('giuseppe_memo.txt', 'w', encoding='utf-8') as f:
        f.write(analysis_results['giuseppe_memo'])
    print("   ✅ Saved investment memo to 'giuseppe_memo.txt'")

    # Save enhanced portfolio
    enhanced_portfolio = analysis_results['portfolio_df']
    enhanced_portfolio.to_csv('enhanced_portfolio.csv', index=False)
    print("   ✅ Saved enhanced portfolio to 'enhanced_portfolio.csv'")

    # Save metrics summary
    metrics_summary = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'portfolio_value': analysis_results['portfolio_value'],
        'market_regime': analysis_results['market_regime'],
        'metrics': analysis_results['portfolio_metrics'],
        'sector_concentration': analysis_results['sector_concentration_analysis'],
        'sector_allocation': analysis_results['sector_allocation'],
    }

    with open('analysis_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    print("   ✅ Saved analysis summary to 'analysis_summary.json'")


def create_backtest_visualization(backtest_results):
    """Create a simple backtest performance chart."""
    if not backtest_results or 'portfolio_value_series' not in backtest_results:
        return

    print("\n📊 Creating backtest visualization...")

    portfolio_values = backtest_results['portfolio_value_series']
    drawdown = backtest_results['drawdown_series']

    # Create subplot figure
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Portfolio Value Over Time', 'Drawdown %')
    )

    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            name='Portfolio Value',
            line=dict(color='#1976D2', width=2)
        ),
        row=1, col=1
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#D32F2F', width=1),
            fillcolor='rgba(211, 47, 47, 0.3)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title='Portfolio Backtest Performance',
        height=700,
        showlegend=False,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value (MAD)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    # Save to HTML
    fig.write_html('backtest_chart.html')
    print("   ✅ Saved backtest chart to 'backtest_chart.html'")


def main():
    """Main test execution."""
    print("\n🇲🇦 CSE PORTFOLIO ANALYTICS TEST SUITE")
    print("=" * 50)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Check for AI API keys
    print("\n🤖 AI Recommendation Setup:")
    if os.environ.get('OPENAI_API_KEY'):
        print("   ✅ OpenAI API key detected - AI recommendations enabled")
    elif os.environ.get('ANTHROPIC_API_KEY'):
        print("   ✅ Anthropic API key detected - AI recommendations enabled")
    else:
        print("   ⚠️  No AI API key detected - Using enhanced fallback recommendations")
        print("   💡 To enable AI recommendations, set environment variable:")
        print("      export OPENAI_API_KEY='your-api-key'")
        print("      or")
        print("      export ANTHROPIC_API_KEY='your-api-key'")

    if not os.environ.get('MONGO_URI'):
        print("\n⚠️  MONGO_URI not set - falling back to mongodb://localhost:27017/")

    # Step 1: Create test portfolio
    portfolio_df = create_test_portfolio()

    # Step 2: Run full analysis
    print("\n🚀 Running comprehensive portfolio analysis...")
    print("   This may take a few seconds for data fetching...")

    try:
        analysis_results = run_full_analysis('test_portfolio.csv')

        # Step 3: Display summary
        display_portfolio_summary(portfolio_df, analysis_results)

        # Step 4: Save results
        save_analysis_results(analysis_results)

        # Step 5: Create visualizations
        if 'backtest_results' in analysis_results:
            create_backtest_visualization(analysis_results['backtest_results'])

        # Step 6: Display Giuseppe Memo
        print("\n" + "="*80)
        print("📝 GIUSEPPE PALEOLOGO INVESTMENT MEMO")
        print("="*80)
        print(analysis_results['giuseppe_memo'])
        print("="*80)

        # Step 7: Sector Concentration Warnings
        concentration = analysis_results['sector_concentration_analysis']
        if concentration['risk_level'] in ['High', 'Critical']:
            print("\n⚠️  " + "="*76)
            print("⚠️  SECTOR CONCENTRATION WARNING")
            print("⚠️  " + "="*76)
            print(f"⚠️  {concentration.get('concentration_percentage', 0):.1f}% of your portfolio is in {concentration.get('top_sector')}.")
            print(f"⚠️  Risk Level: {concentration['risk_level']}")
            print(f"⚠️  {concentration['recommendation']}")
            print("⚠️  " + "="*76)

        print("\n✅ Test completed successfully!")
        print("\n📁 Output files created:")
        print("   - giuseppe_memo.txt (Investment memo)")
        print("   - enhanced_portfolio.csv (Portfolio with analytics)")
        print("   - analysis_summary.json (Metrics summary)")
        print("   - backtest_chart.html (Performance visualization)")

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n🏁 Test suite execution complete!")


if __name__ == "__main__":
    main()
