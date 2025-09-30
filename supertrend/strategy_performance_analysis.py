import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_strategy_data(file_path):
    """Load and prepare strategy data from CSV file"""
    df = pd.read_csv(file_path)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    return df

def calculate_trades(df, leverage=50, transaction_fee=30):
    """Extract individual trades from position data"""
    trades = []
    entry_price = None
    entry_time = None

    for i, row in df.iterrows():
        if row['entry_signal'] and entry_price is None:  # New entry
            entry_price = row['close']
            entry_time = row['start_time']
        elif row['exit_signal'] and entry_price is not None:  # Exit
            exit_price = row['close']
            exit_time = row['start_time']

            # Calculate trade metrics with leverage
            pnl_points = exit_price - entry_price
            pnl_ntd_gross = pnl_points * leverage  # Apply 50x leverage
            pnl_ntd_net = pnl_ntd_gross - (transaction_fee * 2)  # Entry + Exit fees
            pnl_percent = (pnl_ntd_net / (entry_price * leverage)) * 100
            duration = exit_time - entry_time

            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_points': pnl_points,
                'pnl_ntd_gross': pnl_ntd_gross,
                'pnl_ntd_net': pnl_ntd_net,
                'transaction_fees': transaction_fee * 2,
                'pnl_percent': pnl_percent,
                'duration_minutes': duration.total_seconds() / 60,
                'trade_num': len(trades) + 1
            })

            entry_price = None
            entry_time = None

    return pd.DataFrame(trades)

def calculate_performance_metrics(trades_df, initial_capital=100000, position_value=76500):
    """Calculate comprehensive performance metrics"""
    if trades_df.empty:
        return {}

    # Basic metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_ntd_net'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_ntd_net'] < 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    # P&L metrics (in NTD)
    total_pnl_ntd_net = trades_df['pnl_ntd_net'].sum()
    total_pnl_ntd_gross = trades_df['pnl_ntd_gross'].sum()
    total_fees = trades_df['transaction_fees'].sum()
    total_pnl_percent = (total_pnl_ntd_net / initial_capital) * 100
    avg_trade_pnl = trades_df['pnl_ntd_net'].mean()

    # Win/Loss analysis (in NTD)
    winning_trades_df = trades_df[trades_df['pnl_ntd_net'] > 0]
    losing_trades_df = trades_df[trades_df['pnl_ntd_net'] < 0]

    avg_win = winning_trades_df['pnl_ntd_net'].mean() if not winning_trades_df.empty else 0
    avg_loss = losing_trades_df['pnl_ntd_net'].mean() if not losing_trades_df.empty else 0

    total_win_amount = winning_trades_df['pnl_ntd_net'].sum() if not winning_trades_df.empty else 0
    total_loss_amount = abs(losing_trades_df['pnl_ntd_net'].sum()) if not losing_trades_df.empty else 0
    profit_factor = total_win_amount / total_loss_amount if total_loss_amount > 0 else float('inf')

    # Drawdown calculation (in NTD)
    trades_df['cumulative_pnl'] = trades_df['pnl_ntd_net'].cumsum()
    trades_df['running_max'] = trades_df['cumulative_pnl'].expanding().max()
    trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']

    max_drawdown_ntd = trades_df['drawdown'].min()
    max_drawdown_percent = (max_drawdown_ntd / initial_capital) * 100

    # Trading frequency
    if total_trades > 1:
        first_trade = trades_df['entry_time'].min()
        last_trade = trades_df['exit_time'].max()
        total_days = (last_trade - first_trade).days
        trades_per_day = total_trades / total_days if total_days > 0 else 0
    else:
        trades_per_day = 0

    # Risk metrics
    sharpe_ratio = trades_df['pnl_ntd_net'].mean() / trades_df['pnl_ntd_net'].std() if trades_df['pnl_ntd_net'].std() > 0 else 0

    # Duration analysis
    avg_trade_duration = trades_df['duration_minutes'].mean()

    # Return on capital
    roi = (total_pnl_ntd_net / initial_capital) * 100

    # Position size metrics
    max_positions = int(initial_capital / position_value)
    position_utilization = (position_value / initial_capital) * 100

    metrics = {
        'Initial Capital (NTD)': f'{initial_capital:,}',
        'Position Value (NTD)': f'{position_value:,}',
        'Max Positions': max_positions,
        'Capital Utilization (%)': round(position_utilization, 2),
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Total P&L (NTD)': round(total_pnl_ntd_net, 2),
        'Total P&L Gross (NTD)': round(total_pnl_ntd_gross, 2),
        'Total Transaction Fees (NTD)': round(total_fees, 2),
        'ROI (%)': round(roi, 2),
        'Average Trade P&L (NTD)': round(avg_trade_pnl, 2),
        'Average Win (NTD)': round(avg_win, 2),
        'Average Loss (NTD)': round(avg_loss, 2),
        'Profit Factor': round(profit_factor, 2) if profit_factor != float('inf') else 'Infinite',
        'Max Drawdown (NTD)': round(max_drawdown_ntd, 2),
        'Max Drawdown (%)': round(max_drawdown_percent, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Trades per Day': round(trades_per_day, 2),
        'Avg Trade Duration (min)': round(avg_trade_duration, 2)
    }

    return metrics

def plot_performance_charts(trades_df, df):
    """Create performance visualization charts"""
    if trades_df.empty:
        print("No trades to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Cumulative P&L (in NTD)
    trades_df['cumulative_pnl'] = trades_df['pnl_ntd_net'].cumsum()
    ax1.plot(trades_df['trade_num'], trades_df['cumulative_pnl'], 'b-', linewidth=2)
    ax1.set_title('Cumulative P&L Over Time (with 50x Leverage & Fees)')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Cumulative P&L (NTD)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 2. Drawdown (in NTD)
    trades_df['running_max'] = trades_df['cumulative_pnl'].expanding().max()
    trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
    ax2.fill_between(trades_df['trade_num'], trades_df['drawdown'], 0, color='red', alpha=0.3)
    ax2.plot(trades_df['trade_num'], trades_df['drawdown'], 'r-', linewidth=1)
    ax2.set_title('Drawdown Over Time')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Drawdown (NTD)')
    ax2.grid(True, alpha=0.3)

    # 3. Trade P&L Distribution (in NTD)
    ax3.hist(trades_df['pnl_ntd_net'], bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_title('Trade P&L Distribution (Net of Fees)')
    ax3.set_xlabel('P&L per Trade (NTD)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)

    # 4. Monthly Returns (if enough data)
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly_pnl = trades_df.groupby('month')['pnl_ntd_net'].sum()

    if len(monthly_pnl) > 1:
        monthly_pnl.plot(kind='bar', ax=ax4)
        ax4.set_title('Monthly P&L (Net)')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Monthly P&L (NTD)')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis',
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Monthly P&L')

    plt.tight_layout()
    plt.savefig('strategy_performance_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_trade_report(trades_df):
    """Generate detailed trade-by-trade report"""
    if trades_df.empty:
        print("No trades found in the data")
        return

    print("\n" + "="*80)
    print("DETAILED TRADE REPORT")
    print("="*80)

    for _, trade in trades_df.iterrows():
        print(f"\nTrade #{int(trade['trade_num'])}")
        print(f"Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')} @ {trade['entry_price']:.1f}")
        print(f"Exit:  {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')} @ {trade['exit_price']:.1f}")
        print(f"P&L: {trade['pnl_points']:.1f} points")
        print(f"P&L (Gross): NTD {trade['pnl_ntd_gross']:.2f}")
        print(f"P&L (Net):   NTD {trade['pnl_ntd_net']:.2f} (after NTD {trade['transaction_fees']:.0f} fees)")
        print(f"Duration: {trade['duration_minutes']:.1f} minutes")
        print("-" * 50)

def main():
    """Main function to run the complete analysis"""
    print("SuperTrend Strategy Performance Analysis")
    print("=" * 50)

    # Load data
    file_path = "merged_timeframe_signals.csv"
    df = load_strategy_data(file_path)

    print(f"Loaded {len(df)} bars of data")
    print(f"Date range: {df['start_time'].min()} to {df['end_time'].max()}")

    # Calculate trades
    trades_df = calculate_trades(df)

    if trades_df.empty:
        print("\nNo completed trades found in the data!")
        return

    print(f"\nFound {len(trades_df)} completed trades")

    # Calculate performance metrics
    metrics = calculate_performance_metrics(trades_df)

    # Print performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)

    for key, value in metrics.items():
        print(f"{key:.<30} {value}")

    # Generate detailed trade report
    generate_trade_report(trades_df)

    # Create performance charts
    plot_performance_charts(trades_df, df)

    # Save results
    trades_df.to_csv('strategy_trades_analysis.csv', index=False)

    # Performance summary to file
    with open('strategy_performance_summary.txt', 'w') as f:
        f.write("SuperTrend Strategy Performance Summary\n")
        f.write("="*50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"\nResults saved to:")
    print("- strategy_trades_analysis.csv (detailed trades)")
    print("- strategy_performance_summary.txt (metrics summary)")
    print("- strategy_performance_charts.png (visualizations)")

if __name__ == "__main__":
    main()