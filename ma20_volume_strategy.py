import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import glob
import os

def create_volume_bars(df, volume_per_bar=1000):
    """
    Create volume bars for MA20 crossover strategy
    
    Parameters:
    df: DataFrame with tick data
    volume_per_bar: Fixed volume amount per bar
    
    Returns:
    DataFrame with OHLCV data
    """
    # Combine date and time to create proper datetime
    df['datetime'] = pd.to_datetime(df['成交日期'] + ' ' + df['成交時間'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate cumulative volume
    df['cumulative_volume'] = df['成交數量(B+S)'].cumsum()
    df['volume_bar'] = (df['cumulative_volume'] // volume_per_bar).astype(int)
    
    # Group by volume bar and create OHLCV data
    volume_bars = df.groupby('volume_bar').agg({
        'datetime': ['first', 'last'],
        '成交價格': ['first', 'max', 'min', 'last'],
        '成交數量(B+S)': 'sum'
    }).reset_index()
    
    # Flatten column names
    volume_bars.columns = ['volume_bar', 'start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume']
    
    # Calculate MA20
    volume_bars['ma20'] = talib.SMA(volume_bars['close'].values, timeperiod=20)
    
    # Generate trading signals
    signals = generate_ma20_signals(volume_bars)
    volume_bars = pd.concat([volume_bars, signals], axis=1)
    
    return volume_bars

def generate_ma20_signals(volume_bars):
    """
    Generate MA20 crossover signals - long only strategy with advanced exit logic
    
    Entry: Close price crosses above MA20
    Exit: 
    1. Stop loss: -1% from local minima in previous 20-bar window
    2. Trend following: After 10 bars from entry, exit when close below MA20
    """
    signals = pd.DataFrame(index=volume_bars.index)
    
    # Calculate rolling minimum for stop loss
    volume_bars['rolling_min_20'] = volume_bars['low'].rolling(window=20).min()
    volume_bars['stop_loss_level'] = volume_bars['rolling_min_20'] * 0.99  # -1% from local minima
    
    # Entry signal: close crosses above MA20
    signals['buy_signal'] = (
        (volume_bars['close'].shift(1) <= volume_bars['ma20'].shift(1)) & 
        (volume_bars['close'] > volume_bars['ma20'])
    )
    
    # Initialize exit signals
    signals['exit_long_stop'] = False
    signals['exit_long_trend'] = False
    signals['exit_long'] = False
    
    # No short signals (long only strategy)
    signals['sell_signal'] = False
    signals['exit_short'] = False
    
    return signals

def calculate_strategy_performance(volume_bars, initial_capital=100000):
    """
    Calculate strategy performance with advanced exit logic
    """
    # Create positions based on signals
    positions = pd.DataFrame(index=volume_bars.index)
    positions['position'] = 0
    positions['entry_bar'] = 0
    positions['entry_price'] = 0.0
    
    # Track position state for proper entry/exit logic
    current_position = 0
    entry_bar = 0
    entry_price = 0.0
    
    # Apply advanced exit logic
    for i in volume_bars.index:
        # Entry logic - LONG ONLY
        if current_position == 0:  # No position
            if volume_bars.loc[i, 'buy_signal']:
                current_position = 1  # Go long
                entry_bar = i
                entry_price = volume_bars.loc[i, 'close']
        
        # Exit logic - LONG ONLY with advanced conditions
        elif current_position == 1:  # Long position
            bars_since_entry = i - entry_bar
            
            # Stop loss: close below stop loss level
            stop_loss_exit = volume_bars.loc[i, 'close'] < volume_bars.loc[i, 'stop_loss_level']
            
            # Trend following: after 10 bars, exit when close below MA20
            trend_exit = (bars_since_entry >= 10) and (volume_bars.loc[i, 'close'] < volume_bars.loc[i, 'ma20'])
            
            # Update exit signals
            volume_bars.loc[i, 'exit_long_stop'] = stop_loss_exit
            volume_bars.loc[i, 'exit_long_trend'] = trend_exit
            volume_bars.loc[i, 'exit_long'] = stop_loss_exit or trend_exit
            
            if stop_loss_exit or trend_exit:
                current_position = 0  # Exit long
                entry_bar = 0
                entry_price = 0.0
        
        positions.loc[i, 'position'] = current_position
        positions.loc[i, 'entry_bar'] = entry_bar
        positions.loc[i, 'entry_price'] = entry_price
    
    # Calculate returns
    volume_bars['returns'] = volume_bars['close'].pct_change()
    positions['strategy_returns'] = positions['position'].shift(1) * volume_bars['returns']
    
    # Calculate cumulative performance
    positions['cumulative_returns'] = (1 + positions['strategy_returns']).cumprod()
    positions['equity_curve'] = initial_capital * positions['cumulative_returns']
    
    # Performance metrics
    total_return = positions['cumulative_returns'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(volume_bars)) - 1
    volatility = positions['strategy_returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Win rate
    winning_trades = positions['strategy_returns'][positions['strategy_returns'] > 0]
    losing_trades = positions['strategy_returns'][positions['strategy_returns'] < 0]
    win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if len(winning_trades) + len(losing_trades) > 0 else 0
    
    # Maximum drawdown
    rolling_max = positions['equity_curve'].expanding().max()
    drawdown = (positions['equity_curve'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    performance_metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Win Rate': f"{win_rate:.2%}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Total Trades': len(winning_trades) + len(losing_trades)
    }
    
    return positions, performance_metrics

def plot_ma20_strategy(volume_bars, positions):
    """
    Visualization of MA20 crossover strategy with advanced exit logic
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # 1. Price chart with MA20 and signals
    ax1 = axes[0]
    ax1.plot(volume_bars.index, volume_bars['close'], color='black', linewidth=1, label='Close Price')
    ax1.plot(volume_bars.index, volume_bars['ma20'], color='blue', linewidth=1.5, label='MA20')
    ax1.plot(volume_bars.index, volume_bars['stop_loss_level'], color='red', linewidth=1, alpha=0.7, label='Stop Loss Level')
    
    # Buy and exit signals
    buy_signals = volume_bars.index[volume_bars['buy_signal']]
    exit_signals = volume_bars.index[volume_bars['exit_long']]
    stop_exits = volume_bars.index[volume_bars['exit_long_stop']]
    trend_exits = volume_bars.index[volume_bars['exit_long_trend']]
    
    if len(buy_signals) > 0:
        ax1.scatter(buy_signals, volume_bars.loc[buy_signals, 'close'], 
                   color='green', marker='^', s=100, label='Long Entry', zorder=5)
    
    if len(stop_exits) > 0:
        ax1.scatter(stop_exits, volume_bars.loc[stop_exits, 'close'], 
                   color='red', marker='v', s=100, label='Stop Loss Exit', zorder=5)
                   
    if len(trend_exits) > 0:
        ax1.scatter(trend_exits, volume_bars.loc[trend_exits, 'close'], 
                   color='orange', marker='v', s=100, label='Trend Exit', zorder=5)
    
    ax1.set_title('MA20 Strategy with Advanced Exit Logic')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Exit signal breakdown
    ax2 = axes[1]
    exit_types = pd.DataFrame(index=volume_bars.index)
    exit_types['stop_loss'] = volume_bars['exit_long_stop'].astype(int)
    exit_types['trend_exit'] = volume_bars['exit_long_trend'].astype(int)
    
    ax2.bar(volume_bars.index, exit_types['stop_loss'], color='red', alpha=0.7, label='Stop Loss Exits')
    ax2.bar(volume_bars.index, exit_types['trend_exit'], color='orange', alpha=0.7, label='Trend Exits', bottom=exit_types['stop_loss'])
    ax2.set_title('Exit Signal Types')
    ax2.set_ylabel('Exit Signal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position tracking
    ax3 = axes[2]
    ax3.plot(volume_bars.index, positions['position'], color='blue', linewidth=2, label='Position')
    ax3.fill_between(volume_bars.index, 0, positions['position'], alpha=0.3, color='blue')
    ax3.set_title('Position Tracking (1=Long, 0=Flat)')
    ax3.set_ylabel('Position')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Equity curve
    ax4 = axes[3]
    ax4.plot(volume_bars.index, positions['equity_curve'], color='green', linewidth=2, label='Strategy Equity')
    ax4.set_title('Strategy Performance - Equity Curve')
    ax4.set_ylabel('Portfolio Value')
    ax4.set_xlabel('Volume Bar Number')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution function
    """
    print("Loading TXF data for MA20 Volume Strategy Analysis...")
    
    # Load data
    csv_pattern = 'data/Daily_*_cleaned.csv'
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {csv_pattern}")
    
    # Load and combine data
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} total records")
    
    # Create volume bars
    volume_per_bar = 1000
    print(f"Creating volume bars with {volume_per_bar} volume per bar...")
    volume_bars = create_volume_bars(df, volume_per_bar)
    
    print(f"Created {len(volume_bars)} volume bars")
    print(f"Long entry signals: {volume_bars['buy_signal'].sum()}")
    print(f"Long exit signals: {volume_bars['exit_long'].sum()}")
    print(f"Stop loss exits: {volume_bars['exit_long_stop'].sum()}")
    print(f"Trend exits: {volume_bars['exit_long_trend'].sum()}")
    
    # Calculate performance
    positions, performance_metrics = calculate_strategy_performance(volume_bars)
    
    print("\n=== STRATEGY PERFORMANCE ===")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")
    
    # Display additional metrics
    print("\n=== VOLUME BAR STATISTICS ===")
    print(f"Average volume per bar: {volume_bars['volume'].mean():.0f}")
    print(f"Total volume bars: {len(volume_bars)}")
    print(f"Price range: {volume_bars['close'].min():.2f} - {volume_bars['close'].max():.2f}")
    
    # Plot results
    plot_ma20_strategy(volume_bars, positions)
    
    # Save results
    output_file = 'ma20_volume_strategy_results.csv'
    volume_bars.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return volume_bars, positions, performance_metrics

if __name__ == "__main__":
    volume_bars, positions, performance = main()