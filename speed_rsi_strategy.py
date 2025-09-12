import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import glob
import os

def create_speed_volume_bars(df, volume_per_bar=1000):
    """
    Create volume bars with speed metrics for quantitative trading
    
    Parameters:
    df: DataFrame with tick data
    volume_per_bar: Fixed volume amount per bar
    
    Returns:
    DataFrame with OHLCV data and speed metrics
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
    
    # Calculate speed metrics
    volume_bars = add_speed_metrics(volume_bars)
    
    # Calculate technical indicators
    volume_bars['rsi'] = talib.RSI(volume_bars['close'].values, timeperiod=14)
    volume_bars['atr'] = talib.ATR(volume_bars['high'].values, 
                                   volume_bars['low'].values, 
                                   volume_bars['close'].values, 
                                   timeperiod=14)
    volume_bars['ma20'] = talib.SMA(volume_bars['close'].values, timeperiod=20)
    
    # Generate enhanced trading signals
    signals = generate_speed_rsi_signals(volume_bars)
    volume_bars = pd.concat([volume_bars, signals], axis=1)
    
    return volume_bars

def add_speed_metrics(volume_bars, lookback_fast=10, lookback_slow=50):
    """
    Add quantitative speed metrics to volume bars
    """
    # Calculate duration for each bar (time to accumulate volume)
    volume_bars['duration'] = (volume_bars['end_time'] - volume_bars['start_time']).dt.total_seconds()
    
    # Handle zero duration (same timestamp trades)
    volume_bars['duration'] = volume_bars['duration'].replace(0, 0.1)  # minimum 0.1 second
    
    # Primary speed metric: bars per second
    volume_bars['speed'] = 1 / volume_bars['duration']
    
    # Volume velocity: contracts per minute
    volume_bars['volume_velocity'] = volume_bars['volume'] / (volume_bars['duration'] / 60)
    
    # Speed rankings and percentiles
    volume_bars['speed_rank_20'] = volume_bars['speed'].rolling(20).rank(pct=True)
    volume_bars['speed_rank_50'] = volume_bars['speed'].rolling(50).rank(pct=True)
    volume_bars['speed_rank_100'] = volume_bars['speed'].rolling(100).rank(pct=True)
    
    # Speed moving averages
    volume_bars['speed_ma_fast'] = volume_bars['speed'].rolling(lookback_fast).mean()
    volume_bars['speed_ma_slow'] = volume_bars['speed'].rolling(lookback_slow).mean()
    volume_bars['speed_ratio'] = volume_bars['speed_ma_fast'] / volume_bars['speed_ma_slow']
    
    # Speed acceleration (change in speed)
    volume_bars['speed_acceleration'] = volume_bars['speed'].diff()
    volume_bars['acceleration_rank'] = volume_bars['speed_acceleration'].rolling(20).rank(pct=True)
    
    # Volume velocity z-score
    velocity_mean = volume_bars['volume_velocity'].rolling(50).mean()
    velocity_std = volume_bars['volume_velocity'].rolling(50).std()
    volume_bars['velocity_zscore'] = (volume_bars['volume_velocity'] - velocity_mean) / velocity_std
    
    # Speed volatility and Sharpe-like ratio
    speed_mean = volume_bars['speed'].rolling(50).mean()
    speed_std = volume_bars['speed'].rolling(20).std()
    volume_bars['speed_sharpe'] = (volume_bars['speed'] - speed_mean) / speed_std
    
    # Composite speed score (0-100 scale)
    volume_bars['speed_score'] = (
        volume_bars['speed_rank_50'] * 0.4 +
        volume_bars['speed_ratio'].rank(pct=True) * 0.3 +
        volume_bars['velocity_zscore'].rank(pct=True) * 0.3
    ) * 100
    
    return volume_bars

def generate_speed_rsi_signals(volume_bars, 
                              rsi_oversold=30, rsi_overbought=70,
                              min_speed_threshold=20):
    """
    Generate trading signals based on RSI entry with advanced exit logic and minimum speed filter
    
    Parameters:
    volume_bars: DataFrame with speed metrics and RSI
    rsi_oversold/overbought: RSI threshold levels for entry/exit
    min_speed_threshold: Minimum speed percentile to allow trading (0-100)
    """
    signals = pd.DataFrame(index=volume_bars.index)
    
    # Calculate rolling minimum for stop loss
    volume_bars['rolling_min_20'] = volume_bars['low'].rolling(window=20).min()
    volume_bars['stop_loss_level'] = volume_bars['rolling_min_20'] * 0.99  # -1% from local minima
    
    # Speed filter - only trade when market is NOT too slow
    signals['speed_ok'] = volume_bars['speed_rank_50'] > (min_speed_threshold / 100)
    
    # RSI-based entry conditions
    signals['rsi_oversold'] = volume_bars['rsi'] < rsi_oversold
    signals['rsi_overbought'] = volume_bars['rsi'] > rsi_overbought
    
    # Entry signals: RSI crosses from oversold (long only strategy)
    signals['rsi_buy_entry'] = (volume_bars['rsi'].shift(1) <= rsi_oversold) & (volume_bars['rsi'] > rsi_oversold)
    
    # Combined entry signals - only when speed is adequate (LONG ONLY)
    signals['buy_signal'] = signals['rsi_buy_entry'] & signals['speed_ok']
    signals['sell_signal'] = False  # No short signals
    
    # Initialize exit signals (will be calculated in performance function)
    signals['exit_long_stop'] = False
    signals['exit_long_trend'] = False
    signals['exit_long'] = False
    signals['exit_short'] = False  # No short positions
    
    # Additional market condition flags for analysis
    signals['too_slow'] = ~signals['speed_ok']
    signals['missed_entry_slow'] = (
        signals['rsi_buy_entry'] & 
        signals['too_slow']
    )
    
    return signals

def calculate_strategy_performance(volume_bars, initial_capital=100000):
    """
    Calculate strategy performance metrics with advanced exit logic
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
    annualized_return = (1 + total_return) ** (252 / len(volume_bars)) - 1  # Assuming daily bars
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

def plot_speed_rsi_strategy(volume_bars, positions):
    """
    Comprehensive visualization of speed-RSI strategy with advanced exit logic
    """
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    
    # 1. Price chart with signals
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
    
    ax1.set_title('Price Chart with Speed-RSI Signals and Advanced Exits')
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
    
    # 3. Speed metrics
    ax3 = axes[2]
    ax3.plot(volume_bars.index, volume_bars['speed_score'], color='blue', label='Speed Score')
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Min Speed Threshold (20)')
    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax3.fill_between(volume_bars.index, 0, volume_bars['speed_score'], 
                     where=volume_bars['speed_ok'], alpha=0.3, color='green', label='Speed OK Regime')
    
    # Mark missed entries due to slow speed
    missed_entries = volume_bars.index[volume_bars['missed_entry_slow']]
    if len(missed_entries) > 0:
        ax3.scatter(missed_entries, volume_bars.loc[missed_entries, 'speed_score'], 
                   color='red', marker='x', s=50, label='Missed Entry (Too Slow)', zorder=5)
    
    ax3.set_title('Market Speed Score')
    ax3.set_ylabel('Speed Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. RSI
    ax4 = axes[3]
    ax4.plot(volume_bars.index, volume_bars['rsi'], color='purple', linewidth=1.5, label='RSI')
    ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax4.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    
    # Highlight signal areas (long only)
    if len(buy_signals) > 0:
        ax4.scatter(buy_signals, volume_bars.loc[buy_signals, 'rsi'], 
                   color='green', marker='^', s=100, zorder=5)
    
    if len(exit_signals) > 0:
        ax4.scatter(exit_signals, volume_bars.loc[exit_signals, 'rsi'], 
                   color='red', marker='v', s=100, zorder=5)
    
    ax4.set_title('RSI with Trading Signals')
    ax4.set_ylabel('RSI')
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Equity curve
    ax5 = axes[4]
    ax5.plot(volume_bars.index, positions['equity_curve'], color='green', linewidth=2, label='Strategy Equity')
    ax5.set_title('Strategy Performance - Equity Curve')
    ax5.set_ylabel('Portfolio Value')
    ax5.set_xlabel('Volume Bar Number')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution function
    """
    print("Loading TXF data for Speed-RSI Strategy Analysis...")
    
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
    
    # Create volume bars with speed metrics
    volume_per_bar = 1000
    print(f"Creating volume bars with {volume_per_bar} volume per bar...")
    volume_bars = create_speed_volume_bars(df, volume_per_bar)
    
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
    
    # Display speed metrics summary  
    print("\n=== SPEED METRICS SUMMARY ===")
    print(f"Average bar duration: {volume_bars['duration'].mean():.2f} seconds")
    print(f"Fastest bar duration: {volume_bars['duration'].min():.2f} seconds")
    print(f"Slowest bar duration: {volume_bars['duration'].max():.2f} seconds")
    print(f"Average speed score: {volume_bars['speed_score'].mean():.2f}")
    print(f"Missed entries due to slow speed: {volume_bars['missed_entry_slow'].sum()}")
    print(f"Percentage of time speed was OK: {(volume_bars['speed_ok'].sum()/len(volume_bars)*100):.1f}%")
    
    # Plot results
    plot_speed_rsi_strategy(volume_bars, positions)
    
    # Save results
    output_file = 'speed_rsi_analysis_results.csv'
    volume_bars.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return volume_bars, positions, performance_metrics

if __name__ == "__main__":
    volume_bars, positions, performance = main()