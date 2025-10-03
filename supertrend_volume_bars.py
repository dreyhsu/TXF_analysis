import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import glob
import os

def create_volume_bars(df, volume_per_bar=1000):
    """
    Create volume bars for SuperTrend analysis
    
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
    
    # Calculate ATR for SuperTrend
    volume_bars['atr'] = talib.ATR(volume_bars['high'].values, 
                                   volume_bars['low'].values, 
                                   volume_bars['close'].values, 
                                   timeperiod=14)
    
    # Add SuperTrend indicator
    volume_bars = add_supertrend(volume_bars, atr_multiplier=3.0)
    
    return volume_bars

def add_supertrend(df, atr_multiplier=3.0):
    """
    Calculate SuperTrend indicator using the correct method
    
    Parameters:
    df: DataFrame with OHLC and ATR data
    atr_multiplier: ATR multiplier (default 3.0)
    
    Returns:
    DataFrame with SuperTrend columns added
    """
    # Calculate the basic upper and lower bands
    current_average_high_low = (df['high'] + df['low']) / 2
    df['basic_upperband'] = current_average_high_low + (atr_multiplier * df['atr'])
    df['basic_lowerband'] = current_average_high_low - (atr_multiplier * df['atr'])
    
    # Initialize final bands with first values
    first_upperband_value = df['basic_upperband'].iloc[0]
    first_lowerband_value = df['basic_lowerband'].iloc[0]
    
    upperband = [first_upperband_value]
    lowerband = [first_lowerband_value]
    
    # Calculate final upper and lower bands
    for i in range(1, len(df)):
        # Final upper band logic
        if (df['basic_upperband'].iloc[i] < upperband[i-1] or 
            df['close'].iloc[i-1] > upperband[i-1]):
            upperband.append(df['basic_upperband'].iloc[i])
        else:
            upperband.append(upperband[i-1])
        
        # Final lower band logic
        if (df['basic_lowerband'].iloc[i] > lowerband[i-1] or 
            df['close'].iloc[i-1] < lowerband[i-1]):
            lowerband.append(df['basic_lowerband'].iloc[i])
        else:
            lowerband.append(lowerband[i-1])
    
    # Assign final bands to dataframe
    df['upperband'] = upperband
    df['lowerband'] = lowerband
    
    # Generate signals using correct logic
    signals = [0]  # Start with neutral signal
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i]:
            signals.append(1)  # Bullish signal
        elif df['close'].iloc[i] < df['lowerband'].iloc[i]:
            signals.append(-1)  # Bearish signal
        else:
            signals.append(signals[i-1])  # Keep previous signal
    
    # Add signals to dataframe and shift to remove look-ahead bias
    df['signals'] = signals
    df['signals'] = df['signals'].shift(1)
    
    # Create positions for visualization (hide bands when not in use)
    df['upperband_plot'] = df['upperband'].copy()
    df['lowerband_plot'] = df['lowerband'].copy()
    
    # Hide upperband when signal is bullish (1), show lowerband
    df.loc[df['signals'] == 1, 'upperband_plot'] = np.nan
    # Hide lowerband when signal is bearish (-1), show upperband  
    df.loc[df['signals'] == -1, 'lowerband_plot'] = np.nan
    
    # Generate buy/sell positions (only at signal changes)
    buy_positions = [np.nan]
    sell_positions = [np.nan]
    
    for i in range(1, len(df)):
        # Buy signal: transition from bearish to bullish
        if df['signals'].iloc[i] == 1 and df['signals'].iloc[i] != df['signals'].iloc[i-1]:
            buy_positions.append(df['close'].iloc[i])
            sell_positions.append(np.nan)
        # Sell signal: transition from bullish to bearish
        elif df['signals'].iloc[i] == -1 and df['signals'].iloc[i] != df['signals'].iloc[i-1]:
            sell_positions.append(df['close'].iloc[i])
            buy_positions.append(np.nan)
        else:
            buy_positions.append(np.nan)
            sell_positions.append(np.nan)
    
    df['buy_positions'] = buy_positions
    df['sell_positions'] = sell_positions
    
    # Clean up intermediate columns
    df.drop(['basic_upperband', 'basic_lowerband'], axis=1, inplace=True)
    
    return df

def plot_supertrend_analysis(volume_bars):
    """
    Plot SuperTrend analysis using the correct visualization method
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Main Price chart with SuperTrend bands
    ax1 = axes[0]
    
    # Plot price line
    ax1.plot(volume_bars.index, volume_bars['close'], color='black', linewidth=1.5, label='Close Price', zorder=3)
    
    # Plot SuperTrend bands (only show active band based on signal)
    # Green line for lowerband when bullish (signal = 1)
    ax1.plot(volume_bars.index, volume_bars['lowerband_plot'], color='green', linewidth=2, 
             label='SuperTrend Support (Bullish)', zorder=2)
    
    # Red line for upperband when bearish (signal = -1)  
    ax1.plot(volume_bars.index, volume_bars['upperband_plot'], color='red', linewidth=2,
             label='SuperTrend Resistance (Bearish)', zorder=2)
    
    # Plot buy/sell signals
    buy_indices = volume_bars.index[~np.isnan(volume_bars['buy_positions'])]
    sell_indices = volume_bars.index[~np.isnan(volume_bars['sell_positions'])]
    
    if len(buy_indices) > 0:
        ax1.scatter(buy_indices, volume_bars.loc[buy_indices, 'buy_positions'], 
                   color='#2cf651', marker='^', s=150, label='Buy Signal', zorder=5, 
                   edgecolors='white', linewidth=2)
    
    if len(sell_indices) > 0:
        ax1.scatter(sell_indices, volume_bars.loc[sell_indices, 'sell_positions'], 
                   color='#f50100', marker='v', s=150, label='Sell Signal', zorder=5, 
                   edgecolors='white', linewidth=2)
    
    # Fill between price and SuperTrend bands for better visualization
    # Fill green when bullish (below price), red when bearish (above price)
    bullish_mask = volume_bars['signals'] == 1
    bearish_mask = volume_bars['signals'] == -1
    
    if bullish_mask.any():
        ax1.fill_between(volume_bars.index[bullish_mask], 
                        volume_bars['close'][bullish_mask], 
                        volume_bars['lowerband'][bullish_mask],
                        alpha=0.2, color='green', interpolate=True)
    
    if bearish_mask.any():
        ax1.fill_between(volume_bars.index[bearish_mask], 
                        volume_bars['close'][bearish_mask], 
                        volume_bars['upperband'][bearish_mask],
                        alpha=0.2, color='red', interpolate=True)
    
    ax1.set_title('Volume Bars with SuperTrend Indicator (ATR=14, Multiplier=3.0)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ATR for SuperTrend calculation context
    ax2 = axes[1]
    ax2.plot(volume_bars.index, volume_bars['atr'], color='purple', linewidth=2, label='ATR(14)')
    ax2.fill_between(volume_bars.index, volume_bars['atr'], alpha=0.2, color='purple')
    ax2.set_title('Average True Range (ATR) - Used for SuperTrend Calculation')
    ax2.set_ylabel('ATR')
    ax2.set_xlabel('Volume Bar Number')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print SuperTrend calculation explanation (CORRECTED)
    print("\n=== CORRECT SUPERTREND CALCULATION METHOD ===")
    print("SuperTrend is calculated using the following steps:")
    print("1. Calculate HL2 (Typical Price) = (High + Low) / 2")
    print("2. Calculate Basic Upper Band = HL2 + (Multiplier × ATR)")
    print("3. Calculate Basic Lower Band = HL2 - (Multiplier × ATR)")
    print("4. Calculate Final Upper Band:")
    print("   - If Basic Upper Band < Previous Final Upper Band OR Previous Close > Previous Final Upper Band:")
    print("     Final Upper Band = Basic Upper Band")
    print("   - Else: Final Upper Band = Previous Final Upper Band")
    print("5. Calculate Final Lower Band:")
    print("   - If Basic Lower Band > Previous Final Lower Band OR Previous Close < Previous Final Lower Band:")
    print("     Final Lower Band = Basic Lower Band")
    print("   - Else: Final Lower Band = Previous Final Lower Band")
    print("6. Generate Trading Signals:")
    print("   - If Close > Upper Band: Signal = 1 (Bullish)")
    print("   - If Close < Lower Band: Signal = -1 (Bearish)")
    print("   - Otherwise: Keep previous signal")
    print("7. Visualization Logic:")
    print("   - Show Lower Band (green) when signal is Bullish (1)")
    print("   - Show Upper Band (red) when signal is Bearish (-1)")
    print("   - Hide opposite band for cleaner visualization")
    print(f"\nParameters used: ATR Period = 14, Multiplier = 3.0")
    print(f"Volume per bar = 1000 contracts")

def calculate_supertrend_performance(volume_bars, initial_capital=100000):
    """
    Calculate performance metrics for SuperTrend strategy using correct signals
    """
    positions = pd.DataFrame(index=volume_bars.index)
    positions['position'] = 0
    
    # Use the correct signals column
    positions['position'] = volume_bars['signals']
    
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
    
    # Count actual buy/sell signals (transitions)
    buy_signals = (~np.isnan(volume_bars['buy_positions'])).sum()
    sell_signals = (~np.isnan(volume_bars['sell_positions'])).sum()
    
    performance_metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Win Rate': f"{win_rate:.2%}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Buy Signals': buy_signals,
        'Sell Signals': sell_signals
    }
    
    return positions, performance_metrics

def main():
    """
    Main execution function
    """
    print("Loading TXF data for SuperTrend Volume Bar Analysis...")
    
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
    
    # Create volume bars with SuperTrend
    volume_per_bar = 1000
    print(f"Creating volume bars with {volume_per_bar} volume per bar...")
    volume_bars = create_volume_bars(df, volume_per_bar)
    
    print(f"Created {len(volume_bars)} volume bars")
    buy_signals = (~np.isnan(volume_bars['buy_positions'])).sum()
    sell_signals = (~np.isnan(volume_bars['sell_positions'])).sum()
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    # Calculate performance
    positions, performance_metrics = calculate_supertrend_performance(volume_bars)
    
    print("\n=== SUPERTREND STRATEGY PERFORMANCE ===")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")
    
    # Display SuperTrend metrics summary  
    print("\n=== SUPERTREND METRICS SUMMARY ===")
    print(f"Average ATR: {volume_bars['atr'].mean():.2f}")
    print(f"Bullish signal bars: {(volume_bars['signals'] == 1).sum()}")
    print(f"Bearish signal bars: {(volume_bars['signals'] == -1).sum()}")
    print(f"Neutral signal bars: {(volume_bars['signals'] == 0).sum()}")
    print(f"Bullish percentage: {(volume_bars['signals'] == 1).sum()/len(volume_bars)*100:.1f}%")
    print(f"Bearish percentage: {(volume_bars['signals'] == -1).sum()/len(volume_bars)*100:.1f}%")
    
    # Plot results
    plot_supertrend_analysis(volume_bars)
    
    # Save results
    output_file = 'supertrend_volume_bars_results.csv'
    volume_bars.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return volume_bars, positions, performance_metrics

if __name__ == "__main__":
    volume_bars, positions, performance = main()