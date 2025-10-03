import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib

# buy at the place where only a few people can buy, 
# and sell at where only a few people can sell.

def create_volume_bars(df, volume_per_bar=1000):
    """
    Create volume bars from tick data
    
    Parameters:
    df: DataFrame with columns including volume and price data
    volume_per_bar: Fixed volume amount per bar
    
    Returns:
    DataFrame with OHLCV data for each volume bar
    """
    # Combine date and time to create proper datetime
    df['datetime'] = pd.to_datetime(df['成交日期'] + ' ' + df['成交時間'])
    
    # Sort by datetime to ensure proper order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Save the sorted dataframe with datetime column
    df.to_csv('data/sorted_with_datetime.csv', index=False)
    
    # Calculate cumulative volume
    df['cumulative_volume'] = df['成交數量(B+S)'].cumsum()
    
    # Create volume bar groups
    df['volume_bar'] = (df['cumulative_volume'] // volume_per_bar).astype(int)
    
    # Group by volume bar and create OHLCV data
    volume_bars = df.groupby('volume_bar').agg({
        'datetime': ['first', 'last'],
        '成交價格': ['first', 'max', 'min', 'last'],
        '成交數量(B+S)': 'sum'
    }).reset_index()
    
    # Flatten column names
    volume_bars.columns = ['volume_bar', 'start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume']
    
    # Calculate ATR using talib
    volume_bars['atr'] = talib.ATR(volume_bars['high'].values, 
                                   volume_bars['low'].values, 
                                   volume_bars['close'].values, 
                                   timeperiod=14)
    
    # Calculate RSI
    volume_bars['rsi'] = talib.RSI(volume_bars['close'].values, timeperiod=14)
    
    # Calculate SMA 20
    volume_bars['sma_20'] = talib.SMA(volume_bars['close'].values, timeperiod=20)
    
    # Generate buy and sell signals based on RSI
    signals = generate_rsi_signals(volume_bars['rsi'].values)
    volume_bars['buy_signal'] = signals['buy']
    volume_bars['sell_signal'] = signals['sell']
    
    # Calculate Supertrend
    supertrend, trend = calculate_supertrend(volume_bars['high'].values,
                                           volume_bars['low'].values,
                                           volume_bars['close'].values,
                                           period=10, 
                                           multiplier=3.0)
    
    volume_bars['supertrend'] = supertrend
    volume_bars['trend'] = trend
    
    return volume_bars

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator using ATR
    
    Parameters:
    high, low, close: price arrays
    period: ATR period (default 10)
    multiplier: multiplier for ATR (default 3.0)
    
    Returns:
    supertrend, trend: arrays for supertrend values and trend direction
    """
    # Calculate ATR
    atr = talib.ATR(high, low, close, timeperiod=period)
    
    # Handle NaN values in ATR by forward filling
    atr = pd.Series(atr).fillna(method='bfill').fillna(method='ffill').values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    basic_upper_band = hl2 + (multiplier * atr)
    basic_lower_band = hl2 - (multiplier * atr)
    
    # Initialize arrays
    final_upper_band = np.zeros_like(close)
    final_lower_band = np.zeros_like(close)
    supertrend = np.zeros_like(close)
    trend = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    # Start calculations from the period index to avoid NaN issues
    start_idx = max(1, period)
    
    for i in range(len(close)):
        if i < start_idx:
            # For initial values, use simple calculation
            final_upper_band[i] = basic_upper_band[i] if not np.isnan(basic_upper_band[i]) else hl2[i] + (multiplier * np.nanmean(atr[:i+5]) if i < 5 else np.nanmean(atr[max(0,i-period):i+1]))
            final_lower_band[i] = basic_lower_band[i] if not np.isnan(basic_lower_band[i]) else hl2[i] - (multiplier * np.nanmean(atr[:i+5]) if i < 5 else np.nanmean(atr[max(0,i-period):i+1]))
        else:
            # Final upper band
            if np.isnan(basic_upper_band[i]):
                final_upper_band[i] = final_upper_band[i-1]
            elif basic_upper_band[i] < final_upper_band[i-1] or close[i-1] > final_upper_band[i-1]:
                final_upper_band[i] = basic_upper_band[i]
            else:
                final_upper_band[i] = final_upper_band[i-1]
                
            # Final lower band
            if np.isnan(basic_lower_band[i]):
                final_lower_band[i] = final_lower_band[i-1]
            elif basic_lower_band[i] > final_lower_band[i-1] or close[i-1] < final_lower_band[i-1]:
                final_lower_band[i] = basic_lower_band[i]
            else:
                final_lower_band[i] = final_lower_band[i-1]
        
        # Determine trend and supertrend
        if i == 0:
            if close[i] <= final_lower_band[i]:
                supertrend[i] = final_upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = final_lower_band[i]
                trend[i] = 1
        else:
            if trend[i-1] == 1 and close[i] > final_lower_band[i]:
                supertrend[i] = final_lower_band[i]
                trend[i] = 1
            elif trend[i-1] == 1 and close[i] <= final_lower_band[i]:
                supertrend[i] = final_upper_band[i]
                trend[i] = -1
            elif trend[i-1] == -1 and close[i] < final_upper_band[i]:
                supertrend[i] = final_upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = final_lower_band[i]
                trend[i] = 1
    
    return supertrend, trend

def generate_rsi_signals(rsi, oversold_level=30, overbought_level=70):
    """
    Generate buy and sell signals based on RSI levels
    
    Parameters:
    rsi: RSI values array
    oversold_level: RSI level below which to generate buy signals (default 30)
    overbought_level: RSI level above which to generate sell signals (default 70)
    
    Returns:
    Dictionary with 'buy' and 'sell' signal arrays
    """
    buy_signals = np.zeros_like(rsi, dtype=bool)
    sell_signals = np.zeros_like(rsi, dtype=bool)
    
    # Generate signals when RSI crosses the thresholds
    for i in range(1, len(rsi)):
        if not np.isnan(rsi[i]) and not np.isnan(rsi[i-1]):
            # Buy signal: RSI crosses above oversold level from below
            if rsi[i-1] <= oversold_level and rsi[i] > oversold_level:
                buy_signals[i] = True
            
            # Sell signal: RSI crosses below overbought level from above
            if rsi[i-1] >= overbought_level and rsi[i] < overbought_level:
                sell_signals[i] = True
    
    return {'buy': buy_signals, 'sell': sell_signals}

def plot_volume_bars(volume_bars, title="Volume Bars Chart"):
    """
    Plot volume bars as candlestick chart with RSI buy/sell signals
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Price chart (candlesticks only)
    for i, row in volume_bars.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Draw the bar (high-low line)
        ax1.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
        
        # Draw the body (open-close rectangle)
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        ax1.bar(i, body_height, bottom=body_bottom, color=color, alpha=0.7, width=0.8)
    
    # Plot buy and sell signals
    buy_indices = volume_bars.index[volume_bars['buy_signal']].tolist()
    sell_indices = volume_bars.index[volume_bars['sell_signal']].tolist()
    
    if buy_indices:
        ax1.scatter(buy_indices, volume_bars.loc[buy_indices, 'close'], 
                   color='lime', marker='^', s=100, label='Buy Signal', zorder=5)
    
    if sell_indices:
        ax1.scatter(sell_indices, volume_bars.loc[sell_indices, 'close'], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    # Plot SMA 20 line
    ax1.plot(volume_bars.index, volume_bars['sma_20'], color='blue', linewidth=2, label='SMA 20', alpha=0.8)
    
    ax1.set_title(f'{title} - Price with RSI Signals')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # RSI subplot
    ax2.plot(volume_bars.index, volume_bars['rsi'], color='purple', linewidth=1.5, label='RSI')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='black', linestyle='-', alpha=0.5)
    
    # Highlight RSI signal areas
    if buy_indices:
        ax2.scatter(buy_indices, volume_bars.loc[buy_indices, 'rsi'], 
                   color='lime', marker='^', s=100, zorder=5)
    
    if sell_indices:
        ax2.scatter(sell_indices, volume_bars.loc[sell_indices, 'rsi'], 
                   color='red', marker='v', s=100, zorder=5)
    
    ax2.set_title('RSI Indicator')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Volume Bar Number')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Load and filter data
import glob
import os

# Get all CSV files matching the pattern
csv_pattern = 'data/Daily_*_cleaned.csv'
csv_files = glob.glob(csv_pattern)

if not csv_files:
    raise FileNotFoundError(f"No CSV files found matching pattern: {csv_pattern}")

print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")

# Load and combine all CSV files
df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    print(f"Loaded {len(temp_df)} records from {os.path.basename(file)}")
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
# con1 = df.商品代號 == 'TX'
# con2 = df['到期月份(週別)'] == '202509'
# df = df[con1 & con2]

print(f"Total records: {len(df)}")
print(f"Total volume: {df['成交數量(B+S)'].sum()}")

# Create volume bars with 1000 volume per bar
volume_per_bar = 1000
volume_bars = create_volume_bars(df, volume_per_bar)

print(f"\nCreated {len(volume_bars)} volume bars with {volume_per_bar} volume each")
print("\nFirst few volume bars:")
print(volume_bars.head())

# Plot the volume bars
plot_volume_bars(volume_bars, f"TX Volume Bars ({volume_per_bar} per bar)")