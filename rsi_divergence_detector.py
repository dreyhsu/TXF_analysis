import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import glob
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

def create_volume_bars(df, volume_per_bar=1000):
    """
    Create volume bars from tick data
    """
    df['datetime'] = pd.to_datetime(df['成交日期'] + ' ' + df['成交時間'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    df['cumulative_volume'] = df['成交數量(B+S)'].cumsum()
    df['volume_bar'] = (df['cumulative_volume'] // volume_per_bar).astype(int)
    
    volume_bars = df.groupby('volume_bar').agg({
        'datetime': ['first', 'last'],
        '成交價格': ['first', 'max', 'min', 'last'],
        '成交數量(B+S)': 'sum'
    }).reset_index()
    
    volume_bars.columns = ['volume_bar', 'start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume']
    volume_bars.set_index('volume_bar', inplace=True)
    
    return volume_bars

def calculate_rsi(df, period=14):
    """
    Calculate RSI (Relative Strength Index) using talib
    """
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=period)
    return df

def find_local_extrema(series, window=5):
    """
    Find local highs and lows in a price or RSI series
    """
    highs_idx, _ = find_peaks(series.values, distance=window)
    lows_idx, _ = find_peaks(-series.values, distance=window)
    
    return highs_idx, lows_idx

def detect_bullish_divergence(df, lookback_window=25):
    """
    Detect bullish RSI divergence in the last lookback_window bars
    Bullish divergence: price makes lower lows while RSI makes higher lows
    """
    if len(df) < lookback_window:
        return df
    
    df['bullish_divergence'] = False
    
    # Only check the last lookback_window bars
    start_idx = max(0, len(df) - lookback_window)
    window_df = df.iloc[start_idx:].copy()
    
    if len(window_df) < 10:  # Need minimum bars for meaningful analysis
        return df
    
    # Find local extrema in the window
    price_highs_idx, price_lows_idx = find_local_extrema(window_df['close'], window=3)
    rsi_highs_idx, rsi_lows_idx = find_local_extrema(window_df['rsi'], window=3)
    
    # Adjust indices to match original dataframe
    price_lows_idx = price_lows_idx + start_idx
    rsi_lows_idx = rsi_lows_idx + start_idx
    
    # Need at least 2 lows to compare
    if len(price_lows_idx) >= 2 and len(rsi_lows_idx) >= 2:
        # Get the two most recent lows for price and RSI
        recent_price_lows = price_lows_idx[-2:]
        recent_rsi_lows = rsi_lows_idx[-2:]
        
        # Check for bullish divergence pattern
        for i, price_low_idx in enumerate(recent_price_lows[1:], 1):
            prev_price_low_idx = recent_price_lows[i-1]
            
            # Find corresponding RSI lows near these price lows
            price_low_rsi = None
            prev_price_low_rsi = None
            
            # Find RSI low closest to current price low
            for rsi_idx in recent_rsi_lows:
                if abs(rsi_idx - price_low_idx) <= 3:  # Within 3 bars
                    price_low_rsi = rsi_idx
                    break
            
            # Find RSI low closest to previous price low
            for rsi_idx in recent_rsi_lows:
                if abs(rsi_idx - prev_price_low_idx) <= 3:  # Within 3 bars
                    prev_price_low_rsi = rsi_idx
                    break
            
            # Check for bullish divergence
            if (price_low_rsi is not None and prev_price_low_rsi is not None and
                price_low_rsi != prev_price_low_rsi):
                
                current_price_low = df.iloc[price_low_idx]['close']
                prev_price_low = df.iloc[prev_price_low_idx]['close']
                current_rsi_low = df.iloc[price_low_rsi]['rsi']
                prev_rsi_low = df.iloc[prev_price_low_rsi]['rsi']
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                if (current_price_low < prev_price_low and 
                    current_rsi_low > prev_rsi_low):
                    
                    df.loc[price_low_idx, 'bullish_divergence'] = True
                    print(f"Bullish divergence detected at bar {price_low_idx}:")
                    print(f"  Price: {prev_price_low:.2f} -> {current_price_low:.2f} (lower)")
                    print(f"  RSI: {prev_rsi_low:.2f} -> {current_rsi_low:.2f} (higher)")
    
    return df

def plot_data_with_divergence(df, title="RSI Bullish Divergence Detection"):
    """
    Plot price and RSI with divergence signals highlighted
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot price
    ax1.plot(df.index, df['close'], color='black', linewidth=1.5, label='Close Price')
    
    # Highlight bullish divergence points
    divergence_points = df[df['bullish_divergence'] == True]
    if not divergence_points.empty:
        ax1.scatter(divergence_points.index, divergence_points['close'], 
                   color='lime', marker='o', s=100, label='Bullish Divergence', 
                   zorder=5, edgecolors='darkgreen', linewidth=2)
    
    ax1.set_title(f'{title} - Price Chart')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot RSI
    ax2.plot(df.index, df['rsi'], color='purple', linewidth=1.5, label='RSI')
    
    # Add RSI reference lines
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Midline (50)')
    
    # Highlight RSI at divergence points
    if not divergence_points.empty:
        ax2.scatter(divergence_points.index, divergence_points['rsi'], 
                   color='lime', marker='o', s=80, label='Bullish Divergence', 
                   zorder=5, edgecolors='darkgreen', linewidth=2)
    
    # Fill oversold and overbought areas
    ax2.fill_between(df.index, 0, 30, alpha=0.2, color='green', label='Oversold Zone')
    ax2.fill_between(df.index, 70, 100, alpha=0.2, color='red', label='Overbought Zone')
    
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Volume Bar Number')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution function
    """
    print("Loading TXF data for RSI Bullish Divergence Analysis...")
    
    # Load data
    csv_pattern = 'data/downtrend/Daily_*_cleaned.csv'
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
    data = create_volume_bars(df, volume_per_bar)
    
    print(f"Created {len(data)} volume bars")
    
    # Calculate RSI
    data = calculate_rsi(data, period=14)
    
    # Remove NaN values from RSI calculation
    data = data.dropna().reset_index(drop=True)
    
    print(f"Data after RSI calculation: {len(data)} bars")
    
    # Detect bullish divergence in last 25 bars
    lookback_window = 25
    print(f"Detecting bullish RSI divergence in last {lookback_window} bars...")
    data = detect_bullish_divergence(data, lookback_window=lookback_window)
    
    # Count divergence signals
    divergence_count = (data['bullish_divergence'] == True).sum()
    print(f"Bullish divergence signals found: {divergence_count}")
    
    # Show recent data with divergence signals
    print("\n=== RECENT DATA SUMMARY ===")
    recent_data = data.tail(lookback_window)
    print(f"Last {lookback_window} bars:")
    print(f"Price range: {recent_data['close'].min():.2f} - {recent_data['close'].max():.2f}")
    print(f"RSI range: {recent_data['rsi'].min():.2f} - {recent_data['rsi'].max():.2f}")
    
    if divergence_count > 0:
        divergence_bars = data[data['bullish_divergence'] == True]
        print(f"\nDivergence detected at bars: {divergence_bars.index.tolist()}")
        for idx, row in divergence_bars.iterrows():
            print(f"Bar {idx}: Price={row['close']:.2f}, RSI={row['rsi']:.2f}")
    
    # Plot the data
    plot_data_with_divergence(data, "TXF RSI Bullish Divergence Detection")
    
    # Save results
    output_file = 'rsi_divergence_results.csv'
    data.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return data

if __name__ == '__main__':
    result_df = main()