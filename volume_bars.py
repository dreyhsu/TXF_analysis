import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    
    return volume_bars

def plot_volume_bars(volume_bars, title="Volume Bars Chart"):
    """
    Plot volume bars as candlestick chart
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Price chart
    for i, row in volume_bars.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Draw the bar (high-low line)
        ax1.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
        
        # Draw the body (open-close rectangle)
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        ax1.bar(i, body_height, bottom=body_bottom, color=color, alpha=0.7, width=0.8)
    
    ax1.set_title(f'{title} - Price')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    ax2.bar(range(len(volume_bars)), volume_bars['volume'], color='blue', alpha=0.7)
    ax2.set_title('Volume')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Volume Bar Number')
    ax2.grid(True, alpha=0.3)
    
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