import warnings
import pandas as pd
import glob
import argparse
warnings.filterwarnings('ignore')

# Configuration for different data sources
DATA_CONFIGS = {
    'TXF': {
        'pattern': './data/Daily_*_cleaned.csv',
        'datetime_cols': ['成交日期', '成交時間'],
        'price_col': '成交價格',
        'volume_col': '成交數量(B+S)',
        'volume_per_bar': 9000
    },
    'BTC': {
        'pattern': './binance_tick/BTCUSD_250926-trades-2025-08*.csv',
        'datetime_cols': ['time'],  # Single datetime column
        'price_col': 'price',
        'volume_col': 'base_qty',  # USDT value (price × qty)
        'volume_per_bar': 100000  # 100,000 USDT per volume bar
    }
}

def create_volume_bars(df, config, volume_per_bar=None):
    """
    Create volume bars from tick data

    Parameters:
    -----------
    df : DataFrame
        Raw tick data
    config : dict
        Configuration dict containing column names
    volume_per_bar : int or float, optional
        Override the default volume per bar from config
    """
    # Use override or default from config
    if volume_per_bar is None:
        volume_per_bar = config['volume_per_bar']

    # Combine date and time to create proper datetime
    if len(config['datetime_cols']) == 2:
        df['datetime'] = pd.to_datetime(df[config['datetime_cols'][0]] + ' ' + df[config['datetime_cols'][1]])
    else:
        # If only one datetime column
        datetime_col = config['datetime_cols'][0]
        # Check if the column contains numeric values (milliseconds timestamp)
        if pd.api.types.is_numeric_dtype(df[datetime_col]):
            df['datetime'] = pd.to_datetime(df[datetime_col], unit='ms')
        else:
            df['datetime'] = pd.to_datetime(df[datetime_col])

    df = df.sort_values('datetime').reset_index(drop=True)

    # Calculate cumulative volume
    df['cumulative_volume'] = df[config['volume_col']].cumsum()
    df['volume_bar'] = (df['cumulative_volume'] // volume_per_bar).astype(int)

    # Group by volume bar and create OHLCV data
    volume_bars = df.groupby('volume_bar').agg({
        'datetime': ['first', 'last'],
        config['price_col']: ['first', 'max', 'min', 'last'],
        config['volume_col']: ['sum', 'count']
    }).reset_index()

    # Flatten column names
    volume_bars.columns = ['volume_bar', 'start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume', 'trades']

    # Set index for easier handling
    volume_bars.set_index('volume_bar', inplace=True)

    return volume_bars

def main(asset='TXF', volume_per_bar=None):
    """
    Main execution function for creating volume bars

    Parameters:
    -----------
    asset : str
        Asset type ('TXF' or 'BTC')
    volume_per_bar : int or float, optional
        Override the default volume per bar
        For TXF: integer (e.g., 9000 contracts)
        For BTC: float (e.g., 100000 USDT)
    """
    asset = asset.upper()

    if asset not in DATA_CONFIGS:
        raise ValueError(f"Unknown asset type: {asset}. Available: {list(DATA_CONFIGS.keys())}")

    config = DATA_CONFIGS[asset]

    print(f"Loading {asset} data for Volume Bar Creation...")

    # Load data
    csv_pattern = config['pattern']
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {csv_pattern}")

    print(f"Found {len(csv_files)} file(s) matching pattern: {csv_pattern}")

    # Load and combine data
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} total records")

    # Create volume bars
    volume_bars = create_volume_bars(df, config, volume_per_bar)

    # Use actual volume per bar for filename
    actual_volume_per_bar = volume_per_bar if volume_per_bar else config['volume_per_bar']

    print(f"Created {len(volume_bars)} volume bars")

    # Save volume bars to CSV with volume_per_bar in filename
    # Format volume_per_bar - remove .0 for whole numbers
    volume_str = f'{actual_volume_per_bar:g}'
    output_file = f'supertrend/volume_bars_{asset.lower()}_{volume_str}.csv'
    volume_bars.to_csv(output_file, index=True)
    print(f"Volume bars saved to: {output_file}")

    # Display summary statistics
    print(f"\n=== {asset} VOLUME BARS SUMMARY ===")
    print(f"Total volume bars: {len(volume_bars)}")
    print(f"Average trades per bar: {volume_bars['trades'].mean():.2f}")
    print(f"Min trades per bar: {volume_bars['trades'].min()}")
    print(f"Max trades per bar: {volume_bars['trades'].max()}")
    print(f"Average volume per bar: {volume_bars['volume'].mean():.2f}")
    print(f"Date range: {volume_bars['start_time'].min()} to {volume_bars['end_time'].max()}")

    return volume_bars

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create volume bars from tick data')
    parser.add_argument('--asset', type=str, default='TXF',
                        help='Asset type (TXF or BTC)')
    parser.add_argument('--volume', type=float, default=None,
                        help='Volume per bar (overrides default). For TXF: contracts (e.g., 9000), For BTC: USDT value (e.g., 100000)')

    args = parser.parse_args()

    volume_bars = main(asset=args.asset, volume_per_bar=args.volume)
