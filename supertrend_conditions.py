import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import glob
warnings.filterwarnings('ignore')

def create_volume_bars(df, volume_per_bar=1000):
    """
    Create volume bars from tick data
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
    
    # Set index for easier handling
    volume_bars.set_index('volume_bar', inplace=True)
    
    return volume_bars

def calculate_atr(df, period=15):
    """
    Calculate Average True Range manually using talib
    """
    df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
    df.dropna(inplace=True)
    return df

def supertrend(df, atr_multiplier=3):
    """
    Calculate SuperTrend indicator
    """
    # Calculate the Upper Band(UB) and the Lower Band(LB)
    current_average_high_low = (df['high'] + df['low']) / 2
    
    # Calculate ATR
    df = calculate_atr(df, period=15)
    
    df['basicUpperband'] = current_average_high_low + (atr_multiplier * df['atr'])
    df['basicLowerband'] = current_average_high_low - (atr_multiplier * df['atr'])
    
    first_upperBand_value = df['basicUpperband'].iloc[0]
    first_lowerBand_value = df['basicLowerband'].iloc[0]
    upperBand = [first_upperBand_value]
    lowerBand = [first_lowerBand_value]

    for i in range(1, len(df)):
        if df['basicUpperband'].iloc[i] < upperBand[i-1] or df['close'].iloc[i-1] > upperBand[i-1]:
            upperBand.append(df['basicUpperband'].iloc[i])
        else:
            upperBand.append(upperBand[i-1])

        if df['basicLowerband'].iloc[i] > lowerBand[i-1] or df['close'].iloc[i-1] < lowerBand[i-1]:
            lowerBand.append(df['basicLowerband'].iloc[i])
        else:
            lowerBand.append(lowerBand[i-1])

    df['upperband'] = upperBand
    df['lowerband'] = lowerBand
    df.drop(['basicUpperband', 'basicLowerband'], axis=1, inplace=True)
    return df

def generate_conditions_signals(df):
    """
    Generate trading signals with con1 (entry) and con2 (exit) conditions
    """
    # Initialize condition columns
    df['con1'] = 0  # Entry condition
    df['con2'] = 0  # Exit condition
    df['position'] = 0  # Current position: 1=long, -1=short, 0=neutral
    df['signals'] = 0  # Trading signals
    
    # Loop through the dataframe starting from index 1
    for i in range(1, len(df)):
        current_close = df['close'].iloc[i]
        prev_close = df['close'].iloc[i-1]
        current_upper = df['upperband'].iloc[i]
        current_lower = df['lowerband'].iloc[i]
        prev_upper = df['upperband'].iloc[i-1]
        prev_lower = df['lowerband'].iloc[i-1]
        prev_position = df['position'].iloc[i-1]
        
        # Entry Conditions (con1)
        # Long entry: price breaks above upperband
        long_entry = (prev_close <= prev_upper) and (current_close > current_upper)
        # Short entry: price breaks below lowerband  
        short_entry = (prev_close >= prev_lower) and (current_close < current_lower)
        
        # Exit Conditions (con2)
        # Long exit: price breaks below lowerband while in long position
        long_exit = (prev_position == 1) and (prev_close >= prev_lower) and (current_close < current_lower)
        # Short exit: price breaks above upperband while in short position
        short_exit = (prev_position == -1) and (prev_close <= prev_upper) and (current_close > current_upper)
        
        # Set conditions
        if long_entry:
            df.loc[i, 'con1'] = 1  # Long entry
            df.loc[i, 'position'] = 1
            df.loc[i, 'signals'] = 1
        elif short_entry:
            df.loc[i, 'con1'] = -1  # Short entry
            df.loc[i, 'position'] = -1
            df.loc[i, 'signals'] = -1
        elif long_exit:
            df.loc[i, 'con2'] = -1  # Long exit
            df.loc[i, 'position'] = 0
            df.loc[i, 'signals'] = 0
        elif short_exit:
            df.loc[i, 'con2'] = 1  # Short exit
            df.loc[i, 'position'] = 0
            df.loc[i, 'signals'] = 0
        else:
            # Continue previous position
            df.loc[i, 'position'] = prev_position
            df.loc[i, 'signals'] = prev_position
    
    # Shift signals to remove look-ahead bias
    df['signals'] = df['signals'].shift(1)
    df['position'] = df['position'].shift(1)
    
    return df

def create_entry_exit_positions(df):
    """
    Create position markers for entry and exit points
    """
    # Create copies for plotting
    df['upperband_plot'] = df['upperband'].copy()
    df['lowerband_plot'] = df['lowerband'].copy()
    
    # Hide bands based on current signal
    df.loc[df['signals'] == 1, 'upperband_plot'] = np.nan
    df.loc[df['signals'] == -1, 'lowerband_plot'] = np.nan

    # Create position lists for entry and exit
    long_entry_positions = []
    short_entry_positions = []
    long_exit_positions = []
    short_exit_positions = []

    for i in range(len(df)):
        # Entry positions (con1)
        if df['con1'].iloc[i] == 1:  # Long entry
            long_entry_positions.append(df['close'].iloc[i])
        else:
            long_entry_positions.append(np.nan)
            
        if df['con1'].iloc[i] == -1:  # Short entry
            short_entry_positions.append(df['close'].iloc[i])
        else:
            short_entry_positions.append(np.nan)
        
        # Exit positions (con2)
        if df['con2'].iloc[i] == -1:  # Long exit
            long_exit_positions.append(df['close'].iloc[i])
        else:
            long_exit_positions.append(np.nan)
            
        if df['con2'].iloc[i] == 1:  # Short exit
            short_exit_positions.append(df['close'].iloc[i])
        else:
            short_exit_positions.append(np.nan)

    # Add position columns
    df['long_entry'] = long_entry_positions
    df['short_entry'] = short_entry_positions
    df['long_exit'] = long_exit_positions
    df['short_exit'] = short_exit_positions
    
    return df

def plot_conditions_data(df, title="SuperTrend Conditions Strategy"):
    """
    Plot SuperTrend data with entry/exit conditions
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    
    # Plot price line
    ax1.plot(df.index, df['close'], color='black', linewidth=1.5, label='Close Price', zorder=3)
    
    # Plot SuperTrend bands
    ax1.plot(df.index, df['lowerband_plot'], color='green', linewidth=2, 
             label='SuperTrend Support (Bullish)', zorder=2)
    ax1.plot(df.index, df['upperband_plot'], color='red', linewidth=2,
             label='SuperTrend Resistance (Bearish)', zorder=2)
    
    # Plot entry signals (con1)
    long_entry_indices = df.index[~np.isnan(df['long_entry'])]
    short_entry_indices = df.index[~np.isnan(df['short_entry'])]
    
    if len(long_entry_indices) > 0:
        ax1.scatter(long_entry_indices, df.loc[long_entry_indices, 'long_entry'], 
                   color='#00ff00', marker='^', s=120, label='Long Entry (con1)', zorder=5, 
                   edgecolors='black', linewidth=2)
    
    if len(short_entry_indices) > 0:
        ax1.scatter(short_entry_indices, df.loc[short_entry_indices, 'short_entry'], 
                   color='#ff0000', marker='v', s=120, label='Short Entry (con1)', zorder=5, 
                   edgecolors='black', linewidth=2)
    
    # Plot exit signals (con2)
    long_exit_indices = df.index[~np.isnan(df['long_exit'])]
    short_exit_indices = df.index[~np.isnan(df['short_exit'])]
    
    if len(long_exit_indices) > 0:
        ax1.scatter(long_exit_indices, df.loc[long_exit_indices, 'long_exit'], 
                   color='#ffff00', marker='x', s=120, label='Long Exit (con2)', zorder=5, 
                   linewidth=3)
    
    if len(short_exit_indices) > 0:
        ax1.scatter(short_exit_indices, df.loc[short_exit_indices, 'short_exit'], 
                   color='#ff8c00', marker='x', s=120, label='Short Exit (con2)', zorder=5, 
                   linewidth=3)
    
    # Fill between price and SuperTrend bands
    bullish_mask = df['signals'] == 1
    bearish_mask = df['signals'] == -1
    
    if bullish_mask.any():
        ax1.fill_between(df.index[bullish_mask], 
                        df['close'][bullish_mask], 
                        df['lowerband'][bullish_mask],
                        alpha=0.2, color='green', interpolate=True)
    
    if bearish_mask.any():
        ax1.fill_between(df.index[bearish_mask], 
                        df['close'][bearish_mask], 
                        df['upperband'][bearish_mask],
                        alpha=0.2, color='red', interpolate=True)
    
    ax1.set_title(f'{title} - Volume Bars with Entry/Exit Conditions')
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Volume Bar Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def strategy_performance(strategy_df, capital=100000, leverage=1):
    """
    Calculate strategy performance metrics
    """
    # Initialize performance variables
    cumulative_balance = capital
    investment = capital
    pl = 0
    max_drawdown = 0
    max_drawdown_percentage = 0

    # Lists to store intermediate values
    balance_list = [capital]
    pnl_list = [0]
    investment_list = [capital]
    peak_balance = capital

    # Loop from the second row
    for index in range(1, len(strategy_df)):
        row = strategy_df.iloc[index]

        # Calculate P/L for each trade signal
        if row['signals'] == 1:
            pl = ((row['close'] - row['open']) / row['open']) * investment * leverage
        elif row['signals'] == -1:
            pl = ((row['open'] - row['close']) / row['close']) * investment * leverage
        else:
            pl = 0

        # Update investment on signal reversal
        if row['signals'] != strategy_df.iloc[index - 1]['signals']:
            investment = cumulative_balance

        # Calculate new balance
        cumulative_balance += pl

        # Update lists
        investment_list.append(investment)
        balance_list.append(cumulative_balance)
        pnl_list.append(pl)

        # Calculate max drawdown
        drawdown = cumulative_balance - peak_balance
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            max_drawdown_percentage = (max_drawdown / peak_balance) * 100

        # Update peak balance
        if cumulative_balance > peak_balance:
            peak_balance = cumulative_balance

    # Add columns to dataframe
    strategy_df['investment'] = investment_list
    strategy_df['cumulative_balance'] = balance_list
    strategy_df['pl'] = pnl_list
    strategy_df['cumPL'] = strategy_df['pl'].cumsum()

    # Calculate performance metrics
    overall_pl_percentage = (strategy_df['cumulative_balance'].iloc[-1] - capital) * 100 / capital
    overall_pl = strategy_df['cumulative_balance'].iloc[-1] - capital
    min_balance = min(strategy_df['cumulative_balance'])
    max_balance = max(strategy_df['cumulative_balance'])

    # Print performance metrics
    print("=== STRATEGY PERFORMANCE ===")
    print("Overall P/L: {:.2f}%".format(overall_pl_percentage))
    print("Overall P/L: {:.2f}".format(overall_pl))
    print("Min balance: {:.2f}".format(min_balance))
    print("Max balance: {:.2f}".format(max_balance))
    print("Maximum Drawdown: {:.2f}".format(max_drawdown))
    print("Maximum Drawdown %: {:.2f}%".format(max_drawdown_percentage))

    return strategy_df

def plot_performance_curve(strategy_df):
    """
    Plot the performance curve
    """
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df.index, strategy_df['cumulative_balance'], label='Strategy', linewidth=2)
    plt.title('SuperTrend Conditions Strategy Performance Curve')
    plt.xlabel('Volume Bar Number')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution function
    """
    print("Loading TXF data for SuperTrend Conditions Analysis...")
    
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
    data = create_volume_bars(df, volume_per_bar)
    
    print(f"Created {len(data)} volume bars")
    
    # SuperTrend parameters
    volatility = 3
    
    # Apply supertrend formula
    supertrend_data = supertrend(df=data, atr_multiplier=volatility)
    
    # Generate signals with conditions
    supertrend_positions = generate_conditions_signals(supertrend_data)
    
    # Generate entry/exit positions
    supertrend_positions = create_entry_exit_positions(supertrend_positions)
    
    # Count entry and exit signals
    long_entries = (~np.isnan(supertrend_positions['long_entry'])).sum()
    short_entries = (~np.isnan(supertrend_positions['short_entry'])).sum()
    long_exits = (~np.isnan(supertrend_positions['long_exit'])).sum()
    short_exits = (~np.isnan(supertrend_positions['short_exit'])).sum()
    
    print(f"\n=== ENTRY/EXIT SIGNALS ===")
    print(f"Long entries (con1=1): {long_entries}")
    print(f"Short entries (con1=-1): {short_entries}")
    print(f"Long exits (con2=-1): {long_exits}")
    print(f"Short exits (con2=1): {short_exits}")
    print(f"Total entry signals: {long_entries + short_entries}")
    print(f"Total exit signals: {long_exits + short_exits}")
    
    # Calculate performance
    supertrend_df = strategy_performance(supertrend_positions, capital=100000, leverage=1)
    
    # Display SuperTrend metrics summary  
    print("\n=== SUPERTREND METRICS SUMMARY ===")
    print(f"Average ATR: {supertrend_df['atr'].mean():.2f}")
    print(f"Long position bars: {(supertrend_df['signals'] == 1).sum()}")
    print(f"Short position bars: {(supertrend_df['signals'] == -1).sum()}")
    print(f"Neutral position bars: {(supertrend_df['signals'] == 0).sum()}")
    print(f"Long percentage: {(supertrend_df['signals'] == 1).sum()/len(supertrend_df)*100:.1f}%")
    print(f"Short percentage: {(supertrend_df['signals'] == -1).sum()/len(supertrend_df)*100:.1f}%")
    
    # Plot data with conditions
    plot_conditions_data(supertrend_positions, "TXF SuperTrend Conditions Strategy")
    
    # Plot performance curve
    plot_performance_curve(supertrend_df)
    
    # Save results
    output_file = 'supertrend_conditions_results.csv'
    supertrend_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return supertrend_df

if __name__ == '__main__':
    supertrend_df = main()