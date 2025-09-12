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

def generate_signals(df):
    """
    Generate trading signals based on SuperTrend
    """
    # Initiate a signals list
    signals = [0]

    # Loop through the dataframe
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i]:
            signals.append(1)
        elif df['close'].iloc[i] < df['lowerband'].iloc[i]:
            signals.append(-1)
        else:
            signals.append(signals[i-1])

    # Add the signals list as a new column in the dataframe
    df['signals'] = signals
    df['signals'] = df["signals"].shift(1)  # Remove look ahead bias
    return df

def create_positions(df):
    """
    Create position markers for visualization
    """
    # Create copies for plotting (hide bands when not in use)
    df['upperband_plot'] = df['upperband'].copy()
    df['lowerband_plot'] = df['lowerband'].copy()
    
    # Hide upperband when signal is bullish (1), show lowerband
    df.loc[df['signals'] == 1, 'upperband_plot'] = np.nan
    # Hide lowerband when signal is bearish (-1), show upperband  
    df.loc[df['signals'] == -1, 'lowerband_plot'] = np.nan

    # Create position lists
    buy_positions = [np.nan]
    sell_positions = [np.nan]

    # Loop through the dataframe
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

    # Add the positions list as a new column in the dataframe
    df['buy_positions'] = buy_positions
    df['sell_positions'] = sell_positions
    return df

def plot_data(df, title="SuperTrend Strategy"):
    """
    Plot SuperTrend data using matplotlib instead of mplfinance
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    
    # Plot price line
    ax1.plot(df.index, df['close'], color='black', linewidth=1.5, label='Close Price', zorder=3)
    
    # Plot SuperTrend bands (only show active band based on signal)
    ax1.plot(df.index, df['lowerband_plot'], color='green', linewidth=2, 
             label='SuperTrend Support (Bullish)', zorder=2)
    ax1.plot(df.index, df['upperband_plot'], color='red', linewidth=2,
             label='SuperTrend Resistance (Bearish)', zorder=2)
    
    # Plot buy/sell signals
    buy_indices = df.index[~np.isnan(df['buy_positions'])]
    sell_indices = df.index[~np.isnan(df['sell_positions'])]
    
    # if len(buy_indices) > 0:
    #     ax1.scatter(buy_indices, df.loc[buy_indices, 'buy_positions'], 
    #                color='#2cf651', marker='^', s=100, label='Buy Signal', zorder=5, 
    #                edgecolors='white', linewidth=2)
    
    # if len(sell_indices) > 0:
    #     ax1.scatter(sell_indices, df.loc[sell_indices, 'sell_positions'], 
    #                color='#f50100', marker='v', s=100, label='Sell Signal', zorder=5, 
    #                edgecolors='white', linewidth=2)
    
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
    
    ax1.set_title(f'{title} - Volume Bars with SuperTrend')
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
    plt.title('SuperTrend Strategy Performance Curve')
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
    
    # Create volume bars
    volume_per_bar = 1000
    print(f"Creating volume bars with {volume_per_bar} volume per bar...")
    data = create_volume_bars(df, volume_per_bar)
    
    print(f"Created {len(data)} volume bars")
    
    # SuperTrend parameters
    volatility = 5
    
    # Apply supertrend formula
    supertrend_data = supertrend(df=data, atr_multiplier=volatility)
    
    # Generate the signals
    supertrend_positions = generate_signals(supertrend_data)
    
    # Generate the positions
    supertrend_positions = create_positions(supertrend_positions)
    
    # Count signals
    buy_signals = (~np.isnan(supertrend_positions['buy_positions'])).sum()
    sell_signals = (~np.isnan(supertrend_positions['sell_positions'])).sum()
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    # Calculate performance
    supertrend_df = strategy_performance(supertrend_positions, capital=100000, leverage=1)
    
    # Display SuperTrend metrics summary  
    print("\n=== SUPERTREND METRICS SUMMARY ===")
    print(f"Average ATR: {supertrend_df['atr'].mean():.2f}")
    print(f"Bullish signal bars: {(supertrend_df['signals'] == 1).sum()}")
    print(f"Bearish signal bars: {(supertrend_df['signals'] == -1).sum()}")
    print(f"Neutral signal bars: {(supertrend_df['signals'] == 0).sum()}")
    print(f"Bullish percentage: {(supertrend_df['signals'] == 1).sum()/len(supertrend_df)*100:.1f}%")
    print(f"Bearish percentage: {(supertrend_df['signals'] == -1).sum()/len(supertrend_df)*100:.1f}%")
    
    # Plot data
    plot_data(supertrend_positions, "TXF SuperTrend Strategy")
    
    # Plot performance curve
    plot_performance_curve(supertrend_df)
    
    # Save results
    output_file = 'supertrend_modified_results.csv'
    supertrend_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return supertrend_df

if __name__ == '__main__':
    supertrend_df = main()