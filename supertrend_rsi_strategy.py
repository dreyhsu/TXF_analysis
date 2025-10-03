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

def calculate_rsi(df, period=14):
    """
    Calculate RSI indicator
    """
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=period)
    return df

def calculate_speed_metrics(df):
    """
    Calculate speed metrics similar to the RSI analysis
    """
    # Calculate duration and speed
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60  # in minutes
    df['duration'] = df['duration'].replace(0, 0.1)  # Avoid division by zero
    df['speed'] = 1 / df['duration']  # speed = 1/duration
    
    # Calculate speed ranks
    df['speed_rank_20'] = df['speed'].rolling(20, min_periods=1).rank(pct=True)
    df['speed_rank_50'] = df['speed'].rolling(50, min_periods=1).rank(pct=True)
    
    # Speed condition: fast enough (top 80% in recent 20 bars OR top 70% in recent 50 bars)
    df['speed_ok'] = (df['speed_rank_20'] >= 0.8) | (df['speed_rank_50'] >= 0.7)
    
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

def generate_rsi_signals(df):
    """
    Generate trading signals based on SuperTrend AND RSI conditions
    """
    # First get basic SuperTrend signals
    supertrend_signals = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i]:
            supertrend_signals.append(1)
        elif df['close'].iloc[i] < df['lowerband'].iloc[i]:
            supertrend_signals.append(-1)
        else:
            supertrend_signals.append(supertrend_signals[i-1])
    
    df['supertrend_signals'] = supertrend_signals
    
    # RSI entry conditions (based on analysis results)
    df['rsi_oversold'] = df['rsi'] < 30
    df['rsi_overbought'] = df['rsi'] > 70
    df['rsi_buy_entry'] = (df['rsi'].shift(1) < 30) & (df['rsi'] >= 30)
    
    # Individual condition columns for clarity
    df['con1_st_bullish'] = (df['supertrend_signals'] == 1) & (df['supertrend_signals'] != df['supertrend_signals'].shift(1))
    df['con2_st_bearish'] = (df['supertrend_signals'] == -1) & (df['supertrend_signals'] != df['supertrend_signals'].shift(1))
    df['con3_speed_ok'] = df['speed_ok']
    df['con4_rsi_buy_entry'] = df['rsi_buy_entry']
    df['con5_rsi_overbought'] = df['rsi_overbought']
    
    # Exit conditions
    df['exit_con1_st_reversal'] = df['supertrend_signals'] != df['supertrend_signals'].shift(1)
    df['exit_con2_rsi_oversold'] = df['rsi_oversold']
    df['exit_con3_rsi_overbought'] = df['rsi_overbought']
    
    # Generate final signals combining conditions
    signals = []
    buy_signals = []
    sell_signals = []
    exit_long = []
    exit_short = []
    
    for i in range(len(df)):
        if i == 0:
            signals.append(0)
            buy_signals.append(False)
            sell_signals.append(False)
            exit_long.append(False)
            exit_short.append(False)
            continue
            
        # Entry conditions
        # Buy: con1 (ST bullish) + con3 (speed ok) + con4 (RSI buy entry)
        con1 = df['con1_st_bullish'].iloc[i]
        con3 = df['con3_speed_ok'].iloc[i] if not pd.isna(df['con3_speed_ok'].iloc[i]) else False
        con4 = df['con4_rsi_buy_entry'].iloc[i]
        
        # Sell: con2 (ST bearish) + con3 (speed ok) + con5 (RSI overbought)
        # con2 = df['con2_st_bearish'].iloc[i]
        # con5 = df['con5_rsi_overbought'].iloc[i]
        
        # Exit conditions
        exit_con1 = df['exit_con1_st_reversal'].iloc[i]
        exit_con2 = df['exit_con2_rsi_oversold'].iloc[i]
        exit_con3 = df['exit_con3_rsi_overbought'].iloc[i]
        
        prev_signal = signals[i-1] if i > 0 else 0
        
        # Buy signal
        # if con1 and con3 and con4:
        if con4 and con3:
            signals.append(1)
            buy_signals.append(True)
            sell_signals.append(False)
            exit_long.append(False)
            exit_short.append(False)
        # Sell signal
        # elif con2 and con3 and con5:
        #     signals.append(-1)
        #     buy_signals.append(False)
        #     sell_signals.append(True)
        #     exit_long.append(False)
        #     exit_short.append(False)
        # Exit long position
        elif prev_signal == 1 and (exit_con1 and df['supertrend_signals'].iloc[i] == -1):
            signals.append(0)
            buy_signals.append(False)
            sell_signals.append(False)
            exit_long.append(True)
            exit_short.append(False)
        # Exit short position
        # elif prev_signal == -1 and (exit_con1 and df['supertrend_signals'].iloc[i] == 1):
        #     signals.append(0)
        #     buy_signals.append(False)
        #     sell_signals.append(False)
        #     exit_long.append(False)
        #     exit_short.append(True)
        # Hold position
        else:
            signals.append(prev_signal)
            buy_signals.append(False)
            sell_signals.append(False)
            exit_long.append(False)
            exit_short.append(False)
    
    df['signals'] = signals
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['exit_long'] = exit_long
    df['exit_short'] = exit_short
    
    # Shift signals to remove look-ahead bias
    df['signals'] = df['signals'].shift(1).fillna(0)
    df['buy_signal'] = df['buy_signal'].shift(1).fillna(False)
    df['sell_signal'] = df['sell_signal'].shift(1).fillna(False)
    df['exit_long'] = df['exit_long'].shift(1).fillna(False)
    df['exit_short'] = df['exit_short'].shift(1).fillna(False)
    
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

    # Create position lists based on buy/sell signals
    buy_positions = []
    sell_positions = []

    for i in range(len(df)):
        if df['buy_signal'].iloc[i]:
            buy_positions.append(df['close'].iloc[i])
            sell_positions.append(np.nan)
        elif df['sell_signal'].iloc[i]:
            sell_positions.append(df['close'].iloc[i])
            buy_positions.append(np.nan)
        else:
            buy_positions.append(np.nan)
            sell_positions.append(np.nan)

    df['buy_positions'] = buy_positions
    df['sell_positions'] = sell_positions
    return df

def plot_data(df, title="SuperTrend + RSI Strategy"):
    """
    Plot SuperTrend + RSI data
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Main price plot
    ax1.plot(df.index, df['close'], color='black', linewidth=1.5, label='Close Price', zorder=3)
    
    # Plot SuperTrend bands
    ax1.plot(df.index, df['lowerband_plot'], color='green', linewidth=2, 
             label='SuperTrend Support (Bullish)', zorder=2)
    ax1.plot(df.index, df['upperband_plot'], color='red', linewidth=2,
             label='SuperTrend Resistance (Bearish)', zorder=2)
    
    # Plot buy/sell signals
    buy_indices = df.index[~np.isnan(df['buy_positions'])]
    sell_indices = df.index[~np.isnan(df['sell_positions'])]
    
    if len(buy_indices) > 0:
        ax1.scatter(buy_indices, df.loc[buy_indices, 'buy_positions'], 
                   color='#2cf651', marker='^', s=100, label='Buy Signal', zorder=5, 
                   edgecolors='white', linewidth=2)
    
    if len(sell_indices) > 0:
        ax1.scatter(sell_indices, df.loc[sell_indices, 'sell_positions'], 
                   color='#f50100', marker='v', s=100, label='Sell Signal', zorder=5, 
                   edgecolors='white', linewidth=2)
    
    ax1.set_title(f'{title} - Volume Bars with SuperTrend + RSI')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI subplot
    ax2.plot(df.index, df['rsi'], color='purple', linewidth=1.5, label='RSI')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    
    # Highlight RSI conditions
    oversold_mask = df['rsi_oversold']
    overbought_mask = df['rsi_overbought']
    
    if oversold_mask.any():
        ax2.scatter(df.index[oversold_mask], df['rsi'][oversold_mask], 
                   color='green', alpha=0.6, s=20, label='Oversold')
    if overbought_mask.any():
        ax2.scatter(df.index[overbought_mask], df['rsi'][overbought_mask], 
                   color='red', alpha=0.6, s=20, label='Overbought')
    
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Volume Bar Number')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
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
    print("=== SUPERTREND + RSI STRATEGY PERFORMANCE ===")
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
    plt.plot(strategy_df.index, strategy_df['cumulative_balance'], label='SuperTrend + RSI Strategy', linewidth=2)
    plt.title('SuperTrend + RSI Strategy Performance Curve')
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
    print("Loading TXF data for SuperTrend + RSI Volume Bar Analysis...")
    
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
    
    # Calculate speed metrics
    print("Calculating speed metrics...")
    data = calculate_speed_metrics(data)
    
    # Calculate RSI
    print("Calculating RSI...")
    data = calculate_rsi(data)
    
    # SuperTrend parameters
    volatility = 5
    
    # Apply supertrend formula
    print("Applying SuperTrend...")
    supertrend_data = supertrend(df=data, atr_multiplier=volatility)
    
    # Generate the RSI + SuperTrend signals
    print("Generating RSI + SuperTrend signals...")
    supertrend_positions = generate_rsi_signals(supertrend_data)
    
    # Generate the positions for plotting
    supertrend_positions = create_positions(supertrend_positions)
    
    # Count signals
    buy_signals = supertrend_positions['buy_signal'].sum()
    sell_signals = supertrend_positions['sell_signal'].sum()
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    # Calculate performance
    supertrend_df = strategy_performance(supertrend_positions, capital=100000, leverage=1)
    
    # Display strategy metrics summary  
    print("\n=== SUPERTREND + RSI METRICS SUMMARY ===")
    print(f"Average ATR: {supertrend_df['atr'].mean():.2f}")
    print(f"Average RSI: {supertrend_df['rsi'].mean():.2f}")
    print(f"Bullish signal bars: {(supertrend_df['signals'] == 1).sum()}")
    print(f"Bearish signal bars: {(supertrend_df['signals'] == -1).sum()}")
    print(f"Neutral signal bars: {(supertrend_df['signals'] == 0).sum()}")
    
    print("\n=== INDIVIDUAL CONDITIONS SUMMARY ===")
    print(f"con1 (SuperTrend Bullish): {supertrend_df['con1_st_bullish'].sum()}")
    print(f"con2 (SuperTrend Bearish): {supertrend_df['con2_st_bearish'].sum()}")
    print(f"con3 (Speed OK): {supertrend_df['con3_speed_ok'].sum()}")
    print(f"con4 (RSI Buy Entry): {supertrend_df['con4_rsi_buy_entry'].sum()}")
    print(f"con5 (RSI Overbought): {supertrend_df['con5_rsi_overbought'].sum()}")
    
    print("\n=== BUY/SELL CONDITIONS BREAKDOWN ===")
    buy_condition = supertrend_df['con1_st_bullish'] & supertrend_df['con3_speed_ok'] & supertrend_df['con4_rsi_buy_entry']
    sell_condition = supertrend_df['con2_st_bearish'] & supertrend_df['con3_speed_ok'] & supertrend_df['con5_rsi_overbought']
    print(f"Buy condition (con1 & con3 & con4): {buy_condition.sum()}")
    print(f"Sell condition (con2 & con3 & con5): {sell_condition.sum()}")
    
    print("\n=== EXIT CONDITIONS SUMMARY ===")
    print(f"Exit long signals: {supertrend_df['exit_long'].sum()}")
    print(f"Exit short signals: {supertrend_df['exit_short'].sum()}")
    
    print("\n=== RSI ANALYSIS ===")
    print(f"RSI oversold instances: {supertrend_df['rsi_oversold'].sum()}")
    print(f"RSI overbought instances: {supertrend_df['rsi_overbought'].sum()}")
    print(f"RSI buy entries: {supertrend_df['rsi_buy_entry'].sum()}")
    print(f"Speed OK instances: {supertrend_df['speed_ok'].sum()}")
    
    # Plot data
    plot_data(supertrend_positions, "TXF SuperTrend + RSI Strategy")
    
    # Plot performance curve
    plot_performance_curve(supertrend_df)
    
    # Save results
    output_file = 'supertrend_rsi_strategy_results.csv'
    supertrend_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return supertrend_df

if __name__ == '__main__':
    supertrend_df = main()