import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import vectorbt as vbt
warnings.filterwarnings('ignore')

def calculate_atr(df, period=15):
    """
    Calculate Average True Range using talib
    """
    df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
    df.dropna(inplace=True)
    return df

def calculate_rsi(df, period=14):
    """
    Calculate RSI (Relative Strength Index) using talib
    """
    df['rsi'] = talib.RSI(df['close'].values, timeperiod=period)
    return df

def supertrend(df, atr_multiplier=3):
    """
    Calculate SuperTrend indicator
    """
    current_average_high_low = (df['high'] + df['low']) / 2
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

def generate_signals(df, long_timeframe_uptrend=None):
    """
    Generate trading signals based on RSI recovery and SuperTrend exit
    Returns entries and exits as boolean arrays

    Args:
        df: DataFrame with SuperTrend and RSI calculated
        long_timeframe_uptrend: Series indicating long timeframe uptrend periods (optional)
    """
    # Calculate SuperTrend signals for exit only
    signals = [0]

    for i in range(1, len(df)):
        current_signal = signals[i-1]

        if df['close'].iloc[i] > df['upperband'].iloc[i]:
            signals.append(1)  # Bullish
        elif df['close'].iloc[i] < df['lowerband'].iloc[i]:
            signals.append(-1)  # Bearish
        else:
            signals.append(current_signal)

    df['signals'] = signals

    # Generate entry signals based on RSI recovery (from below 40 to above 50)
    rsi_recovery = []
    lookback = 20  # Look back up to 10 periods for RSI < 40

    for i in range(len(df)):
        if i == 0:
            rsi_recovery.append(False)
            continue

        current_rsi = df['rsi'].iloc[i]
        prev_rsi = df['rsi'].iloc[i-1]

        # Check if RSI was recently below 40
        was_below_40 = False
        for j in range(1, min(lookback + 1, i + 1)):
            if df['rsi'].iloc[i - j] < 40:
                was_below_40 = True
                break

        # Signal when RSI crosses above 50 after being below 40
        # if was_below_40 and prev_rsi <= 50 and current_rsi > 50:
        if was_below_40:
            rsi_recovery.append(True)
        else:
            rsi_recovery.append(False)

    entries = pd.Series(rsi_recovery, index=df.index)

    # Exit: SuperTrend changes from bullish (1) to bearish (-1)
    exits = (df['signals'] == -1) & (df['signals'].shift(1) == 1)

    # Filter entries by long timeframe uptrend if provided
    if long_timeframe_uptrend is not None:
        print(f"Original RSI recovery entries: {entries.sum()}")
        entries = entries & long_timeframe_uptrend
        print(f"Filtered entries (with long TF filter): {entries.sum()}")

    return entries, exits

def load_long_timeframe_uptrend(csv_file, atr_multiplier=3):
    """
    Load long timeframe data and calculate uptrend periods
    Returns a Series indicating uptrend periods
    """
    print(f"\nLoading long timeframe data from {csv_file}...")
    df_long = pd.read_csv(csv_file)

    # Convert timestamps to datetime and set as index
    df_long['datetime'] = pd.to_datetime(df_long['open_time'], unit='ms')
    df_long.set_index('datetime', inplace=True)

    print(f"Loaded {len(df_long)} bars (long timeframe)")
    print(f"Long TF range: {df_long.index[0]} to {df_long.index[-1]}")

    # Calculate SuperTrend on long timeframe
    df_long = supertrend(df_long, atr_multiplier=atr_multiplier)

    # Calculate signals
    signals = [0]
    for i in range(1, len(df_long)):
        current_signal = signals[i-1]
        if df_long['close'].iloc[i] > df_long['upperband'].iloc[i]:
            signals.append(1)
        elif df_long['close'].iloc[i] < df_long['lowerband'].iloc[i]:
            signals.append(-1)
        else:
            signals.append(current_signal)

    df_long['signals'] = signals

    # Uptrend periods are when signal == 1
    df_long['uptrend'] = df_long['signals'] == 1

    uptrend_periods = (df_long['uptrend'] == True).sum()
    uptrend_pct = (uptrend_periods / len(df_long)) * 100
    print(f"Long TF uptrend periods: {uptrend_periods} ({uptrend_pct:.1f}%)")

    return df_long[['uptrend']]

def merge_long_timeframe_to_short(df_short, df_long_uptrend):
    """
    Merge long timeframe uptrend periods to short timeframe
    Uses forward fill to propagate long TF signals to short TF bars
    """
    # Merge and forward fill
    df_merged = df_short.join(df_long_uptrend, how='left')
    df_merged['uptrend'] = df_merged['uptrend'].ffill().fillna(False)

    return df_merged['uptrend']

def main():
    """
    Main execution function
    """
    # === CONFIGURATION ===
    USE_LONG_TIMEFRAME_FILTER = True  # Set to False to disable long TF filter
    short_tf_file = 'binance_tick/BTCUSDT-1h-2025-08.csv'
    long_tf_file = 'binance_tick/BTCUSDT-4h-2025-08.csv'
    atr_multiplier = 3

    print("="*60)
    print("MULTI-TIMEFRAME SUPERTREND BACKTEST")
    print("="*60)
    print(f"Long timeframe filter: {'ENABLED' if USE_LONG_TIMEFRAME_FILTER else 'DISABLED'}")
    print(f"Short timeframe: {short_tf_file}")
    print(f"Long timeframe: {long_tf_file}")
    print("="*60)

    # Load short timeframe data
    print(f"\nLoading short timeframe data...")
    df = pd.read_csv(short_tf_file)

    # Convert timestamps to datetime and set as index
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('datetime', inplace=True)

    print(f"Loaded {len(df)} bars (short timeframe)")
    print(f"Short TF range: {df.index[0]} to {df.index[-1]}")

    # Calculate SuperTrend on short timeframe
    print("\nCalculating SuperTrend (short timeframe)...")
    df = supertrend(df, atr_multiplier=atr_multiplier)

    # Calculate RSI (optional, for analysis)
    df = calculate_rsi(df, period=14)

    # Load and merge long timeframe uptrend if enabled
    long_timeframe_uptrend = None
    if USE_LONG_TIMEFRAME_FILTER:
        df_long_uptrend = load_long_timeframe_uptrend(long_tf_file, atr_multiplier=atr_multiplier)
        long_timeframe_uptrend = merge_long_timeframe_to_short(df, df_long_uptrend)

        # Add to dataframe for analysis
        df['long_tf_uptrend'] = long_timeframe_uptrend

        long_uptrend_bars = long_timeframe_uptrend.sum()
        long_uptrend_pct = (long_uptrend_bars / len(df)) * 100
        print(f"Short TF bars in long TF uptrend: {long_uptrend_bars} ({long_uptrend_pct:.1f}%)")

    # Generate signals
    print("\nGenerating signals...")
    entries, exits = generate_signals(df, long_timeframe_uptrend)

    print(f"Entry signals: {entries.sum()}")
    print(f"Exit signals: {exits.sum()}")

    # Backtest parameters
    initial_capital = 100000
    fees = 0.0004  # 0.04% per trade
    slippage = 0.0001  # 0.01% slippage

    print(f"\nRunning vectorbt backtest...")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Fees: {fees*100:.2f}%")
    print(f"Slippage: {slippage*100:.2f}%")

    # Create portfolio using vectorbt
    portfolio = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=entries,
        exits=exits,
        init_cash=initial_capital,
        fees=fees,
        slippage=slippage,
        freq='1h'  # Change to '1m' if using 1-minute data
    )

    # Print statistics
    print("\n" + "="*60)
    print("SUPERTREND LONG-ONLY BACKTEST RESULTS (VectorBT)")
    print("="*60)

    stats = portfolio.stats()
    print(f"\n{stats}")

    # Additional custom metrics
    print("\n" + "="*60)
    print("DETAILED METRICS")
    print("="*60)

    print(f"\n💰 EQUITY")
    print(f"Start Value:            ${portfolio.init_cash:,.2f}")
    print(f"End Value:              ${portfolio.final_value():,.2f}")
    print(f"Total Return:           {portfolio.total_return() * 100:.2f}%")
    print(f"Total Profit:           ${portfolio.final_value() - portfolio.init_cash:,.2f}")

    print(f"\n📉 RISK METRICS")
    print(f"Max Drawdown:           {portfolio.max_drawdown() * 100:.2f}%")
    print(f"Sharpe Ratio:           {portfolio.sharpe_ratio():.2f}")
    print(f"Calmar Ratio:           {portfolio.calmar_ratio():.2f}")

    print(f"\n📊 TRADE STATISTICS")
    print(f"Total Trades:           {portfolio.trades.count()}")
    print(f"Win Rate:               {portfolio.trades.win_rate() * 100:.2f}%")
    print(f"Profit Factor:          {portfolio.trades.profit_factor():.2f}")
    print(f"Average Win:            ${portfolio.trades.winning.pnl.mean():.2f}")
    print(f"Average Loss:           ${portfolio.trades.losing.pnl.mean():.2f}")
    print(f"Best Trade:             ${portfolio.trades.pnl.max():.2f}")
    print(f"Worst Trade:            ${portfolio.trades.pnl.min():.2f}")
    print(f"Avg Trade Duration:     {portfolio.trades.duration.mean()}")

    # Trade list
    print(f"\n📋 TRADE LIST")
    print("-" * 140)
    trades_df = portfolio.trades.records_readable
    if len(trades_df) > 0:
        # Print available columns to see what's there
        print(f"Available columns: {trades_df.columns.tolist()}")
        print(trades_df.to_string())
    else:
        print("No trades executed")

    # Plot results using VectorBT's built-in plotting
    print("\n\nGenerating plots...")

    try:
        # Main portfolio overview plot
        fig = portfolio.plot(
            subplots=[
                'orders',
                'trade_pnl',
                'cum_returns',
                'drawdowns'
            ]
        )
        fig.show()

        # Plot with asset value and trades overlay
        fig2 = portfolio.plot(subplots='all')
        fig2.show()

    except ImportError as e:
        print(f"\nWarning: VectorBT plotting requires additional dependencies.")
        print(f"Error: {e}")
        print("\nTo fix, install: pip install anywidget")
        print("\nFalling back to basic matplotlib plot...")

        # Fallback matplotlib plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Price and signals
        axes[0].plot(df.index, df['close'], color='black', linewidth=1.5, label='Close Price')
        axes[0].plot(df.index, df['lowerband'], color='green', linewidth=1.5, alpha=0.7, label='Lower Band')
        axes[0].plot(df.index, df['upperband'], color='red', linewidth=1.5, alpha=0.7, label='Upper Band')

        # Mark entries and exits
        entry_idx = df.index[entries]
        exit_idx = df.index[exits]

        if len(entry_idx) > 0:
            axes[0].scatter(entry_idx, df.loc[entry_idx, 'close'], color='green', marker='^',
                           s=100, label='Buy', zorder=5, edgecolors='white', linewidth=2)
        if len(exit_idx) > 0:
            axes[0].scatter(exit_idx, df.loc[exit_idx, 'close'], color='red', marker='v',
                           s=100, label='Sell', zorder=5, edgecolors='white', linewidth=2)

        # Shade long timeframe uptrend periods
        if USE_LONG_TIMEFRAME_FILTER and 'long_tf_uptrend' in df.columns:
            uptrend_mask = df['long_tf_uptrend']
            axes[0].fill_between(df.index, df['close'].min(), df['close'].max(),
                                where=uptrend_mask, alpha=0.1, color='blue',
                                label='Long TF Uptrend')

        axes[0].set_ylabel('Price ($)', fontsize=11)
        axes[0].set_title(f'SuperTrend {"Multi-TF" if USE_LONG_TIMEFRAME_FILTER else "Single-TF"} Strategy',
                         fontsize=14, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Equity curve
        axes[1].plot(portfolio.value(), linewidth=2, color='blue', label='Portfolio Value')
        axes[1].fill_between(df.index, portfolio.value(), alpha=0.3, color='blue')
        axes[1].set_ylabel('Portfolio Value ($)', fontsize=11)
        axes[1].set_xlabel('Date', fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Save results
    output_file = 'binance_tick/backtest_results_vbt.csv'
    results_df = pd.DataFrame({
        'datetime': df.index,
        'close': df['close'].values,
        'signals': df['signals'].values,
        'entries': entries.values,
        'exits': exits.values,
        'portfolio_value': portfolio.value().values,
        'returns': portfolio.returns().values
    })

    # Add long timeframe uptrend if available
    if 'long_tf_uptrend' in df.columns:
        results_df['long_tf_uptrend'] = df['long_tf_uptrend'].values

    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save trade list
    if len(trades_df) > 0:
        trades_output = 'binance_tick/trades_vbt.csv'
        trades_df.to_csv(trades_output, index=False)
        print(f"Trade list saved to: {trades_output}")

    return portfolio, df

if __name__ == '__main__':
    portfolio, df = main()
