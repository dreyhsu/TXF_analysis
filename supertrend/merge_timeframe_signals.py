import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def load_timeframe_data():
    """
    Load both long and short timeframe CSV files
    """
    try:
        long_df = pd.read_csv('supertrend_positions_long.csv')
        short_df = pd.read_csv('supertrend_positions_short.csv')

        print(f"Long timeframe data: {len(long_df)} rows")
        print(f"Short timeframe data: {len(short_df)} rows")

        return long_df, short_df
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return None, None

def match_timeframes(long_df, short_df):
    """
    Match short timeframe periods with long timeframe trends
    """
    # Convert time columns to datetime
    long_df['start_time'] = pd.to_datetime(long_df['start_time'])
    long_df['end_time'] = pd.to_datetime(long_df['end_time'])
    short_df['start_time'] = pd.to_datetime(short_df['start_time'])
    short_df['end_time'] = pd.to_datetime(short_df['end_time'])

    # Sort both dataframes by start_time for efficient processing
    long_df = long_df.sort_values('start_time').reset_index(drop=True)
    short_df = short_df.sort_values('start_time').reset_index(drop=True)

    # Initialize result columns
    short_df['long_signals'] = np.nan

    print("Matching timeframes...")

    # For each row in short timeframe, find matching long timeframe signal
    for short_idx, short_row in short_df.iterrows():
        short_start_time = short_row['start_time']

        # Find matching period in long timeframe
        matching_long = long_df[
            (long_df['start_time'] <= short_start_time) &
            (long_df['end_time'] >= short_start_time)
        ]

        if not matching_long.empty:
            # Take the first match (should be unique)
            long_signal = matching_long.iloc[0]['signals']
            short_df.at[short_idx, 'long_signals'] = long_signal

        # Progress indicator
        if short_idx % 100 == 0:
            print(f"Processed {short_idx}/{len(short_df)} rows...")

    return short_df

def generate_trading_signals(df):
    """
    Generate entry and exit signals based on conditions
    """
    # Initialize signal columns
    df['entry_signal'] = False
    df['exit_signal'] = False
    df['position'] = 0  # 0 = no position, 1 = long position

    current_position = 0
    prev_short_signal = np.nan

    for idx, row in df.iterrows():
        # Current signals
        has_rsi_recovery = row.get('rsi_recovery', False)
        long_bullish = row.get('long_signals', np.nan) == 1
        current_short_signal = row.get('signals', np.nan)

        # Track signal transitions
        short_signal_changed = (not pd.isna(prev_short_signal) and
                               not pd.isna(current_short_signal) and
                               prev_short_signal != current_short_signal)

        short_turned_bearish = (short_signal_changed and prev_short_signal == 1 and current_short_signal == -1)

        # Entry logic: Enter immediately when conditions are met (regardless of short signal)
        if (has_rsi_recovery and long_bullish and current_position == 0):
            df.at[idx, 'entry_signal'] = True
            current_position = 1

        # Exit logic: when short signal changes from 1 to -1
        if (short_turned_bearish and current_position == 1):
            df.at[idx, 'exit_signal'] = True
            current_position = 0

        # Update tracking variables
        df.at[idx, 'position'] = current_position
        prev_short_signal = current_short_signal

    return df

def display_verification_samples(df):
    """
    Display sample signals using short timeframe data for verification
    """
    # Entry signals
    entry_signals = df[df['entry_signal'] == True]
    print(f"\n=== ENTRY SIGNALS ({len(entry_signals)} total) ===")
    if not entry_signals.empty:
        print("Sample entry signals (short timeframe data):")
        sample_cols = ['start_time', 'end_time', 'signals', 'rsi_recovery', 'long_signals', 'entry_signal']
        available_cols = [col for col in sample_cols if col in entry_signals.columns]
        print(entry_signals[available_cols].head(10))

    # Exit signals
    exit_signals = df[df['exit_signal'] == True]
    print(f"\n=== EXIT SIGNALS ({len(exit_signals)} total) ===")
    if not exit_signals.empty:
        print("Sample exit signals (short timeframe data):")
        sample_cols = ['start_time', 'end_time', 'signals', 'long_signals', 'exit_signal']
        available_cols = [col for col in sample_cols if col in exit_signals.columns]
        print(exit_signals[available_cols].head(10))

    # Position summary
    position_changes = df[df['position'].diff() != 0]
    print(f"\n=== POSITION CHANGES ({len(position_changes)} total) ===")
    if not position_changes.empty:
        print("Position change timeline:")
        summary_cols = ['start_time', 'position', 'entry_signal', 'exit_signal', 'signals']
        available_cols = [col for col in summary_cols if col in position_changes.columns]
        print(position_changes[available_cols].head(15))

def visualize_trading_signals(df):
    """
    Visualize trading signals with price chart and position timeline
    """
    # Check if we have the required columns
    required_cols = ['start_time', 'close']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing required columns for visualization")
        return

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12),
                                        gridspec_kw={'height_ratios': [3, 1, 1]})

    # Convert start_time to datetime if not already
    df['start_time'] = pd.to_datetime(df['start_time'])

    # === TOP SUBPLOT: Price Chart with Signals ===
    ax1.plot(df['start_time'], df['close'], color='black', linewidth=1.5,
             label='Close Price', alpha=0.8)

    # Plot entry signals
    entry_signals = df[df['entry_signal'] == True]
    if not entry_signals.empty:
        ax1.scatter(entry_signals['start_time'], entry_signals['close'],
                   color='green', marker='^', s=100, label='Entry Signal',
                   zorder=5, edgecolors='white', linewidth=2)

    # Plot exit signals
    exit_signals = df[df['exit_signal'] == True]
    if not exit_signals.empty:
        ax1.scatter(exit_signals['start_time'], exit_signals['close'],
                   color='red', marker='v', s=100, label='Exit Signal',
                   zorder=5, edgecolors='white', linewidth=2)

    # Plot RSI recovery points
    if 'rsi_recovery' in df.columns:
        rsi_recovery = df[df['rsi_recovery'] == True]
        if not rsi_recovery.empty:
            ax1.scatter(rsi_recovery['start_time'], rsi_recovery['close'],
                       color='orange', marker='o', s=60, label='RSI Recovery',
                       zorder=4, edgecolors='black', linewidth=1, alpha=0.7)

    # Highlight position periods
    if 'position' in df.columns:
        position_periods = df[df['position'] == 1]
        if not position_periods.empty:
            for idx, row in position_periods.iterrows():
                ax1.axvspan(row['start_time'], row['start_time'],
                           alpha=0.1, color='green', zorder=1)

    ax1.set_title('Multi-Timeframe Trading Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # === MIDDLE SUBPLOT: Position Timeline ===
    if 'position' in df.columns:
        ax2.plot(df['start_time'], df['position'], color='blue', linewidth=2,
                 drawstyle='steps-post', label='Position')
        ax2.fill_between(df['start_time'], 0, df['position'],
                        alpha=0.3, color='blue', step='post')
        ax2.set_ylabel('Position', fontsize=12)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No Position', 'Long Position'])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

    # === BOTTOM SUBPLOT: Signal Indicators ===
    # Short timeframe signals
    if 'signals' in df.columns:
        short_signals = df['signals'].fillna(0)
        ax3.plot(df['start_time'], short_signals, color='purple', linewidth=1.5,
                 label='Short TF Signals', alpha=0.8)

    # Long timeframe signals
    if 'long_signals' in df.columns:
        long_signals = df['long_signals'].fillna(0)
        ax3.plot(df['start_time'], long_signals, color='orange', linewidth=1.5,
                 label='Long TF Signals', alpha=0.8)

    ax3.set_ylabel('Signal Value', fontsize=12)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Bearish (-1)', 'Neutral (0)', 'Bullish (1)'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')

    # Format x-axis for bottom subplot
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    # Print visualization summary
    print(f"\n=== VISUALIZATION SUMMARY ===")
    print(f"Total data points plotted: {len(df)}")
    print(f"Time range: {df['start_time'].min()} to {df['start_time'].max()}")
    if not entry_signals.empty:
        print(f"Entry signals plotted: {len(entry_signals)} (green triangles)")
    if not exit_signals.empty:
        print(f"Exit signals plotted: {len(exit_signals)} (red triangles)")
    if 'rsi_recovery' in df.columns and not rsi_recovery.empty:
        print(f"RSI recovery points: {len(rsi_recovery)} (orange circles)")

def merge_timeframe_signals():
    """
    Main function to merge timeframe signals and generate trading signals
    """
    print("=== TIMEFRAME SIGNAL MERGER ===")
    print("Loading timeframe data...")

    # Load data
    long_df, short_df = load_timeframe_data()
    if long_df is None or short_df is None:
        return None

    # Match timeframes
    matched_df = match_timeframes(long_df, short_df)

    # Generate trading signals
    print("\nGenerating trading signals...")
    result_df = generate_trading_signals(matched_df)

    # Summary statistics
    total_matches = result_df['long_signals'].notna().sum()
    total_entries = result_df['entry_signal'].sum()
    total_exits = result_df['exit_signal'].sum()

    print(f"\n=== MERGE RESULTS ===")
    print(f"Total short timeframe rows: {len(result_df)}")
    print(f"Successfully matched with long timeframe: {total_matches}")
    print(f"Match rate: {total_matches/len(result_df)*100:.1f}%")
    print(f"Entry signals generated: {total_entries}")
    print(f"Exit signals generated: {total_exits}")

    if total_matches > 0:
        print(f"Entry signal rate: {total_entries/total_matches*100:.1f}% of matched rows")
        print(f"Exit signal rate: {total_exits/total_matches*100:.1f}% of matched rows")

    # Show signal distribution
    print(f"\n=== LONG TIMEFRAME SIGNAL DISTRIBUTION ===")
    signal_counts = result_df['long_signals'].value_counts().sort_index()
    for signal, count in signal_counts.items():
        if not pd.isna(signal):
            print(f"Signal {int(signal)}: {count} periods")

    # Display verification samples
    display_verification_samples(result_df)

    # Save results
    output_file = 'merged_timeframe_signals.csv'
    result_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Visualize results
    print("\n=== GENERATING VISUALIZATION ===")
    visualize_trading_signals(result_df)

    return result_df

def main():
    """
    Main execution function
    """
    print("SuperTrend Multi-Timeframe Signal Generator")
    print("\nStrategy Logic:")
    print("1. ENTRY CONDITIONS: RSI recovery (short) + Bullish SuperTrend (long)")
    print("   - Enter immediately regardless of short timeframe signal")
    print("2. EXIT: When short signal changes from 1 to -1 (signal transition)")
    print("=" * 60)

    merged_df = merge_timeframe_signals()

    if merged_df is not None:
        print("\n" + "=" * 50)
        print("MERGE COMPLETED SUCCESSFULLY")
        return merged_df
    else:
        print("\n" + "=" * 50)
        print("MERGE FAILED")
        return None

if __name__ == '__main__':
    result_df = main()