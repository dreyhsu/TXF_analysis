import pandas as pd
import numpy as np
from datetime import datetime

def merge_timeframe_signals():
    """
    Merge short and long timeframe SuperTrend signals.

    Logic:
    1. Load both CSV files (long and short timeframe data)
    2. For each row in short timeframe:
       - Find matching time period in long timeframe where short.start_time
         falls between long.start_time and long.end_time
       - Extract the corresponding signals value from long timeframe
    3. Generate entry signals when both conditions are met:
       - rsi_recovery == True (from short timeframe)
       - signals == 1 (from long timeframe)
    """

    # Load the CSV files
    print("Loading timeframe data...")
    try:
        long_df = pd.read_csv('supertrend/supertrend_positions_long.csv')
        short_df = pd.read_csv('supertrend/supertrend_positions_short.csv')
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return None

    print(f"Long timeframe data: {len(long_df)} rows")
    print(f"Short timeframe data: {len(short_df)} rows")

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
    short_df['entry_signal'] = False

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

            # Generate entry signal when both conditions are met
            has_rsi_recovery = short_row.get('rsi_recovery', False)
            if has_rsi_recovery and long_signal == 1:
                short_df.at[short_idx, 'entry_signal'] = True

        # Progress indicator
        if short_idx % 100 == 0:
            print(f"Processed {short_idx}/{len(short_df)} rows...")

    # Summary statistics
    total_matches = short_df['long_signals'].notna().sum()
    total_entries = short_df['entry_signal'].sum()

    print(f"\n=== MERGE RESULTS ===")
    print(f"Total short timeframe rows: {len(short_df)}")
    print(f"Successfully matched with long timeframe: {total_matches}")
    print(f"Match rate: {total_matches/len(short_df)*100:.1f}%")
    print(f"Total entry signals generated: {total_entries}")

    if total_matches > 0:
        print(f"Entry signal rate: {total_entries/total_matches*100:.1f}% of matched rows")

    # Show signal distribution
    print(f"\n=== LONG TIMEFRAME SIGNAL DISTRIBUTION ===")
    signal_counts = short_df['long_signals'].value_counts().sort_index()
    for signal, count in signal_counts.items():
        if not pd.isna(signal):
            print(f"Signal {int(signal)}: {count} periods")

    # Save merged results
    output_file = 'merged_timeframe_signals.csv'
    short_df.to_csv(output_file, index=False)
    print(f"\nMerged results saved to: {output_file}")

    # Show sample of entry signals
    entry_signals = short_df[short_df['entry_signal'] == True]
    if not entry_signals.empty:
        print(f"\n=== SAMPLE ENTRY SIGNALS ===")
        print("First 5 entry signals:")
        print(entry_signals[['start_time', 'end_time', 'rsi_recovery', 'long_signals', 'entry_signal']].head())

    return short_df

def main():
    """
    Main execution function
    """
    print("=== TIMEFRAME MERGER ===")
    print("Merging short and long timeframe SuperTrend signals...")
    print("\nLogic:")
    print("1. Match short timeframe periods with long timeframe trends")
    print("2. Generate entry signals when:")
    print("   - RSI recovery signal = True (short timeframe)")
    print("   - SuperTrend signal = 1 (long timeframe)")
    print()

    merged_df = merge_timeframe_signals()

    if merged_df is not None:
        print("\n=== MERGE COMPLETED SUCCESSFULLY ===")
        return merged_df
    else:
        print("\n=== MERGE FAILED ===")
        return None

if __name__ == '__main__':
    result_df = main()