import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

class VolumeBaseChart:
    def __init__(self, volume_threshold=10000):
        self.volume_threshold = volume_threshold

    def convert_tick_to_volume_bars(self, tick_data):
        """
        Convert tick data to volume-based bars
        Creates new bar when quoteQty cumulates to volume_threshold

        Raw data columns:
        - id: Unique trade ID from Binance
        - price: BTC price in USDT
        - qty: Bitcoin quantity traded (in BTC)
        - quoteQty: Trade value in USDT (price × qty)
        - time: Exact timestamp when trade occurred
        - isBuyerMaker: False = market buy order, True = market sell order
        - isBestMatch: Whether this trade was the best price match
        """
        bars = []
        current_bar = {
            'start_time': None,
            'end_time': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0,
            'quote_volume': 0,
            'trade_count': 0
        }

        cumulative_quote_qty = 0

        for idx, row in tick_data.iterrows():
            price = row['price']
            qty = row['qty']
            quote_qty = row['quoteQty']
            timestamp = row['time']

            # Initialize first bar
            if current_bar['start_time'] is None:
                current_bar['start_time'] = timestamp
                current_bar['open'] = price
                current_bar['high'] = price
                current_bar['low'] = price

            # Update current bar
            current_bar['end_time'] = timestamp
            current_bar['close'] = price
            current_bar['high'] = max(current_bar['high'], price)
            current_bar['low'] = min(current_bar['low'], price)
            current_bar['volume'] += qty
            current_bar['quote_volume'] += quote_qty
            current_bar['trade_count'] += 1

            cumulative_quote_qty += quote_qty

            # Check if we should create new bar
            if cumulative_quote_qty >= self.volume_threshold:
                bars.append(current_bar.copy())

                # Reset for next bar
                current_bar = {
                    'start_time': None,
                    'end_time': None,
                    'open': None,
                    'high': None,
                    'low': None,
                    'close': None,
                    'volume': 0,
                    'quote_volume': 0,
                    'trade_count': 0
                }
                cumulative_quote_qty = 0

        # Add last incomplete bar if it has data
        if current_bar['start_time'] is not None:
            bars.append(current_bar)

        return pd.DataFrame(bars)

    def plot_volume_bars(self, volume_bars, save_path=None):
        """
        Plot volume-based candlestick chart
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

        # Candlestick chart
        for idx, bar in volume_bars.iterrows():
            color = 'green' if bar['close'] >= bar['open'] else 'red'

            # Body
            body_height = abs(bar['close'] - bar['open'])
            body_bottom = min(bar['open'], bar['close'])
            ax1.bar(idx, body_height, bottom=body_bottom, color=color, alpha=0.7, width=0.8)

            # Wicks
            ax1.plot([idx, idx], [bar['low'], bar['high']], color='black', linewidth=1)

        ax1.set_title(f'Volume-Based Candlestick Chart (${self.volume_threshold} per bar)')
        ax1.set_ylabel('Price (USDT)')
        ax1.grid(True, alpha=0.3)

        # Volume chart
        colors = ['green' if bar['close'] >= bar['open'] else 'red' for _, bar in volume_bars.iterrows()]
        ax2.bar(range(len(volume_bars)), volume_bars['quote_volume'], color=colors, alpha=0.7)
        ax2.set_ylabel('Quote Volume (USDT)')
        ax2.set_xlabel('Bar Index')
        ax2.grid(True, alpha=0.3)

        # Set x-axis labels with start times
        x_labels = [bar['start_time'].strftime('%H:%M:%S.%f')[:-3] for _, bar in volume_bars.iterrows()]
        step = max(1, len(x_labels) // 10)  # Show max 10 labels
        ax1.set_xticks(range(0, len(x_labels), step))
        ax1.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45)
        ax2.set_xticks(range(0, len(x_labels), step))
        ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")

        plt.show()

    def save_volume_bars(self, volume_bars, filename=None):
        """
        Save volume bars to CSV
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'volume_bars_{timestamp}.csv'

        filepath = os.path.join(os.path.dirname(__file__), filename)
        volume_bars.to_csv(filepath, index=False)
        print(f"Volume bars saved to: {filepath}")
        return filepath

def main():
    # Find the most recent tick data CSV file
    # tick_files = [f for f in os.listdir('.') if f.startswith('btc_tick_data_') and f.endswith('.csv')]

    # if not tick_files:
    #     print("No tick data CSV files found. Please run fetch_btc_tick.py first.")
    #     return

    # # Use the most recent file
    # tick_file = sorted(tick_files)[-1]
    # print(f"Using tick data file: {tick_file}")

    tick_file = r'C:\Users\Drey\Documents\Python_Scripts\TXF_analysis\binance_tick\btc_tick_data_20250915_173147.csv'
    # Load tick data
    tick_data = pd.read_csv(tick_file)
    tick_data['time'] = pd.to_datetime(tick_data['time'])

    print(f"Loaded {len(tick_data)} tick records")
    print(f"Total quote volume: ${tick_data['quoteQty'].sum():.2f}")

    # Convert to volume bars
    volume_chart = VolumeBaseChart(volume_threshold=30000)
    volume_bars = volume_chart.convert_tick_to_volume_bars(tick_data)

    print(f"Created {len(volume_bars)} volume bars")

    # Display summary
    if len(volume_bars) > 0:
        print("\nFirst 5 volume bars:")
        print(volume_bars[['start_time', 'open', 'high', 'low', 'close', 'quote_volume', 'trade_count']].head())

        # Save volume bars
        volume_chart.save_volume_bars(volume_bars)

        # Plot chart
        chart_path = f'volume_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        volume_chart.plot_volume_bars(volume_bars, chart_path)

if __name__ == "__main__":
    main()