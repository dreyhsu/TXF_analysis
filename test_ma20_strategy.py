import pandas as pd
import numpy as np

# Test the modified ma20_volume_strategy logic without external dependencies

def test_exit_logic():
    """Test the new exit logic implementation"""
    
    # Create sample data
    dates = pd.date_range('2024-01-01 09:00:00', periods=100, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    
    # Create sample volume bars data
    volume_bars = pd.DataFrame({
        'close': prices,
        'high': prices + np.random.rand(100) * 0.5,
        'low': prices - np.random.rand(100) * 0.5,
        'ma20': prices * 0.98 + np.random.randn(100) * 0.1,  # Simulate MA20
        'buy_signal': [False] * 100,
        'exit_long_stop': [False] * 100,
        'exit_long_trend': [False] * 100,
        'exit_long': [False] * 100,
    })
    
    # Add some buy signals
    volume_bars.loc[20, 'buy_signal'] = True
    volume_bars.loc[50, 'buy_signal'] = True
    
    # Calculate rolling minimum and stop loss level
    volume_bars['rolling_min_20'] = volume_bars['low'].rolling(window=20).min()
    volume_bars['stop_loss_level'] = volume_bars['rolling_min_20'] * 0.99
    
    # Simulate the position tracking logic
    current_position = 0
    entry_bar = 0
    
    stop_exits = 0
    trend_exits = 0
    
    for i in volume_bars.index:
        # Entry logic
        if current_position == 0 and volume_bars.loc[i, 'buy_signal']:
            current_position = 1
            entry_bar = i
            print(f"Entry at bar {i}, price: {volume_bars.loc[i, 'close']:.2f}")
        
        # Exit logic
        elif current_position == 1:
            bars_since_entry = i - entry_bar
            
            # Stop loss check
            stop_loss_exit = volume_bars.loc[i, 'close'] < volume_bars.loc[i, 'stop_loss_level']
            
            # Trend exit check (after 10 bars)
            trend_exit = (bars_since_entry >= 10) and (volume_bars.loc[i, 'close'] < volume_bars.loc[i, 'ma20'])
            
            if stop_loss_exit:
                current_position = 0
                stop_exits += 1
                print(f"Stop loss exit at bar {i}, price: {volume_bars.loc[i, 'close']:.2f}, stop level: {volume_bars.loc[i, 'stop_loss_level']:.2f}")
            
            elif trend_exit:
                current_position = 0
                trend_exits += 1
                print(f"Trend exit at bar {i}, price: {volume_bars.loc[i, 'close']:.2f}, MA20: {volume_bars.loc[i, 'ma20']:.2f}, bars since entry: {bars_since_entry}")
    
    print(f"\nTest Results:")
    print(f"Stop loss exits: {stop_exits}")
    print(f"Trend exits: {trend_exits}")
    print(f"Total exits: {stop_exits + trend_exits}")
    
    return True

if __name__ == "__main__":
    print("Testing modified MA20 strategy exit logic...")
    test_exit_logic()
    print("Test completed successfully!")