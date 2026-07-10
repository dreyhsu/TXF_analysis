#!/usr/bin/env python3
"""
XAU 1-minute Candlestick W-Bottom & Consolidation-to-Spike Dual Strategy Backtester & Charting Tool
Author: Antigravity

This script backtests two coexisting entry strategies on XAU 1-minute data:
1. W-Bottom Reversal: MA5 double bottoms with low valley difference, filtered by a 40% buying tail candle.
2. Consolidation-to-Spike: Rises in MA5 support (regression slope of valleys > 0) followed by a strong price breakout.

Exits: 1:1 risk-reward target bounds (R = EntryPrice - MA5 of closest valley).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_latest_date(csv_path):
    """
    Fast method to scan the end of the large CSV file and retrieve the last available date.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(csv_path, 'rb') as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek_pos = max(0, size - 4096)
            f.seek(seek_pos, os.SEEK_SET)
            chunk = f.read().decode('utf-8', errors='ignore')
            lines = chunk.splitlines()
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if parts and parts[0] and len(parts) >= 5:
                    date_part = parts[0].split(' ')[0]
                    if len(date_part.split('.')) == 3:
                        return date_part
        except Exception as e:
            print(f"Warning: Failed to quickly parse latest date due to {e}. Defaulting to scanning method.", file=sys.stderr)
            
    latest_date = None
    with open(csv_path, 'r', encoding='utf-8') as f:
        f.readline() # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if parts and parts[0]:
                latest_date = parts[0].split(' ')[0]
    return latest_date

def load_data(csv_path, target_date, buffer_size=50):
    """
    Loads target_date data along with a preceding buffer of buffer_size rows.
    Uses sequential reading for memory efficiency.
    """
    print(f"Scanning CSV for target date: {target_date}...")
    
    buffer = []
    target_rows = []
    header = None
    
    normalized_target = target_date.replace('-', '.').replace('/', '.')
    
    try:
        from datetime import datetime
        clean_date = target_date.replace('-', '.').replace('/', '.')
        dt = datetime.strptime(clean_date, "%Y.%m.%d")
        normalized_target = dt.strftime("%Y.%m.%d")
    except Exception:
        normalized_target = target_date.replace('-', '.').replace('/', '.')
        
    found_target = False
    with open(csv_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
        header = header_line.split(';')
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) < 6:
                continue
                
            dt_str = parts[0]
            date_part = dt_str.split(' ')[0]
            
            if date_part == normalized_target:
                found_target = True
                target_rows.append(parts)
            elif found_target:
                break
            else:
                buffer.append(parts)
                if len(buffer) > buffer_size:
                    buffer.pop(0)
                    
    if not found_target:
        print(f"Error: Target date '{target_date}' (normalized: '{normalized_target}') not found in dataset.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Successfully loaded {len(target_rows)} rows for {normalized_target} and {len(buffer)} buffer rows from the preceding period.")
    
    all_rows = buffer + target_rows
    df = pd.DataFrame(all_rows, columns=header)
    
    df['Open'] = pd.to_numeric(df['Open'])
    df['High'] = pd.to_numeric(df['High'])
    df['Low'] = pd.to_numeric(df['Low'])
    df['Close'] = pd.to_numeric(df['Close'])
    df['Volume'] = pd.to_numeric(df['Volume'])
    
    return df, len(buffer)

def get_regression_slope(x, y):
    """
    Calculates the linear regression slope.
    """
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def backtest_strategy(df, buffer_len, lookback=30, tolerance=0.003, spacing=3, ma_period=5, min_height=0.0015, max_risk=0.005):
    """
    Backtests the dual W-Bottom and Consolidation-to-Spike strategy.
    """
    # Calculate MA5 using Low prices
    df['ma5'] = df['Low'].rolling(window=ma_period).mean()
    
    df['buy_signal'] = False
    df['sell_signal'] = False
    
    position = None
    trades = []
    m = df['ma5'].values
    closes = df['Close'].values
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    dates = df['Date'].values
    
    for i in range(buffer_len, len(df)):
        if position is not None:
            close_val = closes[i]
            high_val = highs[i]
            low_val = lows[i]
            
            # Check Take Profit: High >= Entry + R
            is_tp = high_val >= position['tp_price']
            # Check Stop Loss: Low <= Entry - R
            is_sl = low_val <= position['sl_price']
            
            if is_tp or is_sl:
                entry_price = position['entry_price']
                exit_price = close_val
                pnl = exit_price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                status = 'Take Profit' if is_tp else 'Stop Loss'
                
                trade = {
                    'entry_time': position['entry_time'],
                    'entry_price': entry_price,
                    'exit_time': dates[i],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'status': status,
                    'tp_price': position['tp_price'],
                    'sl_price': position['sl_price'],
                    'strategy': position['strategy'],
                    'w_bottom': position['w_bottom']
                }
                trades.append(trade)
                df.at[df.index[i], 'sell_signal'] = True
                position = None
            continue
            
        # Entry evaluation
        t = i
        if t - lookback + 1 >= 1:
            start_w = t - lookback + 1
            
            # ----------------------------------------------------
            # Setup 1: Check Consolidation-to-Spike Strategy
            # ----------------------------------------------------
            valleys_x = []
            valleys_y = []
            
            # Scan lookback window [start_w + 1, t - 1] for local valleys
            for j in range(start_w + 1, t):
                if m[j] <= m[j-1] and m[j] <= m[j+1]:
                    valleys_x.append(j)
                    valleys_y.append(m[j])
                    
            trigger_spike = False
            last_valley_idx = None
            
            if len(valleys_x) >= 2:
                # Validate higher lows trend using regression slope
                slope = get_regression_slope(np.array(valleys_x), np.array(valleys_y))
                if slope > 0:
                    # Trigger condition: Red candle + price breakout above highest Close in lookback window
                    is_red = closes[t] > opens[t]
                    is_breakout = closes[t] > np.max(closes[start_w : t])
                    if is_red and is_breakout:
                        trigger_spike = True
                        last_valley_idx = valleys_x[-1]
                        
            if trigger_spike and last_valley_idx is not None:
                entry_price = closes[t]
                r = entry_price - m[last_valley_idx]
                if r <= 0:
                    r = 1.0
                    
                # Skip if entry is too far from reference valley
                if r / entry_price <= max_risk:
                    position = {
                        'entry_time': dates[t],
                        'entry_price': entry_price,
                        'entry_idx': t,
                        'tp_price': entry_price + r,
                        'sl_price': entry_price - r,
                        'strategy': 'Consolidation-Spike',
                        'w_bottom': {
                            # We store the latest valley point for mapping chart shapes
                            'v1_time': dates[last_valley_idx],
                            'v1_val': m[last_valley_idx],
                            'p_time': dates[last_valley_idx],
                            'p_val': m[last_valley_idx],
                            'v2_time': dates[last_valley_idx],
                            'v2_val': m[last_valley_idx]
                        }
                    }
                    df.at[df.index[t], 'buy_signal'] = True
                    continue
                
            # ----------------------------------------------------
            # Setup 2: Check W-Bottom Strategy (Fallback)
            # ----------------------------------------------------
            # 1. Long-tail candle check in lookback window [t - lookback + 1, t]
            long_tail_found = False
            for w_idx in range(t - lookback + 1, t + 1):
                rng = highs[w_idx] - lows[w_idx]
                if rng > 0:
                    tail = min(opens[w_idx], closes[w_idx]) - lows[w_idx]
                    if tail > rng * 0.40:
                        long_tail_found = True
                        break
            if not long_tail_found:
                continue
                
            end_idx = t - 1
            found_w = False
            best_triplet = None
            
            # Loop to find valid W-bottom triplet (i1, ip, i2)
            for i1 in range(start_w + 1, end_idx - spacing):
                if not (m[i1] <= m[i1-1] and m[i1] <= m[i1+1]):
                    continue
                    
                for ip in range(i1 + spacing, end_idx - spacing + 1):
                    if not (m[ip] >= m[ip-1] and m[ip] >= m[ip+1]):
                        continue
                    if not (m[ip] > m[i1]):
                        continue
                        
                    for i2 in range(ip + spacing, end_idx + 1):
                        if not (m[i2] <= m[i2-1] and m[i2] <= m[i2+1]):
                            continue
                        if not (m[ip] > m[i2]):
                            continue
                            
                        # Valley closeness check
                        v1 = m[i1]
                        v2 = m[i2]
                        if abs(v1 - v2) / max(v1, v2) > tolerance:
                            continue
                            
                        # Minimum height check
                        min_allowed_height = m[ip] * min_height
                        if (m[ip] - v1 < min_allowed_height) or (m[ip] - v2 < min_allowed_height):
                            continue
                            
                        # Breakout trigger check at current t
                        if m[t-1] <= m[ip] < m[t]:
                            found_w = True
                            best_triplet = (i1, ip, i2)
                            break
                    if found_w:
                        break
                if found_w:
                    break
                    
            if found_w and best_triplet:
                i1, ip, i2 = best_triplet
                entry_price = closes[t]
                
                # Calculate risk range R based on Valley 2 (i2)
                r = entry_price - m[i2]
                if r <= 0:
                    r = 1.0
                    
                # W-Bottom entries are structural and do not apply the breakout max_risk filter
                position = {
                    'entry_time': dates[t],
                    'entry_price': entry_price,
                    'entry_idx': t,
                    'tp_price': entry_price + r,
                    'sl_price': entry_price - r,
                    'strategy': 'W-Bottom',
                    'w_bottom': {
                        'v1_time': dates[i1],
                        'v1_val': m[i1],
                        'p_time': dates[ip],
                        'p_val': m[ip],
                        'v2_time': dates[i2],
                        'v2_val': m[i2]
                    }
                }
                df.at[df.index[t], 'buy_signal'] = True
                
    # Force close open position at EOD
    if position is not None:
        c = df.iloc[-1]
        entry_price = position['entry_price']
        exit_price = c['Close']
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        
        trade = {
            'entry_time': position['entry_time'],
            'entry_price': entry_price,
            'exit_time': f"{c['Date']} (Forced EOD)",
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'status': 'Forced EOD',
            'tp_price': position['tp_price'],
            'sl_price': position['sl_price'],
            'strategy': position['strategy'],
            'w_bottom': position['w_bottom']
        }
        trades.append(trade)
        df.at[df.index[-1], 'sell_signal'] = True
        
    return trades

def print_performance_summary(trades, target_date):
    """
    Outputs performance statistics and trade log to console.
    """
    print("\n" + "="*70)
    print(f" STRATEGY PERFORMANCE SUMMARY: {target_date} ")
    print("="*70)
    
    if not trades:
        print("No trades were executed on this day.")
        print("="*70)
        return
        
    print(f"{'No.':<4} {'Entry Time':<18} {'Entry Px':<10} {'Exit Time':<24} {'Exit Px':<10} {'PnL (pts)':<10} {'PnL %':<8} {'Status':<12} {'Strategy':<18}")
    print("-"*110)
    
    total_pnl = 0.0
    total_pnl_pct = 0.0
    wins = 0
    losses = 0
    max_win = -99999.0
    max_loss = 99999.0
    
    for i, t in enumerate(trades, 1):
        pnl = t['pnl']
        pnl_pct = t['pnl_pct']
        total_pnl += pnl
        total_pnl_pct += pnl_pct
        
        if pnl > 0:
            wins += 1
            max_win = max(max_win, pnl)
        elif pnl < 0:
            losses += 1
            max_loss = min(max_loss, pnl)
            
        color_start = "\033[92m" if pnl >= 0 else "\033[91m"
        color_end = "\033[0m"
        
        print(f"{i:<4} {t['entry_time']:<18} {t['entry_price']:<10.2f} {t['exit_time']:<24} {t['exit_price']:<10.2f} "
              f"{color_start}{pnl:<10.2f}{color_end} {color_start}{pnl_pct:<8.2%}{color_end} {t['status']:<12} {t['strategy']:<18}")
              
    total_trades = len(trades)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    avg_pnl_pct = total_pnl_pct / total_trades if total_trades > 0 else 0
    
    print("-"*110)
    print(f"Total Executed Trades : {total_trades}")
    print(f"Winning Trades        : {wins} (Win Rate: {win_rate:.2f}%)")
    print(f"Losing Trades         : {losses}")
    print(f"Net Profit/Loss       : {total_pnl:.2f} pts ({total_pnl_pct:.2%})")
    print(f"Average Profit/Trade  : {avg_pnl:.2f} pts ({avg_pnl_pct:.2%})")
    if wins > 0:
        print(f"Max Win (points)      : {max_win:.2f} pts")
    if losses > 0:
        print(f"Max Loss (points)     : {max_loss:.2f} pts")
    print("="*70 + "\n")

def generate_interactive_chart(df, buffer_len, trades, target_date, output_path):
    """
    Generates a beautiful interactive Plotly dark-themed HTML chart with strategy shapes.
    """
    plot_df = df.iloc[buffer_len:].copy()
    
    fig = go.Figure()
    
    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=plot_df['Date'],
        open=plot_df['Open'],
        high=plot_df['High'],
        low=plot_df['Low'],
        close=plot_df['Close'],
        name='XAU/USD 1m',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
    ))
    
    # 2. MA5 Line
    fig.add_trace(go.Scatter(
        x=plot_df['Date'],
        y=plot_df['ma5'],
        mode='lines',
        name='MA5',
        line=dict(color='#ffb74d', width=1.5),
        opacity=0.8
    ))
    
    # 3. Plot Entry strategy outlines
    for idx, t in enumerate(trades):
        if 'w_bottom' in t and t['w_bottom'] and t['strategy'] == 'W-Bottom':
            w = t['w_bottom']
            fig.add_trace(go.Scatter(
                x=[w['v1_time'], w['p_time'], w['v2_time'], t['entry_time']],
                y=[w['v1_val'], w['p_val'], w['v2_val'], t['entry_price']],
                mode='lines+markers',
                name=f"W-Pattern {idx+1}",
                line=dict(color='rgba(255, 183, 77, 0.5)', width=2, dash='dot'),
                marker=dict(size=7, color='#ffb74d', symbol='circle-open'),
                hoverinfo='skip',
                legendgroup="Patterns",
                showlegend=(idx == 0)
            ))
        elif 'w_bottom' in t and t['w_bottom'] and t['strategy'] == 'Consolidation-Spike':
            w = t['w_bottom']
            fig.add_trace(go.Scatter(
                x=[w['v1_time'], t['entry_time']],
                y=[w['v1_val'], t['entry_price']],
                mode='lines+markers',
                name=f"Consolidation-Spike {idx+1}",
                line=dict(color='rgba(100, 181, 246, 0.5)', width=2, dash='dashdot'),
                marker=dict(size=7, color='#64b5f6', symbol='x-open'),
                hoverinfo='skip',
                legendgroup="Patterns",
                showlegend=(idx == 0)
            ))
            
    # 4. Buy Entry markers
    buy_times = [t['entry_time'] for t in trades]
    buy_prices = [t['entry_price'] for t in trades]
    fig.add_trace(go.Scatter(
        x=buy_times,
        y=buy_prices,
        mode='markers',
        name='Buy Entry',
        marker=dict(
            symbol='triangle-up',
            size=14,
            color='#00e676',
            line=dict(color='#003300', width=1.5)
        ),
        text=[f"Buy ({t['strategy']}) at {t['entry_price']:.2f}" for t in trades],
        hoverinfo='text+x'
    ))
    
    # 5. Sell Exit markers
    exit_times = []
    exit_prices = []
    for t in trades:
        raw_time = t['exit_time']
        if " (Forced EOD)" in raw_time:
            raw_time = raw_time.replace(" (Forced EOD)", "")
        exit_times.append(raw_time)
        exit_prices.append(t['exit_price'])
        
    fig.add_trace(go.Scatter(
        x=exit_times,
        y=exit_prices,
        mode='markers',
        name='Sell Exit (TP/SL)',
        marker=dict(
            symbol='triangle-down',
            size=14,
            color='#ff1744',
            line=dict(color='#330000', width=1.5)
        ),
        text=[f"Exit at {p:.2f} ({t['status']})" for p in exit_prices],
        hoverinfo='text+x'
    ))
    
    # 6. Add lines connecting Entry & Exit + SL/TP limit bounds
    for idx, t in enumerate(trades):
        ex_time = t['exit_time'].replace(" (Forced EOD)", "")
        color = '#4caf50' if t['pnl'] >= 0 else '#f44336'
        
        # Trade vector
        fig.add_shape(
            type="line",
            x0=t['entry_time'], y0=t['entry_price'],
            x1=ex_time, y1=t['exit_price'],
            line=dict(
                color=color,
                width=1.5,
                dash="dash"
            )
        )
        
        # Take Profit line
        if 'tp_price' in t:
            fig.add_trace(go.Scatter(
                x=[t['entry_time'], ex_time],
                y=[t['tp_price'], t['tp_price']],
                mode='lines',
                name=f"TP {idx+1}",
                line=dict(color='rgba(38, 166, 154, 0.45)', width=1.5, dash='dash'),
                legendgroup=f"Limits {idx+1}",
                showlegend=False,
                hoverinfo='skip'
            ))
            
        # Stop Loss line
        if 'sl_price' in t:
            fig.add_trace(go.Scatter(
                x=[t['entry_time'], ex_time],
                y=[t['sl_price'], t['sl_price']],
                mode='lines',
                name=f"SL {idx+1}",
                line=dict(color='rgba(239, 83, 80, 0.45)', width=1.5, dash='dash'),
                legendgroup=f"Limits {idx+1}",
                showlegend=False,
                hoverinfo='skip'
            ))
        
    fig.update_layout(
        title={
            'text': f"XAU 1-Minute Dual Strategy Backtest ({target_date})",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': '#ffffff'}
        },
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            title="Time",
            gridcolor='#2d2d2d',
            linecolor='#555555',
            showgrid=True,
            type='category'
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor='#2d2d2d',
            linecolor='#555555',
            showgrid=True,
            autorange=True,
            fixedrange=False
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(20, 20, 20, 0.7)',
            bordercolor='#555555',
            borderwidth=1
        ),
        paper_bgcolor='#121212',
        plot_bgcolor='#121212',
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode='x unified'
    )
    
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Interactive HTML chart saved successfully to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Backtest a W-bottom and Consolidation-to-Spike strategy on XAU 1m data.")
    parser.add_argument(
        "--csv", 
        type=str, 
        default="xau/data/XAU_1m_data.csv", 
        help="Path to XAU 1m CSV data (default: xau/data/XAU_1m_data.csv)"
    )
    parser.add_argument(
        "--date", 
        type=str, 
        default=None, 
        help="Target date YYYY.MM.DD (default: automatically finds the latest date in the CSV)"
    )
    parser.add_argument(
        "--lookback", 
        type=int, 
        default=30, 
        help="Lookback window size in candles for W-bottom detection (default: 30)"
    )
    parser.add_argument(
        "--tolerance", 
        type=float, 
        default=0.003, 
        help="Closeness tolerance percentage for the two valley values (default: 0.003 for 0.3%%)"
    )
    parser.add_argument(
        "--spacing", 
        type=int, 
        default=3, 
        help="Minimum spacing in candles between turning points (default: 3)"
    )
    parser.add_argument(
        "--ma-period", 
        type=int, 
        default=5, 
        help="MA period for W-bottom and exit logic (default: 5)"
    )
    parser.add_argument(
        "--min-height", 
        type=float, 
        default=0.0015, 
        help="Minimum height difference percentage between peak and valleys (default: 0.0015 for 0.15%%)"
    )
    parser.add_argument(
        "--max-risk",
        type=float,
        default=0.005,
        help="Maximum risk ratio (R / EntryPrice) allowed for entering trades (default: 0.005 for 0.5%%)"
    )
    args = parser.parse_args()
    
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        workspace_root = "/Users/chihjuihsu/Documents/python_script/TXF_analysis"
        csv_path = os.path.join(workspace_root, csv_path)
        
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at: {csv_path}", file=sys.stderr)
        sys.exit(1)
        
    if args.date is None:
        print("No date specified. Quickly parsing latest date from the end of the file...")
        target_date = get_latest_date(csv_path)
        if target_date is None:
            print("Error: Could not automatically detect the latest date.", file=sys.stderr)
            sys.exit(1)
        print(f"Detected latest date in CSV: {target_date}")
    else:
        target_date = args.date.strip()
        
    buffer_needed = max(50, args.lookback + 5)
    df, buffer_len = load_data(csv_path, target_date, buffer_size=buffer_needed)
    
    trades = backtest_strategy(
        df, 
        buffer_len, 
        lookback=args.lookback, 
        tolerance=args.tolerance, 
        spacing=args.spacing, 
        ma_period=args.ma_period,
        min_height=args.min_height,
        max_risk=args.max_risk
    )
    
    print_performance_summary(trades, target_date)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "xau_strategy_chart.html")
    generate_interactive_chart(df, buffer_len, trades, target_date, output_path)

if __name__ == "__main__":
    main()
