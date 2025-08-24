#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Data Cleaning Script
Cleans and processes CSV files with Chinese headers and encoding issues
"""

import pandas as pd
import glob
import os
import re

def clean_csv_data(input_file, output_file=None):
    """
    Clean CSV data with proper encoding and column standardization
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    """
    
    # Try different encodings to handle Chinese characters
    encodings = ['utf-8-sig', 'big5', 'gb2312', 'gbk', 'utf-8', 'cp950', 'latin1', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            print(f"Successfully read file with {encoding} encoding")
            break
        except (UnicodeDecodeError, UnicodeError) as e:
            print(f"Failed with {encoding}: {str(e)}")
            continue
        except Exception as e:
            print(f"Unexpected error with {encoding}: {str(e)}")
            continue
    
    if df is None:
        raise ValueError(f"Could not read file {input_file} with any supported encoding")
    
    # Clean column names - replace garbled text with proper Chinese headers
    column_mapping = {
        df.columns[0]: '成交日期',  # Transaction Date
        df.columns[1]: '商品代號',  # Product Code  
        df.columns[2]: '到期月份(週別)',  # Expiry Month (Week)
        df.columns[3]: '成交時間',  # Transaction Time
        df.columns[4]: '成交價格',  # Transaction Price
        df.columns[5]: '成交數量(B+S)',  # Transaction Volume (B+S)
        df.columns[6]: '近月價格',  # Near Month Price
        df.columns[7]: '遠月價格',  # Far Month Price  
        df.columns[8]: '開盤集合競價'  # Opening Call Auction
    }
    
    df = df.rename(columns=column_mapping)
    
    # Clean data
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Replace '-' with NaN for numeric columns
    numeric_columns = ['成交價格', '成交數量(B+S)', '近月價格', '遠月價格']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].replace('-', pd.NA)
    
    # Clean whitespace from string columns
    string_columns = ['商品代號', '到期月份(週別)']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Convert date and time columns to proper format if needed
    if '成交日期' in df.columns:
        df['成交日期'] = pd.to_datetime(df['成交日期'], format='%Y%m%d', errors='coerce')
    
    if '成交時間' in df.columns:
        # Convert HHMMSS format to HH:MM:SS
        df['成交時間'] = df['成交時間'].astype(str).str.zfill(6)
        df['成交時間'] = df['成交時間'].str[:2] + ':' + df['成交時間'].str[2:4] + ':' + df['成交時間'].str[4:6]
    
    # Convert numeric columns to appropriate data types
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter data with specified conditions
    con1 = df.商品代號 == 'MTX'
    con2 = df['到期月份(週別)'] == '202509'
    df = df[con1 & con2]
    
    # Set output file name if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_cleaned.csv"
    
    # Save cleaned data
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Cleaned data saved to: {output_file}")
    
    # Print summary
    print(f"\nData Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def clean_all_csv_files(data_folder="data"):
    """
    Clean CSV files with Daily_YYYY_MM_DD format in the specified data folder
    
    Args:
        data_folder (str): Path to folder containing CSV files
    """
    
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' not found")
        return
    
    # Pattern to match Daily_YYYY_MM_DD.csv format
    daily_pattern = re.compile(r'^Daily_\d{4}_\d{2}_\d{2}\.csv$')
    
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    # Filter files to only include those matching the Daily_YYYY_MM_DD pattern
    daily_files = []
    for file in csv_files:
        filename = os.path.basename(file)
        if daily_pattern.match(filename):
            daily_files.append(file)
    
    if not daily_files:
        print(f"No CSV files with Daily_YYYY_MM_DD format found in '{data_folder}' folder")
        return
    
    print(f"Found {len(daily_files)} CSV files with Daily format to clean:")
    for file in daily_files:
        print(f"  - {file}")
    
    for file in daily_files:
        try:
            print(f"\nCleaning {file}...")
            clean_csv_data(file)
            print(f"Successfully cleaned {file}")
        except Exception as e:
            print(f"Error cleaning {file}: {str(e)}")

if __name__ == "__main__":
    # Clean all CSV files in the data folder
    clean_all_csv_files()
    
    # Or clean a specific file
    # clean_csv_data("data/Daily_2025_08_25.csv")