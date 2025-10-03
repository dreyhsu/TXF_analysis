# BTC Tick Data Explanation

## Sample Data
```
First 3 tick records:
      id    price    qty    base_qty           time  is_buyer_maker
3912598  116950.8   36.0  0.03078217  1754006400782            true
3912599  116950.8    1.0  8.55060418  1754006400782            true
3912600  116935.3    1.0  8.55173758  1754006401810            true
```

## Column Definitions

- **id**: Unique trade ID from Binance
- **price**: BTC price in USDT ($116,950.8) - **used as price for volume bars**
- **qty**: Contract quantity (e.g., 36.0 contracts)
- **base_qty**: USDT value of the trade - **used as volume for volume bars**
- **time**: Unix timestamp in milliseconds (e.g., 1754006400782 = 2025-08-01 00:00:00.782) - **used as datetime for volume bars**
- **is_buyer_maker**: true = market sell order, false = market buy order

## Volume Bar Creation

For creating volume bars from BTC tick data:
- **Volume metric**: `base_qty` (USDT value)
- **Datetime**: `time` (Unix timestamp in milliseconds, auto-converted to datetime)
- **Price**: `price` (BTC/USDT price)
- **Suggested volume_per_bar**: 100,000 USDT (adjustable based on market conditions)

## Analysis Notes

The first 2 trades happened at the same millisecond (1754006400782 = 2025-08-01 00:00:00.782) at $116,950.8.
- Trade 3912598: 36.0 contracts worth 0.03078217 USDT (is_buyer_maker=true, sell order)
- Trade 3912599: 1.0 contract worth 8.55060418 USDT (is_buyer_maker=true, sell order)