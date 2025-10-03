import os
import websocket
import json
import sqlite3
import logging
from datetime import datetime
import time
import threading
from typing import Optional, Dict, Any
import signal
import sys

class BinanceTickFetcher:
    def __init__(self, db_path: str = "btc_tick_data.db"):
        self.db_path = db_path
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        self.last_ping_time = time.time()

        # Setup logging
        self.setup_logging()

        # Initialize database
        self.init_database()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('binance_tick_fetcher.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def init_database(self):
        """Initialize SQLite database with tick data schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY,
                    price REAL NOT NULL,
                    qty REAL NOT NULL,
                    quoteQty REAL NOT NULL,
                    time TIMESTAMP NOT NULL,
                    isBuyerMaker BOOLEAN NOT NULL,
                    isBestMatch BOOLEAN NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create index on time for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_tick_time ON tick_data(time)
            ''')

            conn.commit()
            conn.close()
            self.logger.info(f"Database initialized: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def get_last_timestamp(self) -> Optional[int]:
        """Get the last timestamp from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT MAX(time) FROM tick_data')
            result = cursor.fetchone()[0]

            conn.close()

            if result:
                # Convert timestamp to milliseconds for Binance API
                last_time = datetime.fromisoformat(result.replace('Z', '+00:00'))
                return int(last_time.timestamp() * 1000)

            return None

        except Exception as e:
            self.logger.error(f"Error getting last timestamp: {e}")
            return None

    def save_tick_to_db(self, trade_data: Dict[str, Any]):
        """Save single tick data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO tick_data (id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['id'],
                trade_data['price'],
                trade_data['qty'],
                trade_data['quoteQty'],
                trade_data['time'],
                trade_data['isBuyerMaker'],
                trade_data['isBestMatch']
            ))

            conn.commit()
            conn.close()

        except sqlite3.IntegrityError:
            # Skip duplicate records
            pass
        except Exception as e:
            self.logger.error(f"Error saving tick to database: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.ws:
            self.ws.close()
        sys.exit(0)

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)

            # Extract trade data
            trade = {
                'id': data['t'],
                'price': float(data['p']),
                'qty': float(data['q']),
                'quoteQty': float(data['p']) * float(data['q']),
                'time': datetime.fromtimestamp(data['T'] / 1000).isoformat(),
                'isBuyerMaker': data['m'],
                'isBestMatch': True
            }

            # Save to database immediately
            self.save_tick_to_db(trade)

            # Update ping time to track connection health
            self.last_ping_time = time.time()

            # Log progress every 1000 records (optional, can be adjusted)
            if data['t'] % 1000 == 0:
                self.logger.info(f"Processed tick ID: {data['t']} - Price: ${trade['price']}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.logger.warning(f"WebSocket connection closed: {close_status_code} - {close_msg}")

        if self.running:
            self.logger.info("Attempting to reconnect...")
            self.reconnect()

    def on_open(self, ws):
        """Handle WebSocket open"""
        self.logger.info("WebSocket connection opened successfully")
        self.reconnect_attempts = 0  # Reset reconnect counter on successful connection

        last_timestamp = self.get_last_timestamp()
        if last_timestamp:
            self.logger.info(f"Resuming from last timestamp: {datetime.fromtimestamp(last_timestamp / 1000)}")
        else:
            self.logger.info("Starting fresh data collection")

    def on_ping(self, ws, message):
        """Handle ping from server"""
        self.logger.debug("Received ping from server")
        # WebSocket library handles pong automatically, no manual response needed

    def on_pong(self, ws, message):
        """Handle pong response"""
        self.logger.debug("Received pong from server")
        self.last_ping_time = time.time()

    def reconnect(self):
        """Implement exponential backoff reconnection logic"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping.")
            self.running = False
            return

        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 300)  # Max 5 min delay

        self.logger.info(f"Reconnect attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {delay} seconds...")
        time.sleep(delay)

        if self.running:
            self.start_collection()

    def start_collection(self, symbol='BTCUSDT'):
        """
        Start continuous WebSocket data collection
        """
        self.running = True
        self.logger.info(f"Starting continuous data collection for {symbol}")

        # Binance WebSocket URL for trade stream
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"

        try:
            # Create WebSocket connection with ping/pong support
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_ping=self.on_ping,
                on_pong=self.on_pong
            )

            # Run WebSocket with automatic ping/pong (Binance requires response within 10 minutes)
            self.ws.run_forever(ping_interval=60, ping_timeout=10)

        except Exception as e:
            self.logger.error(f"Error in WebSocket connection: {e}")
            if self.running:
                self.reconnect()

    def run_continuous(self, symbol='BTCUSDT'):
        """
        Run continuous data collection with connection monitoring
        """
        self.logger.info("Starting BTC tick data collector...")

        # Start the WebSocket connection
        ws_thread = threading.Thread(target=self.start_collection, args=(symbol,))
        ws_thread.daemon = True
        ws_thread.start()

        # Monitor connection health
        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds

                # Check if connection is still alive
                if time.time() - self.last_ping_time > 300:  # 5 minutes without activity
                    self.logger.warning("No activity detected, forcing reconnection...")
                    if self.ws:
                        self.ws.close()

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
            self.running = False
            if self.ws:
                self.ws.close()

    def get_tick_count(self) -> int:
        """Get total number of ticks in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM tick_data')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            self.logger.error(f"Error getting tick count: {e}")
            return 0

def main():
    """
    Main function to start continuous BTC tick data collection
    """
    try:
        # Initialize fetcher with database path
        db_path = os.path.join(os.path.dirname(__file__), "btc_tick_data.db")
        fetcher = BinanceTickFetcher(db_path)

        # Show existing data count
        existing_count = fetcher.get_tick_count()
        fetcher.logger.info(f"Existing ticks in database: {existing_count}")

        # Start continuous collection
        fetcher.run_continuous('BTCUSDT')

    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()