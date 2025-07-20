#!/usr/bin/env python3
"""
Alpha Vantage Data Collector - Professional financial data with 500 requests/day
High-quality data source for MVP RL trading system
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import json
import time
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class AlphaVantageCollector:
    """
    Alpha Vantage data collector for MVP RL trading system
    500 requests/day limit, high-quality financial data
    """
    
    def __init__(self, api_key: str = "4Y8CDGOF82KMK3R7"):
        """Initialize the Alpha Vantage collector"""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.logger = self._setup_logger()
        
        # Request tracking for rate limiting
        self.requests_made = 0
        self.max_requests_per_day = 500
        self.request_delay = 12  # 12 seconds between requests (5 per minute max)
        
        # MVP stock universe - prioritized list
        self.priority_stocks = [
            # High priority - major tech stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            
            # Medium priority - financials and blue chips
            "JPM", "BAC", "JNJ", "HD", "XOM",
            
            # Market benchmark
            "SPY"
        ]
        
        # Data storage
        self.data_root = project_root / 'data' / 'alpha_vantage'
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - ALPHA_VANTAGE - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def collect_daily_data(self, symbol: str, outputsize: str = "full") -> Optional[pd.DataFrame]:
        """
        Collect daily time series data for a symbol
        
        Args:
            symbol (str): Stock symbol
            outputsize (str): "compact" (100 days) or "full" (20+ years)
            
        Returns:
            pd.DataFrame: Daily OHLCV data with technical indicators
        """
        if self.requests_made >= self.max_requests_per_day:
            self.logger.warning(f"⚠️ Daily request limit reached ({self.max_requests_per_day})")
            return None
        
        self.logger.info(f"📊 Collecting daily data for {symbol}")
        
        # Rate limiting
        if self.requests_made > 0:
            self.logger.info(f"⏱️ Waiting {self.request_delay} seconds for rate limiting...")
            time.sleep(self.request_delay)
        
        try:
            # Make API request
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            self.requests_made += 1
            
            if response.status_code != 200:
                self.logger.error(f"❌ API request failed: {response.status_code}")
                return None
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"❌ API Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                self.logger.warning(f"⚠️ API Note: {data['Note']}")
                return None
            
            # Parse time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                self.logger.error(f"❌ No time series data found for {symbol}")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Adjusted_Close': float(values['5. adjusted close']),
                    'Volume': int(values['6. volume']),
                    'Dividend_Amount': float(values['7. dividend amount']),
                    'Split_Coefficient': float(values['8. split coefficient'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Add symbol and basic info
            df['symbol'] = symbol
            
            # Calculate technical indicators
            df_with_indicators = self._calculate_technical_indicators(df, symbol)
            
            self.logger.info(f"✅ {symbol}: {len(df_with_indicators)} records collected")
            
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"❌ {symbol}: Collection failed - {str(e)}")
            return None
    
    def collect_intraday_data(self, symbol: str, interval: str = "5min") -> Optional[pd.DataFrame]:
        """
        Collect intraday time series data
        
        Args:
            symbol (str): Stock symbol
            interval (str): "1min", "5min", "15min", "30min", "60min"
            
        Returns:
            pd.DataFrame: Intraday OHLCV data
        """
        if self.requests_made >= self.max_requests_per_day:
            self.logger.warning(f"⚠️ Daily request limit reached ({self.max_requests_per_day})")
            return None
        
        self.logger.info(f"📊 Collecting {interval} intraday data for {symbol}")
        
        # Rate limiting
        if self.requests_made > 0:
            time.sleep(self.request_delay)
        
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            self.requests_made += 1
            
            if response.status_code != 200:
                self.logger.error(f"❌ API request failed: {response.status_code}")
                return None
            
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data or 'Note' in data:
                self.logger.error(f"❌ API Error for {symbol}")
                return None
            
            # Parse time series data
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                self.logger.error(f"❌ No intraday data found for {symbol}")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for datetime_str, values in time_series.items():
                df_data.append({
                    'DateTime': pd.to_datetime(datetime_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('DateTime', inplace=True)
            df.sort_index(inplace=True)
            df['symbol'] = symbol
            
            self.logger.info(f"✅ {symbol}: {len(df)} intraday records collected")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ {symbol}: Intraday collection failed - {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators for the data"""
        df = data.copy()
        
        # Use adjusted close for calculations
        close_price = df['Adjusted_Close']
        
        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(close_price, 14)
        macd_line, signal_line, histogram = self._calculate_macd(close_price, 12, 26, 9)
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Stochastic oscillator
        df['stochastic_k'], df['stochastic_d'] = self._calculate_stochastic(df, 14, 3)
        
        # Trend indicators
        df['sma_5'] = close_price.rolling(window=5).mean()
        df['sma_10'] = close_price.rolling(window=10).mean()
        df['sma_20'] = close_price.rolling(window=20).mean()
        df['sma_50'] = close_price.rolling(window=50).mean()
        df['sma_200'] = close_price.rolling(window=200).mean()
        df['ema_12'] = close_price.ewm(span=12).mean()
        df['ema_26'] = close_price.ewm(span=26).mean()
        df['price_vs_sma20'] = close_price / df['sma_20']
        df['price_vs_sma50'] = close_price / df['sma_50']
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(close_price, 20, 2)
        df['bb_position'] = (close_price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility indicators
        df['atr_14'] = self._calculate_atr(df, 14)
        returns = close_price.pct_change()
        df['historical_vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['historical_vol_50'] = returns.rolling(window=50).std() * np.sqrt(252)
        
        # Support/Resistance
        df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['support_1'] = 2 * df['pivot_point'] - df['High']
        df['resistance_1'] = 2 * df['pivot_point'] - df['Low']
        
        # Returns for RL
        df['daily_return'] = close_price.pct_change()
        df['forward_return_1d'] = df['daily_return'].shift(-1)
        df['forward_return_5d'] = close_price.pct_change(5).shift(-5)
        
        # Additional features
        df['price_change'] = df['Close'] - df['Open']
        df['price_range'] = df['High'] - df['Low']
        df['gap'] = df['Open'] - df['Close'].shift(1)
        
        # Clean up
        df = df.dropna()
        
        return df
    
    def collect_batch_data(self, symbols: List[str] = None, max_symbols: int = 20) -> Dict[str, Any]:
        """
        Collect data for multiple symbols with rate limiting
        
        Args:
            symbols (List[str]): List of symbols to collect
            max_symbols (int): Maximum symbols to collect (rate limit protection)
            
        Returns:
            Dict: Collection results
        """
        symbols = symbols or self.priority_stocks[:max_symbols]
        
        self.logger.info(f"🚀 Starting Alpha Vantage batch collection")
        self.logger.info(f"📊 Symbols: {symbols}")
        self.logger.info(f"📞 Request limit: {self.max_requests_per_day - self.requests_made} remaining")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbols_processed': 0,
            'symbols_failed': 0,
            'data_files': [],
            'requests_used': 0,
            'summary': {}
        }
        
        for i, symbol in enumerate(symbols):
            if self.requests_made >= self.max_requests_per_day:
                self.logger.warning(f"⚠️ Stopping collection - daily limit reached")
                break
            
            self.logger.info(f"📊 Processing {symbol} ({i+1}/{len(symbols)})")
            
            # Collect daily data
            daily_data = self.collect_daily_data(symbol, outputsize="compact")  # Last 100 days
            
            if daily_data is not None and len(daily_data) > 50:
                # Save data
                filename = self._save_symbol_data(daily_data, symbol)
                results['data_files'].append(filename)
                results['symbols_processed'] += 1
                
                self.logger.info(f"✅ {symbol}: {len(daily_data)} records saved")
            else:
                results['symbols_failed'] += 1
                self.logger.warning(f"⚠️ {symbol}: Failed or insufficient data")
        
        results['requests_used'] = self.requests_made
        results['summary'] = {
            'success_rate': results['symbols_processed'] / len(symbols) if symbols else 0,
            'requests_remaining': self.max_requests_per_day - self.requests_made
        }
        
        # Save collection summary
        self._save_collection_summary(results)
        
        self.logger.info(f"🎉 Alpha Vantage collection completed:")
        self.logger.info(f"   ✅ Processed: {results['symbols_processed']}")
        self.logger.info(f"   ❌ Failed: {results['symbols_failed']}")
        self.logger.info(f"   📞 Requests used: {results['requests_used']}")
        self.logger.info(f"   📊 Success rate: {results['summary']['success_rate']:.2%}")
        
        return results
    
    # Helper methods for technical indicators (same as Yahoo collector)
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int, d_period: int):
        """Calculate Stochastic Oscillator"""
        lowest_low = data['Low'].rolling(window=k_period).min()
        highest_high = data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _save_symbol_data(self, data: pd.DataFrame, symbol: str) -> str:
        """Save processed data for a symbol"""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{symbol}_alpha_vantage_{timestamp}.csv"
        filepath = self.data_root / filename
        
        data.to_csv(filepath)
        return str(filepath)
    
    def _save_collection_summary(self, results: Dict):
        """Save collection summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.data_root / f"alpha_vantage_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def create_combined_dataset(self) -> pd.DataFrame:
        """Create combined dataset for RL training"""
        self.logger.info("📊 Creating Alpha Vantage RL training dataset")
        
        all_data = []
        
        for data_file in self.data_root.glob("*_alpha_vantage_*.csv"):
            try:
                symbol_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                all_data.append(symbol_data)
                self.logger.info(f"✅ Loaded {data_file.name}: {len(symbol_data)} records")
            except Exception as e:
                self.logger.error(f"❌ Failed to load {data_file.name}: {str(e)}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()
            
            timestamp = datetime.now().strftime("%Y%m%d")
            combined_file = self.data_root / f"alpha_vantage_rl_dataset_{timestamp}.csv"
            combined_data.to_csv(combined_file)
            
            self.logger.info(f"🎉 Alpha Vantage RL dataset created: {len(combined_data)} records")
            self.logger.info(f"   📁 Saved to: {combined_file}")
            self.logger.info(f"   📅 Date range: {combined_data.index.min().date()} to {combined_data.index.max().date()}")
            self.logger.info(f"   🏢 Symbols: {combined_data['symbol'].nunique()}")
            
            return combined_data
        else:
            self.logger.error("❌ No data files found")
            return pd.DataFrame()

def main():
    """Test the Alpha Vantage collector"""
    print("🧪 Testing Alpha Vantage Data Collector")
    print("=" * 42)
    
    collector = AlphaVantageCollector()
    
    # Collect data for top 10 priority stocks
    results = collector.collect_batch_data(max_symbols=10)
    
    print(f"\n📊 Collection Results:")
    print(f"   Symbols processed: {results['symbols_processed']}")
    print(f"   Symbols failed: {results['symbols_failed']}")
    print(f"   Requests used: {results['requests_used']}")
    print(f"   Success rate: {results['summary']['success_rate']:.2%}")
    print(f"   Requests remaining: {results['summary']['requests_remaining']}")
    
    # Create combined dataset
    if results['symbols_processed'] > 0:
        rl_dataset = collector.create_combined_dataset()
        if not rl_dataset.empty:
            print(f"\n🤖 RL Dataset Ready:")
            print(f"   Total records: {len(rl_dataset):,}")
            print(f"   Symbols: {rl_dataset['symbol'].nunique()}")
            print(f"   Features: {len(rl_dataset.columns)}")

if __name__ == "__main__":
    main()
