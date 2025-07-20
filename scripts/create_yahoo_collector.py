#!/usr/bin/env python3
"""
Yahoo Finance Data Collector - Unlimited data collection
Alternative to Polygon.io for MVP development
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class YahooFinanceCollector:
    """
    Yahoo Finance data collector for MVP RL trading system
    No rate limits, perfect for development and testing
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance collector"""
        self.logger = self._setup_logger()
        
        # MVP stock universe
        self.stock_universe = {
            # Technology (XLK)
            "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "AMZN": "XLK", 
            "TSLA": "XLK", "NVDA": "XLK", "META": "XLK",
            
            # Financials (XLF)  
            "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF",
            
            # Healthcare (XLV)
            "JNJ": "XLV", "PFE": "XLV", "UNH": "XLV",
            
            # Energy (XLE)
            "XOM": "XLE", "CVX": "XLE",
            
            # Consumer Discretionary (XLY)
            "HD": "XLY", "MCD": "XLY", "NKE": "XLY"
        }
        
        # Sector ETFs
        self.sector_etfs = [
            "XLK", "XLF", "XLI", "XLV", "XLY", "XLP", 
            "XLE", "XLB", "XLC", "XLRE", "XLU"
        ]
        
        # Market benchmark
        self.benchmark = "SPY"
        
        # Data storage
        self.data_root = project_root / 'data' / 'yahoo_mvp'
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - YAHOO_COLLECTOR - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def collect_all_data(self, period: str = "1y") -> Dict[str, Any]:
        """
        Collect data for all symbols using Yahoo Finance
        
        Args:
            period (str): Data period ('1y', '2y', '5y', 'max')
            
        Returns:
            Dict: Collection results
        """
        self.logger.info(f"🚀 Starting Yahoo Finance data collection ({period})")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'symbols_processed': 0,
            'symbols_failed': 0,
            'data_files': [],
            'summary': {}
        }
        
        all_symbols = list(self.stock_universe.keys()) + self.sector_etfs + [self.benchmark]
        
        for symbol in all_symbols:
            try:
                self.logger.info(f"📊 Collecting data for {symbol}")
                
                # Download data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period=period)
                
                if len(hist_data) > 50:
                    # Add sector information
                    sector = self.stock_universe.get(symbol, 'ETF' if symbol in self.sector_etfs else 'BENCHMARK')
                    
                    # Calculate technical indicators
                    processed_data = self._calculate_technical_indicators(hist_data, symbol, sector)
                    
                    # Save data
                    filename = self._save_symbol_data(processed_data, symbol)
                    results['data_files'].append(filename)
                    results['symbols_processed'] += 1
                    
                    self.logger.info(f"✅ {symbol}: {len(processed_data)} records processed")
                    
                else:
                    self.logger.warning(f"⚠️ {symbol}: Insufficient data ({len(hist_data)} records)")
                    results['symbols_failed'] += 1
                    
            except Exception as e:
                self.logger.error(f"❌ {symbol}: Collection failed - {str(e)}")
                results['symbols_failed'] += 1
        
        # Create summary
        results['summary'] = {
            'total_symbols': len(all_symbols),
            'success_rate': results['symbols_processed'] / len(all_symbols),
            'stocks': len([s for s in all_symbols if s in self.stock_universe]),
            'etfs': len(self.sector_etfs),
            'benchmark': 1
        }
        
        # Save collection summary
        self._save_collection_summary(results)
        
        self.logger.info(f"🎉 Yahoo Finance collection completed:")
        self.logger.info(f"   ✅ Processed: {results['symbols_processed']}")
        self.logger.info(f"   ❌ Failed: {results['symbols_failed']}")
        self.logger.info(f"   📊 Success rate: {results['summary']['success_rate']:.2%}")
        
        return results
    
    def _calculate_technical_indicators(self, data: pd.DataFrame, symbol: str, sector: str) -> pd.DataFrame:
        """
        Calculate technical indicators for the data
        
        Args:
            data (pd.DataFrame): OHLCV data from Yahoo Finance
            symbol (str): Stock symbol
            sector (str): Sector classification
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        # Yahoo Finance uses different column names
        df = data.copy()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        
        # Add basic info
        df['symbol'] = symbol
        df['sector'] = sector
        
        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        macd_line, signal_line, histogram = self._calculate_macd(df['Close'], 12, 26, 9)
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Stochastic oscillator
        df['stochastic_k'], df['stochastic_d'] = self._calculate_stochastic(df, 14, 3)
        
        # Trend indicators
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        df['price_vs_sma20'] = df['Close'] / df['sma_20']
        df['price_vs_sma50'] = df['Close'] / df['sma_50']
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['Close'], 20, 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility indicators
        df['atr_14'] = self._calculate_atr(df, 14)
        returns = df['Close'].pct_change()
        df['historical_vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['historical_vol_50'] = returns.rolling(window=50).std() * np.sqrt(252)
        
        # Support/Resistance (simple pivot points)
        df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['support_1'] = 2 * df['pivot_point'] - df['High']
        df['resistance_1'] = 2 * df['pivot_point'] - df['Low']
        
        # Returns for RL rewards
        df['daily_return'] = df['Close'].pct_change()
        df['forward_return_1d'] = df['daily_return'].shift(-1)  # Next day return for RL
        df['forward_return_5d'] = df['Close'].pct_change(5).shift(-5)  # 5-day forward return
        
        # Additional features
        df['price_change'] = df['Close'] - df['Open']
        df['price_range'] = df['High'] - df['Low']
        df['gap'] = df['Open'] - df['Close'].shift(1)
        
        # Clean up columns we don't need
        df = df.drop(['Dividends', 'Stock Splits'], axis=1)
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
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
        filename = f"{symbol}_yahoo_data_{timestamp}.csv"
        filepath = self.data_root / filename
        
        data.to_csv(filepath)
        return str(filepath)
    
    def _save_collection_summary(self, results: Dict):
        """Save collection summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.data_root / f"yahoo_collection_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def create_combined_dataset(self) -> pd.DataFrame:
        """
        Create a combined dataset for RL training
        
        Returns:
            pd.DataFrame: Combined dataset with all symbols
        """
        self.logger.info("📊 Creating combined RL training dataset")
        
        all_data = []
        
        # Load all symbol data files
        for data_file in self.data_root.glob("*_yahoo_data_*.csv"):
            try:
                symbol_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                all_data.append(symbol_data)
                self.logger.info(f"✅ Loaded {data_file.name}: {len(symbol_data)} records")
            except Exception as e:
                self.logger.error(f"❌ Failed to load {data_file.name}: {str(e)}")
        
        if all_data:
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()  # Sort by date
            
            # Save combined dataset
            timestamp = datetime.now().strftime("%Y%m%d")
            combined_file = self.data_root / f"yahoo_rl_training_dataset_{timestamp}.csv"
            combined_data.to_csv(combined_file)
            
            self.logger.info(f"🎉 Combined RL dataset created: {len(combined_data)} total records")
            self.logger.info(f"   📁 Saved to: {combined_file}")
            self.logger.info(f"   📅 Date range: {combined_data.index.min()} to {combined_data.index.max()}")
            self.logger.info(f"   🏢 Symbols: {combined_data['symbol'].nunique()}")
            self.logger.info(f"   📊 Features: {len(combined_data.columns)}")
            
            return combined_data
        else:
            self.logger.error("❌ No data files found to combine")
            return pd.DataFrame()

def main():
    """Run the Yahoo Finance data collector"""
    print("🧪 Testing Yahoo Finance Data Collector")
    print("=" * 42)
    
    # Install yfinance if not available
    try:
        import yfinance as yf
    except ImportError:
        print("📦 Installing yfinance...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance as yf
    
    collector = YahooFinanceCollector()
    
    # Collect data for the past year
    results = collector.collect_all_data(period="1y")
    
    print(f"\n📊 Collection Results:")
    print(f"   Symbols processed: {results['symbols_processed']}")
    print(f"   Symbols failed: {results['symbols_failed']}")
    print(f"   Success rate: {results['summary']['success_rate']:.2%}")
    print(f"   Data files created: {len(results['data_files'])}")
    
    # Create combined RL dataset
    rl_dataset = collector.create_combined_dataset()
    if not rl_dataset.empty:
        print(f"\n🤖 RL Dataset Ready:")
        print(f"   Total records: {len(rl_dataset):,}")
        print(f"   Date range: {rl_dataset.index.min().date()} to {rl_dataset.index.max().date()}")
        print(f"   Symbols: {rl_dataset['symbol'].nunique()}")
        print(f"   Features: {len(rl_dataset.columns)}")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Test RL agent: python3 src/models/rl_trading_agent.py")
        print(f"   2. Run backtest: python3 scripts/backtest_rl_agent.py")
        print(f"   3. Optimize parameters: python3 scripts/optimize_parameters.py")

if __name__ == "__main__":
    main()
