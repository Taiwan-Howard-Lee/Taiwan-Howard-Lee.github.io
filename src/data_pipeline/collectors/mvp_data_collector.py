#!/usr/bin/env python3
"""
MVP Data Collector - Simplified data collection for RL trading system
Focuses on company-level indicators with sector context
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data_collection.polygon_collector import PolygonCollector
from src.feature_engineering.enhanced_technical_indicators import EnhancedTechnicalIndicators

class MVPDataCollector:
    """
    MVP Data Collector for RL Trading System
    Collects and processes data for company-level technical analysis
    """
    
    def __init__(self):
        """Initialize the MVP data collector"""
        self.polygon_collector = PolygonCollector()
        self.tech_indicators = EnhancedTechnicalIndicators()
        self.logger = self._setup_logger()
        
        # MVP stock universe - focus on liquid, major stocks
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
        
        # Sector ETFs for context
        self.sector_etfs = [
            "XLK", "XLF", "XLI", "XLV", "XLY", "XLP", 
            "XLE", "XLB", "XLC", "XLRE", "XLU"
        ]
        
        # Market benchmark
        self.benchmark = "SPY"
        
        # Data storage
        self.data_root = project_root / 'data' / 'mvp'
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - MVP_COLLECTOR - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def collect_daily_data(self, days_back: int = 100) -> Dict[str, Any]:
        """
        Collect daily data for all symbols in the universe
        
        Args:
            days_back (int): Number of days of historical data to collect
            
        Returns:
            Dict: Collection results
        """
        self.logger.info(f"🚀 Starting MVP daily data collection ({days_back} days)")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'days_collected': days_back,
            'symbols_processed': 0,
            'symbols_failed': 0,
            'data_files': [],
            'summary': {}
        }
        
        all_symbols = list(self.stock_universe.keys()) + self.sector_etfs + [self.benchmark]
        
        for symbol in all_symbols:
            try:
                self.logger.info(f"📊 Collecting data for {symbol}")
                
                # Get historical data
                historical_data = self.polygon_collector.get_aggregates(symbol, days=days_back)
                
                if historical_data is not None and len(historical_data) > 50:
                    # Calculate technical indicators
                    indicators = self.tech_indicators.calculate_company_indicators(historical_data, symbol)
                    
                    # Add sector context
                    sector = self.stock_universe.get(symbol, 'ETF' if symbol in self.sector_etfs else 'BENCHMARK')
                    
                    # Prepare final dataset
                    final_data = self._prepare_final_dataset(historical_data, indicators, symbol, sector)
                    
                    # Save data
                    filename = self._save_symbol_data(final_data, symbol)
                    results['data_files'].append(filename)
                    results['symbols_processed'] += 1
                    
                    self.logger.info(f"✅ {symbol}: {len(final_data)} records processed")
                    
                else:
                    self.logger.warning(f"⚠️ {symbol}: Insufficient data ({len(historical_data) if historical_data is not None else 0} records)")
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
        
        self.logger.info(f"🎉 MVP collection completed:")
        self.logger.info(f"   ✅ Processed: {results['symbols_processed']}")
        self.logger.info(f"   ❌ Failed: {results['symbols_failed']}")
        self.logger.info(f"   📊 Success rate: {results['summary']['success_rate']:.2%}")
        
        return results
    
    def _prepare_final_dataset(self, ohlcv_data: pd.DataFrame, indicators: Dict, symbol: str, sector: str) -> pd.DataFrame:
        """
        Prepare the final dataset combining OHLCV and technical indicators
        
        Args:
            ohlcv_data (pd.DataFrame): Raw OHLCV data
            indicators (Dict): Calculated technical indicators
            symbol (str): Stock symbol
            sector (str): Sector classification
            
        Returns:
            pd.DataFrame: Final dataset ready for ML
        """
        # Start with OHLCV data
        final_data = ohlcv_data.copy()
        
        # Add basic info
        final_data['symbol'] = symbol
        final_data['sector'] = sector
        
        # Add technical indicators as rolling calculations
        # (Note: The indicators dict contains latest values, we need to calculate for all periods)
        
        # Volume indicators
        final_data['volume_sma_20'] = final_data['Volume'].rolling(window=20).mean()
        final_data['volume_ratio'] = final_data['Volume'] / final_data['volume_sma_20']
        
        # Momentum indicators
        final_data['rsi_14'] = self._calculate_rsi(final_data['Close'], 14)
        macd_line, signal_line, histogram = self._calculate_macd(final_data['Close'], 12, 26, 9)
        final_data['macd_line'] = macd_line
        final_data['macd_signal'] = signal_line
        final_data['macd_histogram'] = histogram
        
        # Trend indicators
        final_data['sma_20'] = final_data['Close'].rolling(window=20).mean()
        final_data['sma_50'] = final_data['Close'].rolling(window=50).mean()
        final_data['ema_12'] = final_data['Close'].ewm(span=12).mean()
        final_data['price_vs_sma20'] = final_data['Close'] / final_data['sma_20']
        
        # Volatility indicators
        final_data['atr_14'] = self._calculate_atr(final_data, 14)
        returns = final_data['Close'].pct_change()
        final_data['historical_vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Support/Resistance (simple pivot points)
        final_data['pivot_point'] = (final_data['High'] + final_data['Low'] + final_data['Close']) / 3
        final_data['support_1'] = 2 * final_data['pivot_point'] - final_data['High']
        final_data['resistance_1'] = 2 * final_data['pivot_point'] - final_data['Low']
        
        # Returns for RL rewards
        final_data['daily_return'] = final_data['Close'].pct_change()
        final_data['forward_return_1d'] = final_data['daily_return'].shift(-1)  # Next day return for RL
        
        # Clean up NaN values
        final_data = final_data.dropna()
        
        return final_data
    
    def _save_symbol_data(self, data: pd.DataFrame, symbol: str) -> str:
        """Save processed data for a symbol"""
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{symbol}_mvp_data_{timestamp}.csv"
        filepath = self.data_root / filename
        
        data.to_csv(filepath)
        return str(filepath)
    
    def _save_collection_summary(self, results: Dict):
        """Save collection summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.data_root / f"collection_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Helper methods for technical indicators
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
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def create_rl_dataset(self) -> pd.DataFrame:
        """
        Create a combined dataset for RL training
        
        Returns:
            pd.DataFrame: Combined dataset with all symbols
        """
        self.logger.info("📊 Creating RL training dataset")
        
        all_data = []
        
        # Load all symbol data files
        for data_file in self.data_root.glob("*_mvp_data_*.csv"):
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
            combined_file = self.data_root / f"rl_training_dataset_{timestamp}.csv"
            combined_data.to_csv(combined_file)
            
            self.logger.info(f"🎉 RL dataset created: {len(combined_data)} total records")
            self.logger.info(f"   📁 Saved to: {combined_file}")
            
            return combined_data
        else:
            self.logger.error("❌ No data files found to combine")
            return pd.DataFrame()

def main():
    """Test the MVP data collector"""
    print("🧪 Testing MVP Data Collector")
    print("=" * 35)
    
    collector = MVPDataCollector()
    
    # Collect data for the past 100 days
    results = collector.collect_daily_data(days_back=100)
    
    print(f"\n📊 Collection Results:")
    print(f"   Symbols processed: {results['symbols_processed']}")
    print(f"   Symbols failed: {results['symbols_failed']}")
    print(f"   Success rate: {results['summary']['success_rate']:.2%}")
    print(f"   Data files created: {len(results['data_files'])}")
    
    # Create RL dataset
    rl_dataset = collector.create_rl_dataset()
    if not rl_dataset.empty:
        print(f"\n🤖 RL Dataset:")
        print(f"   Total records: {len(rl_dataset)}")
        print(f"   Date range: {rl_dataset.index.min()} to {rl_dataset.index.max()}")
        print(f"   Symbols: {rl_dataset['symbol'].nunique()}")
        print(f"   Features: {len(rl_dataset.columns)}")

if __name__ == "__main__":
    main()
