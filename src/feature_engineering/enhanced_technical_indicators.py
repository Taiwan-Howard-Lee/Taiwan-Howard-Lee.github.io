#!/usr/bin/env python3
"""
Enhanced Technical Indicators - Comprehensive 6-category framework
Implements company-level and market-level technical analysis metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path
import json

class EnhancedTechnicalIndicators:
    """
    Comprehensive technical indicators following the 6-category framework:
    1. Volume, 2. Momentum, 3. Trend, 4. Volatility, 5. Breadth, 6. Support/Resistance
    
    Supports both company-level and market-level calculations
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize with configuration
        
        Args:
            config_path (str): Path to technical analysis configuration
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # Results storage
        self.company_metrics = {}
        self.market_metrics = {}
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load technical analysis configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "company_level": {
                "volume": {
                    "basic": {
                        "volume_sma": {"period": 20},
                        "volume_ratio": {"period": 10}
                    },
                    "advanced": {
                        "obv": {"enabled": True},
                        "vwap": {"enabled": True}
                    }
                },
                "momentum": {
                    "basic": {
                        "rsi": {"period": 14},
                        "stochastic": {"k_period": 14, "d_period": 3}
                    },
                    "advanced": {
                        "macd": {"fast": 12, "slow": 26, "signal": 9}
                    }
                },
                "trend": {
                    "basic": {
                        "sma": {"periods": [5, 10, 20, 50, 200]},
                        "ema": {"periods": [12, 26]}
                    },
                    "advanced": {
                        "bollinger_bands": {"period": 20, "std_dev": 2}
                    }
                },
                "volatility": {
                    "basic": {
                        "atr": {"period": 14},
                        "historical_volatility": {"period": 20}
                    }
                }
            },
            "market_level": {
                "breadth": {
                    "basic": {
                        "advance_decline_ratio": {"enabled": True}
                    }
                }
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - TECH_INDICATORS - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_company_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate all company-level technical indicators
        
        Args:
            data (pd.DataFrame): OHLCV data for the company
            symbol (str): Stock symbol
            
        Returns:
            Dict: All calculated indicators organized by category
        """
        self.logger.info(f"Calculating company indicators for {symbol}")
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'scope_level': 'company',
            'categories': {}
        }
        
        # 1. VOLUME INDICATORS
        if 'volume' in self.config['company_level']:
            results['categories']['volume'] = self._calculate_volume_indicators(data)
        
        # 2. MOMENTUM INDICATORS  
        if 'momentum' in self.config['company_level']:
            results['categories']['momentum'] = self._calculate_momentum_indicators(data)
        
        # 3. TREND INDICATORS
        if 'trend' in self.config['company_level']:
            results['categories']['trend'] = self._calculate_trend_indicators(data)
        
        # 4. VOLATILITY INDICATORS
        if 'volatility' in self.config['company_level']:
            results['categories']['volatility'] = self._calculate_volatility_indicators(data)
        
        # 5. SUPPORT/RESISTANCE (company-level)
        results['categories']['support_resistance'] = self._calculate_support_resistance(data)
        
        self.company_metrics[symbol] = results
        return results
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        volume_config = self.config['company_level']['volume']
        indicators = {'basic': {}, 'advanced': {}}
        
        # Basic Volume Indicators
        if 'basic' in volume_config:
            basic_config = volume_config['basic']
            
            # Volume SMA
            if 'volume_sma' in basic_config:
                period = basic_config['volume_sma']['period']
                indicators['basic']['volume_sma'] = data['Volume'].rolling(window=period).mean().iloc[-1]
            
            # Volume Ratio (current vs average)
            if 'volume_ratio' in basic_config:
                period = basic_config['volume_ratio']['period']
                avg_volume = data['Volume'].rolling(window=period).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                indicators['basic']['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Advanced Volume Indicators
        if 'advanced' in volume_config:
            advanced_config = volume_config['advanced']
            
            # On-Balance Volume (OBV)
            if advanced_config.get('obv', {}).get('enabled', False):
                obv = self._calculate_obv(data)
                indicators['advanced']['obv'] = obv.iloc[-1]
            
            # Volume Weighted Average Price (VWAP)
            if advanced_config.get('vwap', {}).get('enabled', False):
                vwap = self._calculate_vwap(data)
                indicators['advanced']['vwap'] = vwap.iloc[-1]
        
        return indicators
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-based indicators"""
        momentum_config = self.config['company_level']['momentum']
        indicators = {'basic': {}, 'advanced': {}}
        
        # Basic Momentum Indicators
        if 'basic' in momentum_config:
            basic_config = momentum_config['basic']
            
            # RSI
            if 'rsi' in basic_config:
                period = basic_config['rsi']['period']
                rsi = self._calculate_rsi(data['Close'], period)
                indicators['basic']['rsi'] = rsi.iloc[-1]
            
            # Stochastic Oscillator
            if 'stochastic' in basic_config:
                k_period = basic_config['stochastic']['k_period']
                d_period = basic_config['stochastic']['d_period']
                stoch_k, stoch_d = self._calculate_stochastic(data, k_period, d_period)
                indicators['basic']['stochastic_k'] = stoch_k.iloc[-1]
                indicators['basic']['stochastic_d'] = stoch_d.iloc[-1]
        
        # Advanced Momentum Indicators
        if 'advanced' in momentum_config:
            advanced_config = momentum_config['advanced']
            
            # MACD
            if 'macd' in advanced_config:
                fast = advanced_config['macd']['fast']
                slow = advanced_config['macd']['slow']
                signal = advanced_config['macd']['signal']
                macd_line, signal_line, histogram = self._calculate_macd(data['Close'], fast, slow, signal)
                indicators['advanced']['macd_line'] = macd_line.iloc[-1]
                indicators['advanced']['macd_signal'] = signal_line.iloc[-1]
                indicators['advanced']['macd_histogram'] = histogram.iloc[-1]
        
        return indicators
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend-based indicators"""
        trend_config = self.config['company_level']['trend']
        indicators = {'basic': {}, 'advanced': {}}
        
        # Basic Trend Indicators
        if 'basic' in trend_config:
            basic_config = trend_config['basic']
            
            # Simple Moving Averages
            if 'sma' in basic_config:
                for period in basic_config['sma']['periods']:
                    sma = data['Close'].rolling(window=period).mean()
                    indicators['basic'][f'sma_{period}'] = sma.iloc[-1]
            
            # Exponential Moving Averages
            if 'ema' in basic_config:
                for period in basic_config['ema']['periods']:
                    ema = data['Close'].ewm(span=period).mean()
                    indicators['basic'][f'ema_{period}'] = ema.iloc[-1]
        
        # Advanced Trend Indicators
        if 'advanced' in trend_config:
            advanced_config = trend_config['advanced']
            
            # Bollinger Bands
            if 'bollinger_bands' in advanced_config:
                period = advanced_config['bollinger_bands']['period']
                std_dev = advanced_config['bollinger_bands']['std_dev']
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['Close'], period, std_dev)
                indicators['advanced']['bb_upper'] = bb_upper.iloc[-1]
                indicators['advanced']['bb_middle'] = bb_middle.iloc[-1]
                indicators['advanced']['bb_lower'] = bb_lower.iloc[-1]
                
                # Bollinger Band Position
                current_price = data['Close'].iloc[-1]
                bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                indicators['advanced']['bb_position'] = bb_position
        
        return indicators
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-based indicators"""
        volatility_config = self.config['company_level']['volatility']
        indicators = {'basic': {}}
        
        if 'basic' in volatility_config:
            basic_config = volatility_config['basic']
            
            # Average True Range (ATR)
            if 'atr' in basic_config:
                period = basic_config['atr']['period']
                atr = self._calculate_atr(data, period)
                indicators['basic']['atr'] = atr.iloc[-1]
            
            # Historical Volatility
            if 'historical_volatility' in basic_config:
                period = basic_config['historical_volatility']['period']
                returns = data['Close'].pct_change()
                hist_vol = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
                indicators['basic']['historical_volatility'] = hist_vol.iloc[-1]
        
        return indicators
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        indicators = {'basic': {}}
        
        # Simple pivot points
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        
        indicators['basic']['pivot_point'] = pivot
        indicators['basic']['resistance_1'] = r1
        indicators['basic']['support_1'] = s1
        
        return indicators
    
    # Helper calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = data['Low'].rolling(window=k_period).min()
        highest_high = data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['Volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        return (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

def main():
    """Test the enhanced technical indicators"""
    print("🧪 Testing Enhanced Technical Indicators")
    print("=" * 45)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    volumes = np.random.randint(1000000, 10000000, 100)
    
    sample_data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)
    
    # Initialize indicators
    indicators = EnhancedTechnicalIndicators()
    
    # Calculate company indicators
    results = indicators.calculate_company_indicators(sample_data, 'TEST')
    
    print(f"📊 Results for {results['symbol']}:")
    print(f"   Scope: {results['scope_level']}")
    print(f"   Categories: {len(results['categories'])}")
    
    for category, metrics in results['categories'].items():
        print(f"\n📈 {category.upper()}:")
        for level, indicators_dict in metrics.items():
            if indicators_dict:
                print(f"   {level}: {len(indicators_dict)} indicators")
                for name, value in list(indicators_dict.items())[:3]:  # Show first 3
                    print(f"     {name}: {value:.4f}")

if __name__ == "__main__":
    main()
