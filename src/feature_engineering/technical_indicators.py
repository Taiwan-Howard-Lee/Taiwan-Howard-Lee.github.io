"""
Technical Indicators - Comprehensive technical analysis indicators for financial data
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Install with: pip install ta-lib")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("pandas-ta not available. Install with: pip install pandas-ta")


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator for financial time series data.
    Supports 50+ technical indicators across different categories.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV columns and datetime index
        """
        self.data = data.copy()
        self.logger = self._setup_logger()
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the technical indicators."""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def moving_averages(self, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate various moving averages.
        
        Args:
            periods (List[int]): List of periods for moving averages
            
        Returns:
            pd.DataFrame: DataFrame with moving average columns
        """
        result = pd.DataFrame(index=self.data.index)
        
        for period in periods:
            # Simple Moving Average
            result[f'SMA_{period}'] = self.data['Close'].rolling(window=period).mean()
            
            # Exponential Moving Average
            result[f'EMA_{period}'] = self.data['Close'].ewm(span=period).mean()
            
            # Weighted Moving Average (if TA-Lib available)
            if TALIB_AVAILABLE:
                result[f'WMA_{period}'] = talib.WMA(self.data['Close'].values, timeperiod=period)
        
        self.logger.info(f"Calculated moving averages for periods: {periods}")
        return result
    
    def momentum_indicators(self) -> pd.DataFrame:
        """
        Calculate momentum-based indicators.
        
        Returns:
            pd.DataFrame: DataFrame with momentum indicators
        """
        result = pd.DataFrame(index=self.data.index)
        
        # RSI (Relative Strength Index)
        if TALIB_AVAILABLE:
            result['RSI_14'] = talib.RSI(self.data['Close'].values, timeperiod=14)
        else:
            result['RSI_14'] = self._calculate_rsi(self.data['Close'], period=14)
        
        # MACD (Moving Average Convergence Divergence)
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(self.data['Close'].values)
            result['MACD'] = macd
            result['MACD_Signal'] = macd_signal
            result['MACD_Histogram'] = macd_hist
        else:
            macd_data = self._calculate_macd(self.data['Close'])
            result = pd.concat([result, macd_data], axis=1)
        
        # Stochastic Oscillator
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(self.data['High'].values, 
                                     self.data['Low'].values, 
                                     self.data['Close'].values)
            result['Stoch_K'] = slowk
            result['Stoch_D'] = slowd
        
        # Williams %R
        if TALIB_AVAILABLE:
            result['Williams_R'] = talib.WILLR(self.data['High'].values,
                                             self.data['Low'].values,
                                             self.data['Close'].values,
                                             timeperiod=14)
        
        # Commodity Channel Index (CCI)
        if TALIB_AVAILABLE:
            result['CCI'] = talib.CCI(self.data['High'].values,
                                    self.data['Low'].values,
                                    self.data['Close'].values,
                                    timeperiod=14)
        
        self.logger.info("Calculated momentum indicators")
        return result
    
    def volatility_indicators(self) -> pd.DataFrame:
        """
        Calculate volatility-based indicators.
        
        Returns:
            pd.DataFrame: DataFrame with volatility indicators
        """
        result = pd.DataFrame(index=self.data.index)
        
        # Bollinger Bands
        if TALIB_AVAILABLE:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(self.data['Close'].values,
                                                        timeperiod=20,
                                                        nbdevup=2,
                                                        nbdevdn=2)
            result['BB_Upper'] = bb_upper
            result['BB_Middle'] = bb_middle
            result['BB_Lower'] = bb_lower
            result['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            result['BB_Position'] = (self.data['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Average True Range (ATR)
        if TALIB_AVAILABLE:
            result['ATR_14'] = talib.ATR(self.data['High'].values,
                                       self.data['Low'].values,
                                       self.data['Close'].values,
                                       timeperiod=14)
        else:
            result['ATR_14'] = self._calculate_atr(period=14)
        
        # Keltner Channels
        if PANDAS_TA_AVAILABLE:
            kc = ta.kc(self.data['High'], self.data['Low'], self.data['Close'])
            if kc is not None:
                result = pd.concat([result, kc], axis=1)
        
        # Historical Volatility
        result['HV_20'] = self.data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        self.logger.info("Calculated volatility indicators")
        return result
    
    def volume_indicators(self) -> pd.DataFrame:
        """
        Calculate volume-based indicators.
        
        Returns:
            pd.DataFrame: DataFrame with volume indicators
        """
        result = pd.DataFrame(index=self.data.index)
        
        # On-Balance Volume (OBV)
        if TALIB_AVAILABLE:
            result['OBV'] = talib.OBV(self.data['Close'].values, self.data['Volume'].values)
        else:
            result['OBV'] = self._calculate_obv()
        
        # Volume Weighted Average Price (VWAP)
        result['VWAP'] = self._calculate_vwap()
        
        # Volume Rate of Change
        result['Volume_ROC'] = self.data['Volume'].pct_change(periods=10)
        
        # Accumulation/Distribution Line
        if TALIB_AVAILABLE:
            result['AD_Line'] = talib.AD(self.data['High'].values,
                                       self.data['Low'].values,
                                       self.data['Close'].values,
                                       self.data['Volume'].values)
        
        # Volume Moving Averages
        result['Volume_SMA_20'] = self.data['Volume'].rolling(20).mean()
        result['Volume_Ratio'] = self.data['Volume'] / result['Volume_SMA_20']
        
        self.logger.info("Calculated volume indicators")
        return result
    
    def trend_indicators(self) -> pd.DataFrame:
        """
        Calculate trend-based indicators.
        
        Returns:
            pd.DataFrame: DataFrame with trend indicators
        """
        result = pd.DataFrame(index=self.data.index)
        
        # Average Directional Index (ADX)
        if TALIB_AVAILABLE:
            result['ADX'] = talib.ADX(self.data['High'].values,
                                    self.data['Low'].values,
                                    self.data['Close'].values,
                                    timeperiod=14)
        
        # Aroon Oscillator
        if TALIB_AVAILABLE:
            aroon_down, aroon_up = talib.AROON(self.data['High'].values,
                                             self.data['Low'].values,
                                             timeperiod=14)
            result['Aroon_Up'] = aroon_up
            result['Aroon_Down'] = aroon_down
            result['Aroon_Oscillator'] = aroon_up - aroon_down
        
        # Parabolic SAR
        if TALIB_AVAILABLE:
            result['PSAR'] = talib.SAR(self.data['High'].values, self.data['Low'].values)
        
        # Supertrend (custom implementation)
        result = pd.concat([result, self._calculate_supertrend()], axis=1)
        
        self.logger.info("Calculated trend indicators")
        return result
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually if TA-Lib not available."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate MACD manually if TA-Lib not available."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Histogram': histogram
        })
    
    def _calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate ATR manually if TA-Lib not available."""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_obv(self) -> pd.Series:
        """Calculate OBV manually if TA-Lib not available."""
        obv = pd.Series(index=self.data.index, dtype=float)
        obv.iloc[0] = self.data['Volume'].iloc[0]
        
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.data['Volume'].iloc[i]
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vwap(self) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        return (typical_price * self.data['Volume']).cumsum() / self.data['Volume'].cumsum()
    
    def _calculate_supertrend(self, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate Supertrend indicator."""
        hl2 = (self.data['High'] + self.data['Low']) / 2
        atr = self._calculate_atr(period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize supertrend
        supertrend = pd.Series(index=self.data.index, dtype=float)
        direction = pd.Series(index=self.data.index, dtype=int)
        
        for i in range(len(self.data)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if self.data['Close'].iloc[i] <= supertrend.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
        
        return pd.DataFrame({
            'Supertrend': supertrend,
            'Supertrend_Direction': direction
        })
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Returns:
            pd.DataFrame: DataFrame with all calculated indicators
        """
        self.logger.info("Calculating all technical indicators...")
        
        # Calculate all indicator categories
        ma_indicators = self.moving_averages()
        momentum_indicators = self.momentum_indicators()
        volatility_indicators = self.volatility_indicators()
        volume_indicators = self.volume_indicators()
        trend_indicators = self.trend_indicators()
        
        # Combine all indicators
        all_indicators = pd.concat([
            ma_indicators,
            momentum_indicators,
            volatility_indicators,
            volume_indicators,
            trend_indicators
        ], axis=1)
        
        self.logger.info(f"Calculated {len(all_indicators.columns)} technical indicators")
        return all_indicators


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    
    # Calculate indicators
    indicators = TechnicalIndicators(data)
    all_indicators = indicators.calculate_all_indicators()
    
    print(f"Calculated {len(all_indicators.columns)} indicators:")
    print(all_indicators.columns.tolist())
    print(f"\nLatest values:")
    print(all_indicators.tail(1))
