"""
Apple Data Collector - Main data collection class for AAPL stock data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import time

class AppleDataCollector:
    """
    Main data collector for Apple (AAPL) stock data and related market data.
    Handles multiple data sources and timeframes with error handling and validation.
    """
    
    def __init__(self, ticker: str = "AAPL"):
        """
        Initialize the Apple data collector.
        
        Args:
            ticker (str): Stock ticker symbol, defaults to "AAPL"
        """
        self.ticker = ticker
        self.logger = self._setup_logger()
        
        # Market context tickers
        self.market_tickers = {
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'VIX': 'CBOE Volatility Index',
            'DXY': 'US Dollar Index',
            '^TNX': '10-Year Treasury Note Yield'
        }
        
        # Sector ETFs
        self.sector_tickers = {
            'XLK': 'Technology Select Sector SPDR Fund',
            'XLF': 'Financial Select Sector SPDR Fund',
            'XLE': 'Energy Select Sector SPDR Fund'
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the data collector."""
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
    
    def fetch_daily_data(self, period: str = "5y") -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV data for Apple stock.
        
        Args:
            period (str): Time period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            pd.DataFrame: Daily OHLCV data with datetime index
        """
        try:
            self.logger.info(f"Fetching daily data for {self.ticker} with period {period}")
            
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(period=period)
            
            if data.empty:
                self.logger.error(f"No data returned for {self.ticker}")
                return None
            
            # Validate data
            if self._validate_data(data):
                self.logger.info(f"Successfully fetched {len(data)} days of data")
                return data
            else:
                self.logger.error("Data validation failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching daily data: {str(e)}")
            return None
    
    def fetch_intraday_data(self, period: str = "1mo", interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Fetch intraday data for Apple stock.
        
        Args:
            period (str): Time period to fetch
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            pd.DataFrame: Intraday OHLCV data with datetime index
        """
        try:
            self.logger.info(f"Fetching intraday data for {self.ticker} with period {period} and interval {interval}")
            
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.error(f"No intraday data returned for {self.ticker}")
                return None
            
            if self._validate_data(data):
                self.logger.info(f"Successfully fetched {len(data)} intraday records")
                return data
            else:
                self.logger.error("Intraday data validation failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching intraday data: {str(e)}")
            return None
    
    def fetch_market_data(self, period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        Fetch market context data (SPY, QQQ, VIX, etc.).
        
        Args:
            period (str): Time period to fetch
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of market data by ticker
        """
        market_data = {}
        
        for ticker, description in self.market_tickers.items():
            try:
                self.logger.info(f"Fetching market data for {ticker} ({description})")
                
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period=period)
                
                if not data.empty and self._validate_data(data):
                    market_data[ticker] = data
                    self.logger.info(f"Successfully fetched {len(data)} records for {ticker}")
                else:
                    self.logger.warning(f"Failed to fetch data for {ticker}")
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        return market_data
    
    def fetch_sector_etfs(self, period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        Fetch sector ETF data.
        
        Args:
            period (str): Time period to fetch
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of sector ETF data by ticker
        """
        sector_data = {}
        
        for ticker, description in self.sector_tickers.items():
            try:
                self.logger.info(f"Fetching sector data for {ticker} ({description})")
                
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period=period)
                
                if not data.empty and self._validate_data(data):
                    sector_data[ticker] = data
                    self.logger.info(f"Successfully fetched {len(data)} records for {ticker}")
                else:
                    self.logger.warning(f"Failed to fetch data for {ticker}")
                
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        return sector_data
    
    def get_company_info(self) -> Optional[Dict]:
        """
        Get company information for Apple.
        
        Returns:
            Dict: Company information dictionary
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            info = ticker_obj.info
            
            if info:
                self.logger.info("Successfully fetched company information")
                return info
            else:
                self.logger.error("No company information returned")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching company info: {str(e)}")
            return None
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched data for basic quality checks.
        
        Args:
            data (pd.DataFrame): Data to validate
            
        Returns:
            bool: True if data passes validation, False otherwise
        """
        if data is None or data.empty:
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Expected: {required_columns}, Got: {list(data.columns)}")
            return False
        
        # Check for reasonable price values
        if (data['Close'] <= 0).any():
            self.logger.error("Found non-positive closing prices")
            return False
        
        # Check for reasonable volume values
        if (data['Volume'] < 0).any():
            self.logger.error("Found negative volume values")
            return False
        
        # Check for OHLC consistency
        if ((data['High'] < data['Low']) | 
            (data['High'] < data['Open']) | 
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) | 
            (data['Low'] > data['Close'])).any():
            self.logger.error("Found inconsistent OHLC data")
            return False
        
        return True
    
    def save_to_csv(self, data: pd.DataFrame, filename: str, directory: str = "data/raw") -> bool:
        """
        Save data to CSV file.
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Name of the file
            directory (str): Directory to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import os
            os.makedirs(directory, exist_ok=True)
            
            filepath = os.path.join(directory, filename)
            data.to_csv(filepath)
            
            self.logger.info(f"Data saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data to CSV: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    collector = AppleDataCollector()
    
    # Fetch daily data
    daily_data = collector.fetch_daily_data(period="1y")
    if daily_data is not None:
        collector.save_to_csv(daily_data, "aapl_daily_1y.csv")
    
    # Fetch company info
    company_info = collector.get_company_info()
    if company_info:
        print(f"Company: {company_info.get('longName', 'N/A')}")
        print(f"Sector: {company_info.get('sector', 'N/A')}")
        print(f"Market Cap: ${company_info.get('marketCap', 'N/A'):,}")
