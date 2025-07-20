"""
Polygon.io Data Collector - High-quality US market data for Apple ML Trading
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import time
import json

class PolygonCollector:
    """
    Collector for US market data from Polygon.io API.
    Provides real-time and historical data for Apple stock analysis.
    """
    
    def __init__(self, api_key: str = "YpK43xQz3xo0hRVS2l6u8lqwJPSn_Tgf"):
        """
        Initialize Polygon.io collector.
        
        Args:
            api_key (str): Polygon.io API key
        """
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.logger = self._setup_logger()
        
        # Default headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the collector."""
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
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make API request with error handling.
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Query parameters
            
        Returns:
            Dict: API response data or None if error
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Request error: {str(e)}")
            return None
    
    def get_ticker_details(self, ticker: str = "AAPL") -> Optional[Dict]:
        """
        Get detailed information about a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Ticker details
        """
        self.logger.info(f"Fetching ticker details for {ticker}")
        
        data = self._make_request(f"v3/reference/tickers/{ticker}")
        
        if data and data.get('status') == 'OK':
            return data.get('results', {})
        
        return None
    
    def get_previous_close(self, ticker: str = "AAPL") -> Optional[Dict]:
        """
        Get previous trading day's data.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Previous close data (OHLCV)
        """
        self.logger.info(f"Fetching previous close for {ticker}")
        
        data = self._make_request(f"v2/aggs/ticker/{ticker}/prev")
        
        if data and data.get('status') == 'OK':
            results = data.get('results', [])
            if results:
                result = results[0]
                # Convert timestamp to readable date
                result['date'] = datetime.fromtimestamp(result.get('t', 0)/1000).strftime('%Y-%m-%d')
                return result
        
        return None
    
    def get_aggregates(self, ticker: str = "AAPL", days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical aggregates (OHLCV) data.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days to fetch
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        self.logger.info(f"Fetching {days} days of aggregates for {ticker}")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days+5)).strftime('%Y-%m-%d')
        
        endpoint = f"v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        data = self._make_request(endpoint)
        
        if data and data.get('status') == 'OK':
            results = data.get('results', [])
            
            if results:
                df = pd.DataFrame(results)
                
                # Convert timestamp to datetime
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('date', inplace=True)
                
                # Rename columns to standard format
                df.rename(columns={
                    'o': 'Open',
                    'h': 'High', 
                    'l': 'Low',
                    'c': 'Close',
                    'v': 'Volume'
                }, inplace=True)
                
                # Keep only OHLCV columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                self.logger.info(f"Retrieved {len(df)} days of data")
                return df.tail(days)  # Return only requested number of days
        
        return None
    
    def get_news(self, ticker: str = "AAPL", limit: int = 10) -> Optional[List[Dict]]:
        """
        Get recent news for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            limit (int): Number of articles to fetch
            
        Returns:
            List[Dict]: News articles
        """
        self.logger.info(f"Fetching {limit} news articles for {ticker}")
        
        params = {
            'ticker': ticker,
            'limit': limit,
            'order': 'desc'
        }
        
        data = self._make_request("v2/reference/news", params)
        
        if data and data.get('status') == 'OK':
            results = data.get('results', [])
            
            # Clean and format news data
            for article in results:
                # Convert timestamp to readable date
                if 'published_utc' in article:
                    article['published_date'] = article['published_utc'][:10]
                
                # Truncate title if too long
                if 'title' in article and len(article['title']) > 100:
                    article['title_short'] = article['title'][:97] + '...'
                else:
                    article['title_short'] = article.get('title', '')
            
            self.logger.info(f"Retrieved {len(results)} news articles")
            return results
        
        return None
    
    def get_dividends(self, ticker: str = "AAPL") -> Optional[List[Dict]]:
        """
        Get dividend information for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            List[Dict]: Dividend records
        """
        self.logger.info(f"Fetching dividends for {ticker}")
        
        params = {'ticker': ticker}
        data = self._make_request("v3/reference/dividends", params)
        
        if data and data.get('status') == 'OK':
            results = data.get('results', [])
            self.logger.info(f"Retrieved {len(results)} dividend records")
            return results
        
        return None
    
    def get_market_summary(self, ticker: str = "AAPL") -> Dict[str, any]:
        """
        Get comprehensive market summary for Apple analysis.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict: Market summary with all available data
        """
        self.logger.info(f"Generating market summary for {ticker}")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'ticker_details': {},
            'current_data': {},
            'historical_data': None,
            'news': [],
            'dividends': []
        }
        
        # Get ticker details
        ticker_details = self.get_ticker_details(ticker)
        if ticker_details:
            summary['ticker_details'] = {
                'name': ticker_details.get('name'),
                'market': ticker_details.get('market'),
                'type': ticker_details.get('type'),
                'currency': ticker_details.get('currency_name'),
                'active': ticker_details.get('active')
            }
        
        # Get current data
        current_data = self.get_previous_close(ticker)
        if current_data:
            summary['current_data'] = {
                'open': current_data.get('o'),
                'high': current_data.get('h'),
                'low': current_data.get('l'),
                'close': current_data.get('c'),
                'volume': current_data.get('v'),
                'date': current_data.get('date')
            }
        
        # Get historical data
        historical_data = self.get_aggregates(ticker, days=10)
        if historical_data is not None:
            summary['historical_data'] = historical_data.to_dict('records')
        
        # Get news
        news = self.get_news(ticker, limit=5)
        if news:
            summary['news'] = news[:5]  # Top 5 articles
        
        # Get dividends
        dividends = self.get_dividends(ticker)
        if dividends:
            summary['dividends'] = dividends[:3]  # Recent 3 dividends
        
        self.logger.info("Market summary generated successfully")
        return summary
    
    def save_market_data(self, ticker: str = "AAPL", filename: str = None) -> str:
        """
        Save current market data to JSON file.
        
        Args:
            ticker (str): Stock ticker symbol
            filename (str): Output filename
            
        Returns:
            str: Saved filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/external/polygon_{ticker.lower()}_data_{timestamp}.json"
        
        summary = self.get_market_summary(ticker)
        
        try:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Market data saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return None
