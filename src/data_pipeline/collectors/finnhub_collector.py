#!/usr/bin/env python3
"""
Finnhub Collector
Comprehensive data collection from Finnhub API
API Key: d1u9j09r01qp7ee2e240d1u9j09r01qp7ee2e24g
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import json
import time
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class FinnhubCollector:
    """
    Comprehensive Finnhub data collector
    Collects real-time quotes, fundamentals, news, earnings, and more
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the Finnhub collector"""
        self.base_url = "https://finnhub.io/api/v1"
        self.logger = self._setup_logger()
        
        # Initialize multi-API key manager
        from src.data_pipeline.utils.multi_api_key_manager import MultiAPIKeyManager
        self.key_manager = MultiAPIKeyManager()
        
        # Use provided key or get from key manager
        self.api_key = api_key or self.key_manager.get_next_key('finnhub')
        
        # Request tracking
        self.requests_made = 0
        self.request_delay = 1  # 1 second between requests (60/minute limit)
        
        # Data storage
        self.data_root = project_root / 'data' / 'finnhub'
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - FINNHUB - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        if params is None:
            params = {}
        
        # Get current API key
        current_key = self.api_key or self.key_manager.get_next_key('finnhub')
        if not current_key:
            self.logger.error("❌ No available API keys for Finnhub")
            return None
        
        params['token'] = current_key
        
        # Rate limiting
        if self.requests_made > 0:
            time.sleep(self.request_delay)
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params)
            self.requests_made += 1
            
            if response.status_code == 200:
                self.key_manager.report_key_success('finnhub', current_key)
                return response.json()
            elif response.status_code == 429:
                # Rate limit exceeded
                self.key_manager.report_key_error('finnhub', current_key, 'rate_limit')
                self.logger.warning(f"⚠️ Rate limit exceeded, trying next key...")
                
                # Try with next available key
                next_key = self.key_manager.get_next_key('finnhub')
                if next_key and next_key != current_key:
                    params['token'] = next_key
                    time.sleep(2)  # Extra delay for rate limit
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        self.api_key = next_key
                        self.key_manager.report_key_success('finnhub', next_key)
                        return response.json()
                
                self.logger.error(f"API request failed: {response.status_code} - {url}")
                return None
            else:
                self.logger.error(f"API request failed: {response.status_code} - {url}")
                return None
                
        except Exception as e:
            self.key_manager.report_key_error('finnhub', current_key, 'exception')
            self.logger.error(f"Request error: {str(e)}")
            return None
    
    def collect_quote(self, symbol: str) -> Optional[Dict]:
        """
        Collect real-time quote data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Real-time quote data
        """
        self.logger.info(f"📊 Collecting real-time quote for {symbol}")
        
        data = self._make_request('/quote', {'symbol': symbol})
        
        if data and 'c' in data:  # 'c' is current price
            self.logger.info(f"✅ {symbol}: Real-time quote collected")
            return data
        
        return None
    
    def collect_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        Collect company profile data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Company profile data
        """
        self.logger.info(f"🏢 Collecting company profile for {symbol}")
        
        data = self._make_request('/stock/profile2', {'symbol': symbol})
        
        if data and 'name' in data:
            self.logger.info(f"✅ {symbol}: Company profile collected")
            return data
        
        return None
    
    def collect_basic_financials(self, symbol: str) -> Optional[Dict]:
        """
        Collect basic financial metrics
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Basic financial data
        """
        self.logger.info(f"📈 Collecting basic financials for {symbol}")
        
        data = self._make_request('/stock/metric', {'symbol': symbol, 'metric': 'all'})
        
        if data and 'metric' in data:
            self.logger.info(f"✅ {symbol}: Basic financials collected")
            return data
        
        return None
    
    def collect_earnings(self, symbol: str) -> Optional[List[Dict]]:
        """
        Collect earnings data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[Dict]: Earnings data
        """
        self.logger.info(f"💰 Collecting earnings for {symbol}")
        
        data = self._make_request('/stock/earnings', {'symbol': symbol})
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: {len(data)} earnings records collected")
            return data
        
        return None
    
    def collect_news(self, symbol: str = None, category: str = 'general', limit: int = 50) -> Optional[List[Dict]]:
        """
        Collect news data
        
        Args:
            symbol (str): Stock symbol for company news
            category (str): News category
            limit (int): Number of articles
            
        Returns:
            List[Dict]: News articles
        """
        if symbol:
            self.logger.info(f"📰 Collecting company news for {symbol}")
            
            # Get current date range (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            data = self._make_request('/company-news', params)
        else:
            self.logger.info(f"📰 Collecting general news - {category}")
            
            params = {'category': category}
            data = self._make_request('/news', params)
        
        if data and isinstance(data, list):
            # Limit results
            limited_data = data[:limit] if len(data) > limit else data
            self.logger.info(f"✅ {len(limited_data)} news articles collected")
            return limited_data
        
        return None
    
    def collect_insider_transactions(self, symbol: str) -> Optional[List[Dict]]:
        """
        Collect insider transaction data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[Dict]: Insider transactions
        """
        self.logger.info(f"👥 Collecting insider transactions for {symbol}")
        
        data = self._make_request('/stock/insider-transactions', {'symbol': symbol})
        
        if data and 'data' in data:
            transactions = data['data']
            self.logger.info(f"✅ {symbol}: {len(transactions)} insider transactions collected")
            return transactions
        
        return None
    
    def collect_social_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Collect social sentiment data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Social sentiment data
        """
        self.logger.info(f"😊 Collecting social sentiment for {symbol}")
        
        data = self._make_request('/stock/social-sentiment', {'symbol': symbol})
        
        if data and 'data' in data:
            self.logger.info(f"✅ {symbol}: Social sentiment collected")
            return data
        
        return None
    
    def collect_recommendation_trends(self, symbol: str) -> Optional[List[Dict]]:
        """
        Collect analyst recommendation trends
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[Dict]: Recommendation trends
        """
        self.logger.info(f"📊 Collecting recommendation trends for {symbol}")
        
        data = self._make_request('/stock/recommendation', {'symbol': symbol})
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: {len(data)} recommendation periods collected")
            return data
        
        return None
    
    def collect_technical_indicators(self, symbol: str, indicator: str = 'sma', resolution: str = 'D', timeperiod: int = 20) -> Optional[Dict]:
        """
        Collect technical indicator data
        
        Args:
            symbol (str): Stock symbol
            indicator (str): Technical indicator (sma, ema, rsi, etc.)
            resolution (str): Time resolution
            timeperiod (int): Period for calculation
            
        Returns:
            Dict: Technical indicator data
        """
        self.logger.info(f"📊 Collecting {indicator} for {symbol}")
        
        # Calculate date range (last 100 days)
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=100)).timestamp())
        
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'from': start_time,
            'to': end_time,
            'indicator': indicator,
            'timeperiod': timeperiod
        }
        
        data = self._make_request('/indicator', params)
        
        if data and 's' in data and data['s'] == 'ok':
            self.logger.info(f"✅ {symbol}: {indicator} data collected")
            return data
        
        return None
    
    def collect_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """
        Collect comprehensive data from all Finnhub endpoints
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Comprehensive data collection
        """
        self.logger.info(f"🚀 Starting comprehensive Finnhub collection for {symbol}")
        
        results = {
            'symbol': symbol,
            'collection_date': datetime.now().isoformat(),
            'data': {}
        }
        
        # 1. Real-time Quote
        quote = self.collect_quote(symbol)
        if quote:
            results['data']['quote'] = quote
        
        # 2. Company Profile
        profile = self.collect_company_profile(symbol)
        if profile:
            results['data']['profile'] = profile
        
        # 3. Basic Financials
        financials = self.collect_basic_financials(symbol)
        if financials:
            results['data']['financials'] = financials
        
        # 4. Earnings
        earnings = self.collect_earnings(symbol)
        if earnings:
            results['data']['earnings'] = earnings
        
        # 5. Company News
        news = self.collect_news(symbol, limit=20)
        if news:
            results['data']['news'] = news
        
        # 6. Insider Transactions
        insider = self.collect_insider_transactions(symbol)
        if insider:
            results['data']['insider_transactions'] = insider
        
        # 7. Social Sentiment
        sentiment = self.collect_social_sentiment(symbol)
        if sentiment:
            results['data']['social_sentiment'] = sentiment
        
        # 8. Recommendation Trends
        recommendations = self.collect_recommendation_trends(symbol)
        if recommendations:
            results['data']['recommendations'] = recommendations
        
        self.logger.info(f"🎉 Finnhub comprehensive collection completed for {symbol}")
        self.logger.info(f"   📊 Data types collected: {len(results['data'])}")
        
        return results

def main():
    """Test the Finnhub collector"""
    print("🧪 Testing Finnhub Collector")
    print("=" * 30)
    
    collector = FinnhubCollector()
    
    # Test comprehensive collection
    test_symbol = 'AAPL'
    
    results = collector.collect_comprehensive_data(test_symbol)
    
    print(f"\n📊 Collection Results for {test_symbol}:")
    for data_type, content in results['data'].items():
        if isinstance(content, list):
            print(f"   {data_type}: {len(content)} items")
        elif isinstance(content, dict):
            print(f"   {data_type}: Available")
        else:
            print(f"   {data_type}: {content}")
    
    print(f"\n📞 Total API requests made: {collector.requests_made}")

if __name__ == "__main__":
    main()
