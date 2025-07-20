#!/usr/bin/env python3
"""
Financial Modeling Prep Collector
Comprehensive data collection from Financial Modeling Prep API
API Key: 6Ohb7GA5XxISuo5jSTAd2tUyrAqnCxVt
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

class FinancialModelingPrepCollector:
    """
    Comprehensive Financial Modeling Prep data collector
    Collects fundamentals, financial statements, ratios, DCF valuations, and more
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the Financial Modeling Prep collector"""
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.logger = self._setup_logger()
        
        # Initialize multi-API key manager
        from src.data_pipeline.utils.multi_api_key_manager import MultiAPIKeyManager
        self.key_manager = MultiAPIKeyManager()
        
        # Use provided key or get from key manager
        self.api_key = api_key or self.key_manager.get_next_key('financial_modeling_prep')
        
        # Request tracking
        self.requests_made = 0
        self.request_delay = 0.5  # 0.5 seconds between requests (250/day limit)
        
        # Data storage
        self.data_root = project_root / 'data' / 'financial_modeling_prep'
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - FMP - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Union[Dict, List]]:
        """Make API request with rate limiting and error handling"""
        if params is None:
            params = {}
        
        # Get current API key
        current_key = self.api_key or self.key_manager.get_next_key('financial_modeling_prep')
        if not current_key:
            self.logger.error("❌ No available API keys for Financial Modeling Prep")
            return None
        
        params['apikey'] = current_key
        
        # Rate limiting
        if self.requests_made > 0:
            time.sleep(self.request_delay)
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params)
            self.requests_made += 1
            
            if response.status_code == 200:
                self.key_manager.report_key_success('financial_modeling_prep', current_key)
                return response.json()
            elif response.status_code == 429:
                # Rate limit exceeded
                self.key_manager.report_key_error('financial_modeling_prep', current_key, 'rate_limit')
                self.logger.warning(f"⚠️ Rate limit exceeded, trying next key...")
                
                # Try with next available key
                next_key = self.key_manager.get_next_key('financial_modeling_prep')
                if next_key and next_key != current_key:
                    params['apikey'] = next_key
                    time.sleep(2)  # Extra delay for rate limit
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        self.api_key = next_key
                        self.key_manager.report_key_success('financial_modeling_prep', next_key)
                        return response.json()
                
                self.logger.error(f"API request failed: {response.status_code} - {url}")
                return None
            else:
                self.logger.error(f"API request failed: {response.status_code} - {url}")
                return None
                
        except Exception as e:
            self.key_manager.report_key_error('financial_modeling_prep', current_key, 'exception')
            self.logger.error(f"Request error: {str(e)}")
            return None
    
    def collect_company_profile(self, symbol: str) -> Optional[List[Dict]]:
        """
        Collect company profile data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[Dict]: Company profile data
        """
        self.logger.info(f"🏢 Collecting company profile for {symbol}")
        
        data = self._make_request(f'/profile/{symbol}')
        
        if data and isinstance(data, list) and len(data) > 0:
            self.logger.info(f"✅ {symbol}: Company profile collected")
            return data
        
        return None
    
    def collect_financial_statements(self, symbol: str, statement_type: str = 'income-statement', period: str = 'annual', limit: int = 5) -> Optional[List[Dict]]:
        """
        Collect financial statements
        
        Args:
            symbol (str): Stock symbol
            statement_type (str): Type of statement (income-statement, balance-sheet-statement, cash-flow-statement)
            period (str): Period (annual, quarter)
            limit (int): Number of periods to retrieve
            
        Returns:
            List[Dict]: Financial statement data
        """
        self.logger.info(f"📊 Collecting {statement_type} for {symbol}")
        
        params = {'period': period, 'limit': limit}
        data = self._make_request(f'/{statement_type}/{symbol}', params)
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: {len(data)} {statement_type} periods collected")
            return data
        
        return None
    
    def collect_financial_ratios(self, symbol: str, period: str = 'annual', limit: int = 5) -> Optional[List[Dict]]:
        """
        Collect financial ratios
        
        Args:
            symbol (str): Stock symbol
            period (str): Period (annual, quarter)
            limit (int): Number of periods to retrieve
            
        Returns:
            List[Dict]: Financial ratios data
        """
        self.logger.info(f"📈 Collecting financial ratios for {symbol}")
        
        params = {'period': period, 'limit': limit}
        data = self._make_request(f'/ratios/{symbol}', params)
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: {len(data)} ratio periods collected")
            return data
        
        return None
    
    def collect_key_metrics(self, symbol: str, period: str = 'annual', limit: int = 5) -> Optional[List[Dict]]:
        """
        Collect key financial metrics
        
        Args:
            symbol (str): Stock symbol
            period (str): Period (annual, quarter)
            limit (int): Number of periods to retrieve
            
        Returns:
            List[Dict]: Key metrics data
        """
        self.logger.info(f"🔑 Collecting key metrics for {symbol}")
        
        params = {'period': period, 'limit': limit}
        data = self._make_request(f'/key-metrics/{symbol}', params)
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: {len(data)} key metrics periods collected")
            return data
        
        return None
    
    def collect_dcf_valuation(self, symbol: str) -> Optional[List[Dict]]:
        """
        Collect DCF valuation data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[Dict]: DCF valuation data
        """
        self.logger.info(f"💰 Collecting DCF valuation for {symbol}")
        
        data = self._make_request(f'/discounted-cash-flow/{symbol}')
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: DCF valuation collected")
            return data
        
        return None
    
    def collect_earnings_calendar(self, symbol: str = None, from_date: str = None, to_date: str = None) -> Optional[List[Dict]]:
        """
        Collect earnings calendar
        
        Args:
            symbol (str): Stock symbol (optional)
            from_date (str): Start date (YYYY-MM-DD)
            to_date (str): End date (YYYY-MM-DD)
            
        Returns:
            List[Dict]: Earnings calendar data
        """
        self.logger.info(f"📅 Collecting earnings calendar")
        
        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        if symbol:
            data = self._make_request(f'/historical/earning_calendar/{symbol}', params)
        else:
            data = self._make_request('/earning_calendar', params)
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {len(data)} earnings events collected")
            return data
        
        return None
    
    def collect_insider_trading(self, symbol: str, limit: int = 100) -> Optional[List[Dict]]:
        """
        Collect insider trading data
        
        Args:
            symbol (str): Stock symbol
            limit (int): Number of transactions to retrieve
            
        Returns:
            List[Dict]: Insider trading data
        """
        self.logger.info(f"👥 Collecting insider trading for {symbol}")
        
        params = {'limit': limit}
        data = self._make_request(f'/insider-trading', params)
        
        if data and isinstance(data, list):
            # Filter for the specific symbol
            symbol_data = [item for item in data if item.get('symbol') == symbol]
            self.logger.info(f"✅ {symbol}: {len(symbol_data)} insider transactions collected")
            return symbol_data
        
        return None
    
    def collect_institutional_holdings(self, symbol: str) -> Optional[List[Dict]]:
        """
        Collect institutional holdings data
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[Dict]: Institutional holdings data
        """
        self.logger.info(f"🏛️ Collecting institutional holdings for {symbol}")
        
        data = self._make_request(f'/institutional-holder/{symbol}')
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: {len(data)} institutional holders collected")
            return data
        
        return None
    
    def collect_analyst_estimates(self, symbol: str, period: str = 'annual') -> Optional[List[Dict]]:
        """
        Collect analyst estimates
        
        Args:
            symbol (str): Stock symbol
            period (str): Period (annual, quarter)
            
        Returns:
            List[Dict]: Analyst estimates data
        """
        self.logger.info(f"📊 Collecting analyst estimates for {symbol}")
        
        params = {'period': period}
        data = self._make_request(f'/analyst-estimates/{symbol}', params)
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: {len(data)} analyst estimate periods collected")
            return data
        
        return None
    
    def collect_market_cap_history(self, symbol: str, limit: int = 100) -> Optional[List[Dict]]:
        """
        Collect market capitalization history
        
        Args:
            symbol (str): Stock symbol
            limit (int): Number of data points
            
        Returns:
            List[Dict]: Market cap history data
        """
        self.logger.info(f"📈 Collecting market cap history for {symbol}")
        
        params = {'limit': limit}
        data = self._make_request(f'/historical-market-capitalization/{symbol}', params)
        
        if data and isinstance(data, list):
            self.logger.info(f"✅ {symbol}: {len(data)} market cap data points collected")
            return data
        
        return None
    
    def collect_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """
        Collect comprehensive data from all Financial Modeling Prep endpoints
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Comprehensive data collection
        """
        self.logger.info(f"🚀 Starting comprehensive FMP collection for {symbol}")
        
        results = {
            'symbol': symbol,
            'collection_date': datetime.now().isoformat(),
            'data': {}
        }
        
        # 1. Company Profile
        profile = self.collect_company_profile(symbol)
        if profile:
            results['data']['profile'] = profile
        
        # 2. Financial Statements
        income_statement = self.collect_financial_statements(symbol, 'income-statement', limit=3)
        if income_statement:
            results['data']['income_statement'] = income_statement
        
        balance_sheet = self.collect_financial_statements(symbol, 'balance-sheet-statement', limit=3)
        if balance_sheet:
            results['data']['balance_sheet'] = balance_sheet
        
        cash_flow = self.collect_financial_statements(symbol, 'cash-flow-statement', limit=3)
        if cash_flow:
            results['data']['cash_flow'] = cash_flow
        
        # 3. Financial Ratios
        ratios = self.collect_financial_ratios(symbol, limit=3)
        if ratios:
            results['data']['ratios'] = ratios
        
        # 4. Key Metrics
        key_metrics = self.collect_key_metrics(symbol, limit=3)
        if key_metrics:
            results['data']['key_metrics'] = key_metrics
        
        # 5. DCF Valuation
        dcf = self.collect_dcf_valuation(symbol)
        if dcf:
            results['data']['dcf_valuation'] = dcf
        
        # 6. Institutional Holdings
        institutional = self.collect_institutional_holdings(symbol)
        if institutional:
            results['data']['institutional_holdings'] = institutional
        
        # 7. Analyst Estimates
        estimates = self.collect_analyst_estimates(symbol)
        if estimates:
            results['data']['analyst_estimates'] = estimates
        
        self.logger.info(f"🎉 FMP comprehensive collection completed for {symbol}")
        self.logger.info(f"   📊 Data types collected: {len(results['data'])}")
        
        return results

def main():
    """Test the Financial Modeling Prep collector"""
    print("🧪 Testing Financial Modeling Prep Collector")
    print("=" * 45)
    
    collector = FinancialModelingPrepCollector()
    
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
