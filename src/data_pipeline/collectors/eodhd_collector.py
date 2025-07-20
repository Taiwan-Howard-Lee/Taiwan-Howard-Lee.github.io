#!/usr/bin/env python3
"""
EODHD (End of Day Historical Data) Comprehensive Collector
Extracts maximum data from EODHD's extensive API suite
API Key: 687c985a3deee0.98552733
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

class EODHDCollector:
    """
    Comprehensive EODHD data collector
    Extracts maximum data from all available EODHD APIs
    """
    
    def __init__(self, api_key: str = "687c985a3deee0.98552733"):
        """Initialize the EODHD collector"""
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
        self.logger = self._setup_logger()
        
        # Request tracking
        self.requests_made = 0
        self.request_delay = 1  # 1 second between requests to be respectful
        
        # Data storage
        self.data_root = project_root / 'data' / 'eodhd'
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Available endpoints
        self.endpoints = {
            'eod': '/eod/{symbol}',                    # Historical EOD data
            'fundamentals': '/fundamentals/{symbol}',   # Fundamental data
            'news': '/news',                           # Financial news
            'sentiments': '/sentiments',               # Sentiment analysis
            'live': '/real-time/{symbol}',             # Live prices
            'intraday': '/intraday/{symbol}',          # Intraday data
            'exchanges': '/exchanges/{exchange}',       # Exchange data
            'bulk_fundamentals': '/bulk-fundamentals/{exchange}',  # Bulk fundamentals
            'calendar': '/calendar/earnings',          # Earnings calendar
            'dividends': '/div/{symbol}',              # Dividends
            'splits': '/splits/{symbol}',              # Stock splits
            'insider_transactions': '/insider-transactions',  # Insider trading
            'options': '/options/{symbol}',            # Options data
            'technical': '/technical/{symbol}',        # Technical indicators
            'macro_indicators': '/macro-indicator',    # Economic indicators
            'bonds': '/bond-fundamentals/{symbol}',    # Bond data
            'crypto': '/crypto',                       # Cryptocurrency data
            'forex': '/forex',                         # Forex data
            'etf_holdings': '/fundamentals/{symbol}',  # ETF holdings
            'mutual_fund_holdings': '/fundamentals/{symbol}',  # Mutual fund holdings
        }
        
        # Priority symbols for comprehensive collection
        self.priority_symbols = [
            # Major US Tech
            "AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "NVDA.US", "META.US", "TSLA.US",
            
            # Major US Financials
            "JPM.US", "BAC.US", "WFC.US", "GS.US", "MS.US",
            
            # Major US Healthcare
            "JNJ.US", "PFE.US", "UNH.US", "ABBV.US",
            
            # Major US Consumer
            "HD.US", "MCD.US", "NKE.US", "COST.US",
            
            # Major US Energy
            "XOM.US", "CVX.US",
            
            # Major ETFs
            "SPY.US", "QQQ.US", "IWM.US", "VTI.US",
            
            # Sector ETFs
            "XLK.US", "XLF.US", "XLV.US", "XLE.US", "XLI.US"
        ]
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - EODHD - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        if params is None:
            params = {}
        
        params['api_token'] = self.api_key
        params['fmt'] = 'json'
        
        # Rate limiting
        if self.requests_made > 0:
            time.sleep(self.request_delay)
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params)
            self.requests_made += 1
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API request failed: {response.status_code} - {url}")
                return None
                
        except Exception as e:
            self.logger.error(f"Request error: {str(e)}")
            return None
    
    def collect_eod_data(self, symbol: str, period: str = 'd', from_date: str = None, to_date: str = None) -> Optional[pd.DataFrame]:
        """
        Collect End-of-Day historical data
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL.US')
            period (str): 'd' for daily, 'w' for weekly, 'm' for monthly
            from_date (str): Start date (YYYY-MM-DD)
            to_date (str): End date (YYYY-MM-DD)
        """
        self.logger.info(f"📊 Collecting EOD data for {symbol}")
        
        params = {'period': period}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        data = self._make_request(f'/eod/{symbol}', params)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['symbol'] = symbol
                self.logger.info(f"✅ {symbol}: {len(df)} EOD records collected")
                return df
        
        return None
    
    def collect_fundamentals(self, symbol: str, filters: List[str] = None) -> Optional[Dict]:
        """
        Collect comprehensive fundamental data
        
        Args:
            symbol (str): Stock symbol
            filters (List[str]): Specific data sections to retrieve
        """
        self.logger.info(f"📈 Collecting fundamentals for {symbol}")
        
        params = {}
        if filters:
            params['filter'] = ','.join(filters)
        
        data = self._make_request(f'/fundamentals/{symbol}', params)
        
        if data:
            self.logger.info(f"✅ {symbol}: Fundamental data collected")
            return data
        
        return None
    
    def collect_news(self, symbol: str = None, tag: str = None, limit: int = 100, from_date: str = None, to_date: str = None) -> Optional[List[Dict]]:
        """
        Collect financial news
        
        Args:
            symbol (str): Stock symbol for news
            tag (str): Topic tag for news
            limit (int): Number of articles to retrieve
            from_date (str): Start date
            to_date (str): End date
        """
        self.logger.info(f"📰 Collecting news for {symbol or tag}")
        
        params = {'limit': limit}
        if symbol:
            params['s'] = symbol
        if tag:
            params['t'] = tag
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        data = self._make_request('/news', params)
        
        if data:
            self.logger.info(f"✅ {len(data)} news articles collected")
            return data
        
        return None
    
    def collect_sentiment(self, symbols: Union[str, List[str]], from_date: str = None, to_date: str = None) -> Optional[Dict]:
        """
        Collect sentiment analysis data
        
        Args:
            symbols: Single symbol or list of symbols
            from_date (str): Start date
            to_date (str): End date
        """
        if isinstance(symbols, list):
            symbols_str = ','.join(symbols)
        else:
            symbols_str = symbols
        
        self.logger.info(f"😊 Collecting sentiment for {symbols_str}")
        
        params = {'s': symbols_str}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        data = self._make_request('/sentiments', params)
        
        if data:
            self.logger.info(f"✅ Sentiment data collected for {len(data)} symbols")
            return data
        
        return None
    
    def collect_intraday(self, symbol: str, interval: str = '5m', from_date: str = None, to_date: str = None) -> Optional[pd.DataFrame]:
        """
        Collect intraday data
        
        Args:
            symbol (str): Stock symbol
            interval (str): Time interval ('1m', '5m', '1h')
            from_date (str): Start date
            to_date (str): End date
        """
        self.logger.info(f"⏰ Collecting intraday data for {symbol}")
        
        params = {'interval': interval}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        data = self._make_request(f'/intraday/{symbol}', params)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df['symbol'] = symbol
                self.logger.info(f"✅ {symbol}: {len(df)} intraday records collected")
                return df
        
        return None
    
    def collect_dividends_splits(self, symbol: str, from_date: str = None, to_date: str = None) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Collect dividends and splits data
        
        Args:
            symbol (str): Stock symbol
            from_date (str): Start date
            to_date (str): End date
        """
        self.logger.info(f"💰 Collecting dividends and splits for {symbol}")
        
        results = {}
        
        # Collect dividends
        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        div_data = self._make_request(f'/div/{symbol}', params)
        if div_data:
            df_div = pd.DataFrame(div_data)
            if not df_div.empty:
                df_div['date'] = pd.to_datetime(df_div['date'])
                df_div['symbol'] = symbol
                results['dividends'] = df_div
                self.logger.info(f"✅ {symbol}: {len(df_div)} dividend records")
        
        # Collect splits
        split_data = self._make_request(f'/splits/{symbol}', params)
        if split_data:
            df_split = pd.DataFrame(split_data)
            if not df_split.empty:
                df_split['date'] = pd.to_datetime(df_split['date'])
                df_split['symbol'] = symbol
                results['splits'] = df_split
                self.logger.info(f"✅ {symbol}: {len(df_split)} split records")
        
        return results
    
    def collect_technical_indicators(self, symbol: str, function: str = 'sma', period: int = 50) -> Optional[pd.DataFrame]:
        """
        Collect technical indicators
        
        Args:
            symbol (str): Stock symbol
            function (str): Technical indicator function
            period (int): Period for calculation
        """
        self.logger.info(f"📊 Collecting technical indicators for {symbol}")
        
        params = {
            'function': function,
            'period': period
        }
        
        data = self._make_request(f'/technical/{symbol}', params)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['symbol'] = symbol
                self.logger.info(f"✅ {symbol}: {len(df)} technical indicator records")
                return df
        
        return None
    
    def collect_comprehensive_data(self, symbol: str, days_back: int = 365) -> Dict[str, Any]:
        """
        Collect comprehensive data for a symbol from all available endpoints
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days of historical data
        """
        self.logger.info(f"🚀 Starting comprehensive data collection for {symbol}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        results = {
            'symbol': symbol,
            'collection_date': datetime.now().isoformat(),
            'date_range': {'from': from_date, 'to': to_date},
            'data': {}
        }
        
        # 1. EOD Historical Data
        eod_data = self.collect_eod_data(symbol, from_date=from_date, to_date=to_date)
        if eod_data is not None:
            results['data']['eod'] = eod_data
        
        # 2. Fundamental Data
        fundamentals = self.collect_fundamentals(symbol)
        if fundamentals:
            results['data']['fundamentals'] = fundamentals
        
        # 3. News Data
        news = self.collect_news(symbol=symbol, limit=50, from_date=from_date, to_date=to_date)
        if news:
            results['data']['news'] = news
        
        # 4. Sentiment Data
        sentiment = self.collect_sentiment(symbol, from_date=from_date, to_date=to_date)
        if sentiment:
            results['data']['sentiment'] = sentiment
        
        # 5. Dividends and Splits
        div_split = self.collect_dividends_splits(symbol, from_date=from_date, to_date=to_date)
        if div_split:
            results['data'].update(div_split)
        
        # 6. Technical Indicators (multiple)
        technical_functions = ['sma', 'ema', 'rsi', 'macd', 'bbands']
        for func in technical_functions:
            try:
                tech_data = self.collect_technical_indicators(symbol, function=func)
                if tech_data is not None:
                    results['data'][f'technical_{func}'] = tech_data
            except Exception as e:
                self.logger.warning(f"⚠️ Technical indicator {func} failed: {str(e)}")
        
        # 7. Intraday Data (last 7 days)
        recent_date = (end_date - timedelta(days=7)).strftime('%Y-%m-%d')
        intraday = self.collect_intraday(symbol, interval='1h', from_date=recent_date, to_date=to_date)
        if intraday is not None:
            results['data']['intraday'] = intraday

        # 8. Options Data (if available)
        try:
            options = self.collect_options_data(symbol, from_date=from_date, to_date=to_date)
            if options:
                results['data']['options'] = options
        except Exception as e:
            self.logger.warning(f"⚠️ Options data failed: {str(e)}")

        # 9. Insider Transactions
        try:
            insider = self.collect_insider_transactions(symbol, limit=50)
            if insider:
                results['data']['insider_transactions'] = insider
        except Exception as e:
            self.logger.warning(f"⚠️ Insider transactions failed: {str(e)}")

        # 10. Live Prices
        try:
            live_price = self.collect_live_prices(symbol)
            if live_price:
                results['data']['live_price'] = live_price
        except Exception as e:
            self.logger.warning(f"⚠️ Live price failed: {str(e)}")

        # 11. ETF Holdings (if ETF)
        if 'ETF' in symbol.upper() or any(etf in symbol.upper() for etf in ['SPY', 'QQQ', 'VTI', 'XL']):
            try:
                etf_holdings = self.collect_etf_holdings(symbol)
                if etf_holdings:
                    results['data']['etf_holdings'] = etf_holdings
            except Exception as e:
                self.logger.warning(f"⚠️ ETF holdings failed: {str(e)}")

        self.logger.info(f"🎉 Comprehensive collection completed for {symbol}")
        self.logger.info(f"   📊 Data types collected: {len(results['data'])}")

        return results
    
    def collect_exchange_data(self, exchange: str = 'US') -> Optional[List[Dict]]:
        """
        Collect exchange information and ticker list

        Args:
            exchange (str): Exchange code (e.g., 'US', 'NASDAQ', 'NYSE')
        """
        self.logger.info(f"🏛️ Collecting exchange data for {exchange}")

        data = self._make_request(f'/exchanges/{exchange}')

        if data:
            self.logger.info(f"✅ {exchange}: {len(data)} tickers available")
            return data

        return None

    def collect_earnings_calendar(self, from_date: str = None, to_date: str = None, symbols: str = None) -> Optional[List[Dict]]:
        """
        Collect earnings calendar data

        Args:
            from_date (str): Start date
            to_date (str): End date
            symbols (str): Comma-separated symbols
        """
        self.logger.info(f"📅 Collecting earnings calendar")

        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        if symbols:
            params['symbols'] = symbols

        data = self._make_request('/calendar/earnings', params)

        if data:
            self.logger.info(f"✅ {len(data)} earnings events collected")
            return data

        return None

    def collect_insider_transactions(self, symbol: str = None, limit: int = 100) -> Optional[List[Dict]]:
        """
        Collect insider transaction data

        Args:
            symbol (str): Stock symbol
            limit (int): Number of transactions to retrieve
        """
        self.logger.info(f"👥 Collecting insider transactions for {symbol or 'all'}")

        params = {'limit': limit}
        if symbol:
            params['code'] = symbol

        data = self._make_request('/insider-transactions', params)

        if data:
            self.logger.info(f"✅ {len(data)} insider transactions collected")
            return data

        return None

    def collect_options_data(self, symbol: str, from_date: str = None, to_date: str = None) -> Optional[Dict]:
        """
        Collect options data

        Args:
            symbol (str): Stock symbol
            from_date (str): Start date
            to_date (str): End date
        """
        self.logger.info(f"📊 Collecting options data for {symbol}")

        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        data = self._make_request(f'/options/{symbol}', params)

        if data:
            self.logger.info(f"✅ {symbol}: Options data collected")
            return data

        return None

    def collect_macro_indicators(self, country: str = 'USA', indicator: str = None) -> Optional[List[Dict]]:
        """
        Collect macroeconomic indicators

        Args:
            country (str): Country code
            indicator (str): Specific indicator
        """
        self.logger.info(f"🌍 Collecting macro indicators for {country}")

        params = {'country': country}
        if indicator:
            params['indicator'] = indicator

        data = self._make_request('/macro-indicator', params)

        if data:
            self.logger.info(f"✅ {len(data)} macro indicators collected")
            return data

        return None

    def collect_crypto_data(self, symbol: str = 'BTC-USD') -> Optional[Dict]:
        """
        Collect cryptocurrency data

        Args:
            symbol (str): Crypto symbol
        """
        self.logger.info(f"₿ Collecting crypto data for {symbol}")

        data = self._make_request(f'/real-time/{symbol}.CC')

        if data:
            self.logger.info(f"✅ {symbol}: Crypto data collected")
            return data

        return None

    def collect_forex_data(self, pair: str = 'EURUSD') -> Optional[Dict]:
        """
        Collect forex data

        Args:
            pair (str): Currency pair
        """
        self.logger.info(f"💱 Collecting forex data for {pair}")

        data = self._make_request(f'/real-time/{pair}.FOREX')

        if data:
            self.logger.info(f"✅ {pair}: Forex data collected")
            return data

        return None

    def collect_etf_holdings(self, symbol: str) -> Optional[Dict]:
        """
        Collect ETF holdings data

        Args:
            symbol (str): ETF symbol
        """
        self.logger.info(f"📊 Collecting ETF holdings for {symbol}")

        # ETF holdings are part of fundamentals data
        fundamentals = self.collect_fundamentals(symbol, filters=['ETF_Data'])

        if fundamentals and 'ETF_Data' in fundamentals:
            self.logger.info(f"✅ {symbol}: ETF holdings collected")
            return fundamentals['ETF_Data']

        return None

    def collect_mutual_fund_data(self, symbol: str) -> Optional[Dict]:
        """
        Collect mutual fund data

        Args:
            symbol (str): Mutual fund symbol
        """
        self.logger.info(f"🏦 Collecting mutual fund data for {symbol}")

        # Mutual fund data is part of fundamentals
        fundamentals = self.collect_fundamentals(symbol, filters=['MutualFund_Data'])

        if fundamentals and 'MutualFund_Data' in fundamentals:
            self.logger.info(f"✅ {symbol}: Mutual fund data collected")
            return fundamentals['MutualFund_Data']

        return None

    def collect_bond_data(self, symbol: str) -> Optional[Dict]:
        """
        Collect bond fundamental data

        Args:
            symbol (str): Bond symbol
        """
        self.logger.info(f"🏛️ Collecting bond data for {symbol}")

        data = self._make_request(f'/bond-fundamentals/{symbol}')

        if data:
            self.logger.info(f"✅ {symbol}: Bond data collected")
            return data

        return None

    def collect_bulk_fundamentals(self, exchange: str = 'US', limit: int = 100, offset: int = 0) -> Optional[List[Dict]]:
        """
        Collect bulk fundamental data for an exchange

        Args:
            exchange (str): Exchange code
            limit (int): Number of symbols to retrieve
            offset (int): Offset for pagination
        """
        self.logger.info(f"📊 Collecting bulk fundamentals for {exchange}")

        params = {
            'limit': limit,
            'offset': offset
        }

        data = self._make_request(f'/bulk-fundamentals/{exchange}', params)

        if data:
            self.logger.info(f"✅ {exchange}: {len(data)} bulk fundamentals collected")
            return data

        return None

    def collect_live_prices(self, symbols: Union[str, List[str]]) -> Optional[Dict]:
        """
        Collect live/real-time prices

        Args:
            symbols: Single symbol or list of symbols
        """
        if isinstance(symbols, list):
            symbols_str = ','.join(symbols)
        else:
            symbols_str = symbols

        self.logger.info(f"⚡ Collecting live prices for {symbols_str}")

        params = {'s': symbols_str}
        data = self._make_request('/real-time', params)

        if data:
            self.logger.info(f"✅ Live prices collected for {len(data)} symbols")
            return data

        return None

    def collect_batch_comprehensive(self, symbols: List[str] = None, max_symbols: int = 10) -> Dict[str, Any]:
        """
        Collect comprehensive data for multiple symbols
        
        Args:
            symbols (List[str]): List of symbols to collect
            max_symbols (int): Maximum number of symbols to process
        """
        symbols = symbols or self.priority_symbols[:max_symbols]
        
        self.logger.info(f"🚀 Starting batch comprehensive collection")
        self.logger.info(f"📊 Symbols: {symbols}")
        
        batch_results = {
            'collection_date': datetime.now().isoformat(),
            'symbols_processed': 0,
            'symbols_failed': 0,
            'total_requests': 0,
            'results': {}
        }
        
        for symbol in symbols:
            try:
                self.logger.info(f"📊 Processing {symbol}...")
                
                symbol_data = self.collect_comprehensive_data(symbol)
                
                if symbol_data['data']:
                    batch_results['results'][symbol] = symbol_data
                    batch_results['symbols_processed'] += 1
                    self.logger.info(f"✅ {symbol}: {len(symbol_data['data'])} data types collected")
                else:
                    batch_results['symbols_failed'] += 1
                    self.logger.warning(f"⚠️ {symbol}: No data collected")
                
            except Exception as e:
                batch_results['symbols_failed'] += 1
                self.logger.error(f"❌ {symbol}: Collection failed - {str(e)}")
        
        batch_results['total_requests'] = self.requests_made
        
        # Save batch results
        self._save_batch_results(batch_results)
        
        self.logger.info(f"🎉 Batch collection completed:")
        self.logger.info(f"   ✅ Processed: {batch_results['symbols_processed']}")
        self.logger.info(f"   ❌ Failed: {batch_results['symbols_failed']}")
        self.logger.info(f"   📞 Total requests: {batch_results['total_requests']}")
        
        return batch_results
    
    def _save_batch_results(self, results: Dict):
        """Save batch collection results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_file = self.data_root / f"eodhd_batch_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            # Create a summary without the large data objects
            summary = {
                'collection_date': results['collection_date'],
                'symbols_processed': results['symbols_processed'],
                'symbols_failed': results['symbols_failed'],
                'total_requests': results['total_requests'],
                'symbols': list(results['results'].keys())
            }
            json.dump(summary, f, indent=2)
        
        # Save individual symbol data
        for symbol, symbol_data in results['results'].items():
            symbol_file = self.data_root / f"{symbol.replace('.', '_')}_comprehensive_{timestamp}.json"
            with open(symbol_file, 'w') as f:
                # Convert DataFrames to JSON-serializable format
                serializable_data = {}
                for key, value in symbol_data['data'].items():
                    if isinstance(value, pd.DataFrame):
                        serializable_data[key] = value.to_dict('records')
                    else:
                        serializable_data[key] = value
                
                symbol_data_copy = symbol_data.copy()
                symbol_data_copy['data'] = serializable_data
                json.dump(symbol_data_copy, f, indent=2, default=str)
        
        self.logger.info(f"💾 Batch results saved to {self.data_root}")

def main():
    """Test the EODHD collector"""
    print("🧪 Testing EODHD Comprehensive Collector")
    print("=" * 45)
    
    collector = EODHDCollector()
    
    # Test comprehensive collection for a few symbols
    test_symbols = ['AAPL.US', 'MSFT.US', 'SPY.US']
    
    results = collector.collect_batch_comprehensive(symbols=test_symbols, max_symbols=3)
    
    print(f"\n📊 Collection Results:")
    print(f"   Symbols processed: {results['symbols_processed']}")
    print(f"   Symbols failed: {results['symbols_failed']}")
    print(f"   Total API requests: {results['total_requests']}")
    
    for symbol, data in results['results'].items():
        print(f"\n✅ {symbol}:")
        for data_type, content in data['data'].items():
            if isinstance(content, pd.DataFrame):
                print(f"     {data_type}: {len(content)} records")
            elif isinstance(content, list):
                print(f"     {data_type}: {len(content)} items")
            else:
                print(f"     {data_type}: Available")

if __name__ == "__main__":
    main()
