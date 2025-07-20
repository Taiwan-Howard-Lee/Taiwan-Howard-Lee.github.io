#!/usr/bin/env python3
"""
Enhanced Continuous Collector - Maximizes data diversity and freshness
Ensures new and diverse data collection with smart deduplication
"""

import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging
import os
import sys
import random
import hashlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_collection.polygon_collector import PolygonCollector
from data_collection.trading_economics_collector import TradingEconomicsCollector

class EnhancedContinuousCollector:
    """
    Enhanced continuous collector with maximum data diversity and freshness
    """
    
    def __init__(self):
        """Initialize enhanced collector with diversity tracking"""
        # File paths first
        self.data_dir = "data/continuous_collection"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = f"{self.data_dir}/enhanced_session_{self.session_id}.json"
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Initialize collectors
        self.polygon = PolygonCollector()
        self.trading_economics = TradingEconomicsCollector()
        
        # Rate limiting
        self.polygon_rate_limit = 5
        self.request_interval = 60 / self.polygon_rate_limit
        
        # Diversity tracking
        self.collected_hashes: Set[str] = set()
        self.request_history: List[Dict] = []
        self.last_request_times: Dict[str, datetime] = {}
        
        # Enhanced ticker lists
        self.core_tickers = ['AAPL']
        self.tech_tickers = ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL']
        self.market_etfs = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'GLD', 'TLT', 'HYG']
        self.sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE']
        
        # Time periods for historical data (more diverse)
        self.time_periods = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 252, 365]
        
        # Data collection tracking
        self.collected_data = {
            'ticker_details': {},
            'daily_data': [],
            'news_articles': [],
            'dividends': [],
            'economic_indicators': {},
            'currency_data': [],
            'collection_log': [],
            'diversity_metrics': {
                'unique_requests': 0,
                'duplicate_requests': 0,
                'fresh_data_ratio': 0.0,
                'ticker_coverage': 0,
                'time_period_coverage': 0
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up enhanced logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = f"{self.data_dir}/enhanced_collection.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _generate_request_hash(self, request: Dict) -> str:
        """Generate unique hash for request to track duplicates"""
        # Create a normalized string representation
        request_str = f"{request.get('type')}_{request.get('ticker', '')}_{request.get('days', '')}_{request.get('limit', '')}"
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def _is_request_fresh(self, request: Dict, min_interval_minutes: int = 30) -> bool:
        """Check if request is fresh (not made recently)"""
        request_key = f"{request.get('type')}_{request.get('ticker', '')}"
        
        if request_key in self.last_request_times:
            time_since_last = datetime.now() - self.last_request_times[request_key]
            return time_since_last.total_seconds() > (min_interval_minutes * 60)
        
        return True
    
    def build_enhanced_strategy(self, hours_to_run: int = 4) -> List[Dict]:
        """
        Build enhanced collection strategy with maximum diversity
        """
        total_minutes = hours_to_run * 60
        total_requests = int(total_minutes * self.polygon_rate_limit)
        
        self.logger.info(f"Building ENHANCED strategy for {hours_to_run} hours")
        self.logger.info(f"Total available requests: {total_requests}")
        
        strategy = []
        
        # 1. DIVERSE APPLE DATA (35% - reduced to make room for diversity)
        apple_requests = int(total_requests * 0.35)
        apple_strategies = []
        
        # Vary Apple data collection
        for i in range(apple_requests):
            if i % 6 == 0:
                apple_strategies.append({'type': 'ticker_details', 'ticker': 'AAPL'})
            elif i % 6 == 1:
                apple_strategies.append({'type': 'previous_close', 'ticker': 'AAPL'})
            elif i % 6 == 2:
                # Vary news limits for diversity
                limit = random.choice([5, 10, 15, 20])
                apple_strategies.append({'type': 'news', 'ticker': 'AAPL', 'limit': limit})
            elif i % 6 == 3:
                apple_strategies.append({'type': 'dividends', 'ticker': 'AAPL'})
            elif i % 6 == 4:
                # Random historical periods
                days = random.choice(self.time_periods)
                apple_strategies.append({'type': 'aggregates', 'ticker': 'AAPL', 'days': days})
            else:
                # Intraday data if available
                apple_strategies.append({'type': 'previous_close', 'ticker': 'AAPL'})
        
        strategy.extend(apple_strategies)
        
        # 2. TECH SECTOR DIVERSITY (25%)
        tech_requests = int(total_requests * 0.25)
        for i in range(tech_requests):
            ticker = random.choice(self.tech_tickers)
            request_type = random.choice(['ticker_details', 'previous_close', 'news'])
            
            if request_type == 'news':
                strategy.append({'type': request_type, 'ticker': ticker, 'limit': random.choice([5, 10])})
            elif request_type == 'aggregates':
                days = random.choice(self.time_periods[:8])  # Shorter periods for other tickers
                strategy.append({'type': request_type, 'ticker': ticker, 'days': days})
            else:
                strategy.append({'type': request_type, 'ticker': ticker})
        
        # 3. MARKET CONTEXT DIVERSITY (20%)
        market_requests = int(total_requests * 0.20)
        all_market_tickers = self.market_etfs + self.sector_etfs
        
        for i in range(market_requests):
            ticker = random.choice(all_market_tickers)
            request_type = random.choice(['ticker_details', 'previous_close'])
            strategy.append({'type': request_type, 'ticker': ticker})
        
        # 4. NEWS DIVERSITY (15%)
        news_requests = int(total_requests * 0.15)
        for i in range(news_requests):
            # Mix of general news and specific ticker news
            if i % 3 == 0:
                strategy.append({'type': 'general_news', 'limit': random.choice([10, 15, 20, 25])})
            else:
                ticker = random.choice(self.tech_tickers + ['AAPL'])
                strategy.append({'type': 'news', 'ticker': ticker, 'limit': random.choice([3, 5, 8])})
        
        # 5. TEMPORAL DIVERSITY (5%)
        temporal_requests = int(total_requests * 0.05)
        for i in range(temporal_requests):
            # Focus on different time periods for AAPL
            days = random.choice(self.time_periods[8:])  # Longer periods
            strategy.append({'type': 'aggregates', 'ticker': 'AAPL', 'days': days})
        
        # SMART SHUFFLING - Ensure even distribution
        random.shuffle(strategy)
        
        # Remove potential duplicates while maintaining diversity
        unique_strategy = []
        seen_hashes = set()
        
        for request in strategy:
            request_hash = self._generate_request_hash(request)
            if request_hash not in seen_hashes:
                unique_strategy.append(request)
                seen_hashes.add(request_hash)
        
        self.logger.info(f"Enhanced strategy built: {len(unique_strategy)} unique requests")
        self.logger.info(f"Diversity: {len(seen_hashes)} unique request types")
        
        return unique_strategy
    
    def execute_enhanced_request(self, request: Dict) -> Optional[Dict]:
        """Execute request with freshness and diversity tracking"""
        request_hash = self._generate_request_hash(request)
        request_key = f"{request.get('type')}_{request.get('ticker', '')}"
        
        # Track diversity metrics
        if request_hash in self.collected_hashes:
            self.collected_data['diversity_metrics']['duplicate_requests'] += 1
            self.logger.warning(f"Duplicate request detected: {request}")
        else:
            self.collected_data['diversity_metrics']['unique_requests'] += 1
            self.collected_hashes.add(request_hash)
        
        # Update last request time
        self.last_request_times[request_key] = datetime.now()
        
        # Execute the request (reuse existing logic)
        try:
            request_type = request.get('type')
            ticker = request.get('ticker', 'AAPL')
            
            if request_type == 'ticker_details':
                data = self.polygon.get_ticker_details(ticker)
                if data:
                    self.collected_data['ticker_details'][ticker] = data
                return data
                
            elif request_type == 'previous_close':
                data = self.polygon.get_previous_close(ticker)
                if data:
                    data['ticker'] = ticker
                    data['collection_timestamp'] = datetime.now().isoformat()
                    self.collected_data['daily_data'].append(data)
                return data
                
            elif request_type == 'aggregates':
                days = request.get('days', 30)
                data = self.polygon.get_aggregates(ticker, days)
                if data is not None:
                    records = data.reset_index().to_dict('records')
                    for record in records:
                        record['ticker'] = ticker
                        record['collection_type'] = f'aggregates_{days}d'
                        record['collection_timestamp'] = datetime.now().isoformat()
                    self.collected_data['daily_data'].extend(records)
                return len(data) if data is not None else 0
                
            elif request_type == 'news':
                limit = request.get('limit', 10)
                data = self.polygon.get_news(ticker, limit)
                if data:
                    for article in data:
                        article['ticker'] = ticker
                        article['collection_timestamp'] = datetime.now().isoformat()
                    self.collected_data['news_articles'].extend(data)
                return len(data) if data else 0
                
            elif request_type == 'dividends':
                data = self.polygon.get_dividends(ticker)
                if data:
                    for dividend in data:
                        dividend['ticker'] = ticker
                        dividend['collection_timestamp'] = datetime.now().isoformat()
                    self.collected_data['dividends'].extend(data)
                return len(data) if data else 0
                
            elif request_type == 'general_news':
                limit = request.get('limit', 20)
                data = self.polygon.get_news('', limit)
                if data:
                    for article in data:
                        article['collection_timestamp'] = datetime.now().isoformat()
                    self.collected_data['news_articles'].extend(data)
                return len(data) if data else 0
                
        except Exception as e:
            self.logger.error(f"Error executing enhanced request {request}: {str(e)}")
            return None
    
    def calculate_diversity_metrics(self):
        """Calculate and update diversity metrics"""
        metrics = self.collected_data['diversity_metrics']
        
        # Fresh data ratio
        total_requests = metrics['unique_requests'] + metrics['duplicate_requests']
        if total_requests > 0:
            metrics['fresh_data_ratio'] = metrics['unique_requests'] / total_requests
        
        # Ticker coverage
        all_tickers = set()
        for item in self.collected_data['daily_data']:
            if 'ticker' in item:
                all_tickers.add(item['ticker'])
        metrics['ticker_coverage'] = len(all_tickers)
        
        # Time period coverage
        time_periods = set()
        for item in self.collected_data['daily_data']:
            if 'collection_type' in item and 'aggregates_' in item['collection_type']:
                period = item['collection_type'].replace('aggregates_', '').replace('d', '')
                time_periods.add(period)
        metrics['time_period_coverage'] = len(time_periods)
        
        self.logger.info(f"Diversity Metrics: Fresh ratio: {metrics['fresh_data_ratio']:.2f}, "
                        f"Tickers: {metrics['ticker_coverage']}, Time periods: {metrics['time_period_coverage']}")

def main():
    """Test enhanced collector"""
    collector = EnhancedContinuousCollector()
    
    # Test for 5 minutes
    test_hours = 5/60
    
    print(f"🧪 Testing Enhanced Collector ({test_hours*60:.0f} minutes)")
    print(f"Expected requests: ~{int(test_hours * 60 * 5)}")
    
    # Build strategy
    strategy = collector.build_enhanced_strategy(test_hours)
    
    print(f"📊 Strategy diversity:")
    print(f"   Total requests: {len(strategy)}")
    
    # Show request type distribution
    request_types = {}
    tickers = set()
    for req in strategy:
        req_type = req['type']
        request_types[req_type] = request_types.get(req_type, 0) + 1
        if 'ticker' in req:
            tickers.add(req['ticker'])
    
    print(f"   Request types: {dict(request_types)}")
    print(f"   Unique tickers: {len(tickers)} - {sorted(list(tickers))[:10]}...")
    
    return strategy

if __name__ == "__main__":
    main()
