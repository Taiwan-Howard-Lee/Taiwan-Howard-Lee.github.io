#!/usr/bin/env python3
"""
Continuous Data Collector - Maximizes Polygon.io API usage at 5 requests/minute
Runs for hours collecting comprehensive Apple stock data
"""

import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_collection.polygon_collector import PolygonCollector
from data_collection.trading_economics_collector import TradingEconomicsCollector

class ContinuousCollector:
    """
    Intelligent continuous data collector that maximizes API usage
    Rate limit: 5 requests per minute for Polygon.io
    """
    
    def __init__(self):
        """Initialize the continuous collector"""
        # File paths first
        self.data_dir = "data/continuous_collection"
        os.makedirs(self.data_dir, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = f"{self.data_dir}/session_{self.session_id}.json"

        # Initialize logger after data_dir is set
        self.logger = self._setup_logger()

        # Initialize collectors
        self.polygon = PolygonCollector()
        self.trading_economics = TradingEconomicsCollector()

        # Rate limiting configuration
        self.polygon_rate_limit = 5  # requests per minute
        self.request_interval = 60 / self.polygon_rate_limit  # 12 seconds between requests

        # Data collection strategy
        self.collection_queue = []
        self.collected_data = {
            'ticker_details': {},
            'daily_data': [],
            'news_articles': [],
            'dividends': [],
            'economic_indicators': {},
            'currency_data': [],
            'collection_log': []
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for continuous collection"""
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
            log_file = f"{self.data_dir}/continuous_collection.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def build_collection_strategy(self, hours_to_run: int = 4) -> List[Dict]:
        """
        Build intelligent collection strategy to maximize data gathering
        
        Args:
            hours_to_run (int): Number of hours to run collection
            
        Returns:
            List[Dict]: Collection strategy queue
        """
        total_minutes = hours_to_run * 60
        total_requests = total_minutes * self.polygon_rate_limit
        
        self.logger.info(f"Building collection strategy for {hours_to_run} hours")
        self.logger.info(f"Total available requests: {total_requests}")
        
        strategy = []
        
        # Priority 1: Core Apple data (40% of requests)
        core_requests = int(total_requests * 0.4)
        for i in range(core_requests):
            if i % 4 == 0:
                strategy.append({'type': 'ticker_details', 'ticker': 'AAPL'})
            elif i % 4 == 1:
                strategy.append({'type': 'previous_close', 'ticker': 'AAPL'})
            elif i % 4 == 2:
                strategy.append({'type': 'news', 'ticker': 'AAPL', 'limit': 10})
            else:
                strategy.append({'type': 'dividends', 'ticker': 'AAPL'})
        
        # Priority 2: Historical aggregates (30% of requests)
        historical_requests = int(total_requests * 0.3)
        for i in range(historical_requests):
            # Vary the time periods to get comprehensive historical data
            days = [7, 14, 30, 60, 90][i % 5]
            strategy.append({'type': 'aggregates', 'ticker': 'AAPL', 'days': days})
        
        # Priority 3: Related tickers (20% of requests)
        related_tickers = ['SPY', 'QQQ', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        related_requests = int(total_requests * 0.2)
        for i in range(related_requests):
            ticker = related_tickers[i % len(related_tickers)]
            if i % 2 == 0:
                strategy.append({'type': 'previous_close', 'ticker': ticker})
            else:
                strategy.append({'type': 'ticker_details', 'ticker': ticker})
        
        # Priority 4: News from other sources (10% of requests)
        news_requests = int(total_requests * 0.1)
        for i in range(news_requests):
            strategy.append({'type': 'general_news', 'limit': 20})
        
        # Shuffle strategy to distribute requests evenly
        import random
        random.shuffle(strategy)
        
        self.logger.info(f"Strategy built: {len(strategy)} total requests")
        return strategy
    
    def execute_request(self, request: Dict) -> Optional[Dict]:
        """
        Execute a single API request based on strategy
        
        Args:
            request (Dict): Request configuration
            
        Returns:
            Dict: Response data or None
        """
        request_type = request.get('type')
        ticker = request.get('ticker', 'AAPL')
        
        try:
            if request_type == 'ticker_details':
                data = self.polygon.get_ticker_details(ticker)
                if data:
                    self.collected_data['ticker_details'][ticker] = data
                return data
                
            elif request_type == 'previous_close':
                data = self.polygon.get_previous_close(ticker)
                if data:
                    data['ticker'] = ticker
                    self.collected_data['daily_data'].append(data)
                return data
                
            elif request_type == 'aggregates':
                days = request.get('days', 30)
                data = self.polygon.get_aggregates(ticker, days)
                if data is not None:
                    # Convert DataFrame to records for storage
                    records = data.reset_index().to_dict('records')
                    for record in records:
                        record['ticker'] = ticker
                        record['collection_type'] = f'aggregates_{days}d'
                    self.collected_data['daily_data'].extend(records)
                return len(data) if data is not None else 0
                
            elif request_type == 'news':
                limit = request.get('limit', 10)
                data = self.polygon.get_news(ticker, limit)
                if data:
                    for article in data:
                        article['ticker'] = ticker
                    self.collected_data['news_articles'].extend(data)
                return len(data) if data else 0
                
            elif request_type == 'dividends':
                data = self.polygon.get_dividends(ticker)
                if data:
                    for dividend in data:
                        dividend['ticker'] = ticker
                    self.collected_data['dividends'].extend(data)
                return len(data) if data else 0
                
            elif request_type == 'general_news':
                limit = request.get('limit', 20)
                data = self.polygon.get_news('', limit)  # General market news
                if data:
                    self.collected_data['news_articles'].extend(data)
                return len(data) if data else 0
                
        except Exception as e:
            self.logger.error(f"Error executing request {request}: {str(e)}")
            return None
    
    def collect_trading_economics_data(self):
        """Collect Trading Economics data (no rate limit)"""
        try:
            self.logger.info("Collecting Trading Economics data...")
            
            # Get economic indicators
            econ_data = self.trading_economics.get_economic_indicators()
            if econ_data is not None:
                self.collected_data['economic_indicators'] = econ_data.to_dict('records')
            
            # Get currency data
            currency_data = self.trading_economics.get_currency_data()
            if currency_data is not None:
                self.collected_data['currency_data'] = currency_data.to_dict('records')
            
            self.logger.info("Trading Economics data collected successfully")
            
        except Exception as e:
            self.logger.error(f"Error collecting Trading Economics data: {str(e)}")
    
    def save_session_data(self):
        """Save collected data to file"""
        try:
            # Add metadata
            metadata = {
                'session_id': self.session_id,
                'collection_start': self.collection_start.isoformat(),
                'last_update': datetime.now().isoformat(),
                'total_requests_made': len(self.collected_data['collection_log']),
                'data_summary': {
                    'ticker_details': len(self.collected_data['ticker_details']),
                    'daily_data_points': len(self.collected_data['daily_data']),
                    'news_articles': len(self.collected_data['news_articles']),
                    'dividends': len(self.collected_data['dividends']),
                    'economic_indicators': len(self.collected_data['economic_indicators']),
                    'currency_pairs': len(self.collected_data['currency_data'])
                }
            }
            
            # Combine metadata with collected data
            session_data = {
                'metadata': metadata,
                'data': self.collected_data
            }
            
            # Save to JSON file
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            self.logger.info(f"Session data saved to {self.session_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving session data: {str(e)}")
    
    def run_continuous_collection(self, hours: int = 4):
        """
        Run continuous data collection for specified hours
        
        Args:
            hours (int): Number of hours to run
        """
        self.collection_start = datetime.now()
        end_time = self.collection_start + timedelta(hours=hours)
        
        self.logger.info(f"🚀 Starting continuous collection for {hours} hours")
        self.logger.info(f"Start time: {self.collection_start}")
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Rate limit: {self.polygon_rate_limit} requests/minute")
        self.logger.info(f"Request interval: {self.request_interval:.1f} seconds")
        
        # Build collection strategy
        strategy = self.build_collection_strategy(hours)
        
        # Collect Trading Economics data once (no rate limit)
        self.collect_trading_economics_data()
        
        # Execute strategy
        request_count = 0
        successful_requests = 0
        
        for i, request in enumerate(strategy):
            current_time = datetime.now()
            
            # Check if we should stop
            if current_time >= end_time:
                self.logger.info("Collection time limit reached")
                break
            
            # Execute request
            self.logger.info(f"Request {i+1}/{len(strategy)}: {request}")
            
            start_time = time.time()
            result = self.execute_request(request)
            request_duration = time.time() - start_time
            
            # Log request
            log_entry = {
                'timestamp': current_time.isoformat(),
                'request_number': i + 1,
                'request': request,
                'success': result is not None,
                'duration': request_duration,
                'result_size': len(result) if isinstance(result, (list, dict)) else (result if isinstance(result, int) else 1)
            }
            self.collected_data['collection_log'].append(log_entry)
            
            request_count += 1
            if result is not None:
                successful_requests += 1
                self.logger.info(f"✅ Success: {result}")
            else:
                self.logger.warning(f"❌ Failed")
            
            # Save data every 10 requests
            if request_count % 10 == 0:
                self.save_session_data()
                self.logger.info(f"📊 Progress: {request_count}/{len(strategy)} requests ({successful_requests} successful)")
            
            # Rate limiting - wait before next request
            time.sleep(self.request_interval)
        
        # Final save
        self.save_session_data()
        
        # Summary
        self.logger.info("🎉 Collection completed!")
        self.logger.info(f"📊 Total requests: {request_count}")
        self.logger.info(f"✅ Successful requests: {successful_requests}")
        self.logger.info(f"📁 Data saved to: {self.session_file}")
        
        return self.collected_data

def main():
    """Main function to run continuous collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Apple Stock Data Collection')
    parser.add_argument('--hours', type=int, default=4, help='Hours to run collection (default: 4)')
    parser.add_argument('--test', action='store_true', help='Run in test mode (5 minutes)')
    
    args = parser.parse_args()
    
    if args.test:
        hours = 5/60  # 5 minutes for testing
        print("🧪 Running in TEST MODE (5 minutes)")
    else:
        hours = args.hours
        print(f"🚀 Running continuous collection for {hours} hours")
    
    collector = ContinuousCollector()
    data = collector.run_continuous_collection(hours)
    
    print(f"\n📊 Collection Summary:")
    print(f"   Ticker details: {len(data['ticker_details'])}")
    print(f"   Daily data points: {len(data['daily_data'])}")
    print(f"   News articles: {len(data['news_articles'])}")
    print(f"   Dividends: {len(data['dividends'])}")
    print(f"   Economic indicators: {len(data['economic_indicators'])}")
    print(f"   Currency pairs: {len(data['currency_data'])}")

if __name__ == "__main__":
    main()
