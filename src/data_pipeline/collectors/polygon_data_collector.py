#!/usr/bin/env python3
"""
Polygon Data Collector - Pipeline-integrated version
Collects data from Polygon.io API and saves to organized data structure
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data_collection.polygon_collector import PolygonCollector

class PolygonDataCollector:
    """
    Pipeline-integrated Polygon.io data collector
    Handles data collection, organization, and storage
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Polygon data collector
        
        Args:
            config (Dict): Configuration parameters
        """
        self.config = config or {}
        self.collector = PolygonCollector()
        self.logger = self._setup_logging()
        
        # Data paths
        self.data_root = project_root / 'data'
        self.raw_data_path = self.data_root / 'raw' / 'polygon'
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Collection parameters
        self.default_ticker = 'AAPL'
        self.batch_size = self.config.get('batch_size', 50)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the collector"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - POLYGON - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def collect_batch(self, ticker: str = None) -> Dict[str, Any]:
        """
        Collect a batch of data from Polygon.io
        
        Args:
            ticker (str): Stock ticker to collect data for
            
        Returns:
            Dict: Collection results with data and metadata
        """
        ticker = ticker or self.default_ticker
        collection_timestamp = datetime.now()
        
        self.logger.info(f"📊 Starting Polygon data collection for {ticker}")
        
        batch_data = {
            'metadata': {
                'ticker': ticker,
                'collection_timestamp': collection_timestamp.isoformat(),
                'source': 'polygon.io',
                'collector_version': '2.0.0'
            },
            'data': {
                'ticker_details': None,
                'current_data': None,
                'historical_data': None,
                'news': None,
                'dividends': None
            },
            'metrics': {
                'total_records': 0,
                'api_calls_made': 0,
                'success_rate': 0.0,
                'collection_duration': 0.0
            }
        }
        
        start_time = datetime.now()
        successful_calls = 0
        total_calls = 0
        
        try:
            # 1. Collect ticker details
            self.logger.info(f"🔍 Collecting ticker details for {ticker}")
            ticker_details = self.collector.get_ticker_details(ticker)
            if ticker_details:
                batch_data['data']['ticker_details'] = ticker_details
                successful_calls += 1
                self.logger.info(f"✅ Ticker details collected")
            total_calls += 1
            
            # 2. Collect current data
            self.logger.info(f"📈 Collecting current data for {ticker}")
            current_data = self.collector.get_previous_close(ticker)
            if current_data:
                batch_data['data']['current_data'] = current_data
                successful_calls += 1
                self.logger.info(f"✅ Current data collected: ${current_data.get('c', 'N/A')}")
            total_calls += 1
            
            # 3. Collect historical data
            self.logger.info(f"📊 Collecting historical data for {ticker}")
            historical_data = self.collector.get_aggregates(ticker, days=30)
            if historical_data is not None:
                # Convert DataFrame to records for JSON serialization
                historical_records = historical_data.reset_index().to_dict('records')
                batch_data['data']['historical_data'] = historical_records
                batch_data['metrics']['total_records'] += len(historical_records)
                successful_calls += 1
                self.logger.info(f"✅ Historical data collected: {len(historical_records)} records")
            total_calls += 1
            
            # 4. Collect news
            self.logger.info(f"📰 Collecting news for {ticker}")
            news_data = self.collector.get_news(ticker, limit=10)
            if news_data:
                batch_data['data']['news'] = news_data
                batch_data['metrics']['total_records'] += len(news_data)
                successful_calls += 1
                self.logger.info(f"✅ News collected: {len(news_data)} articles")
            total_calls += 1
            
            # 5. Collect dividends
            self.logger.info(f"💰 Collecting dividends for {ticker}")
            dividend_data = self.collector.get_dividends(ticker)
            if dividend_data:
                batch_data['data']['dividends'] = dividend_data
                batch_data['metrics']['total_records'] += len(dividend_data)
                successful_calls += 1
                self.logger.info(f"✅ Dividends collected: {len(dividend_data)} records")
            total_calls += 1
            
            # Calculate metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            batch_data['metrics'].update({
                'api_calls_made': total_calls,
                'success_rate': successful_calls / total_calls if total_calls > 0 else 0.0,
                'collection_duration': duration
            })
            
            # Save batch data
            self._save_batch_data(batch_data)
            
            self.logger.info(f"🎉 Polygon collection completed:")
            self.logger.info(f"   📊 Total records: {batch_data['metrics']['total_records']}")
            self.logger.info(f"   📞 API calls: {total_calls} ({successful_calls} successful)")
            self.logger.info(f"   ⏱️ Duration: {duration:.2f} seconds")
            self.logger.info(f"   ✅ Success rate: {batch_data['metrics']['success_rate']:.2%}")
            
            return batch_data
            
        except Exception as e:
            self.logger.error(f"❌ Polygon collection failed: {str(e)}")
            batch_data['metadata']['error'] = str(e)
            batch_data['metadata']['status'] = 'failed'
            return batch_data
    
    def _save_batch_data(self, batch_data: Dict[str, Any]):
        """
        Save batch data to organized file structure
        
        Args:
            batch_data (Dict): Batch data to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = batch_data['metadata']['ticker']
        
        # Save complete batch data
        batch_file = self.raw_data_path / f"{ticker}_batch_{timestamp}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Batch data saved to: {batch_file}")
        
        # Save individual data types for easy access
        data = batch_data['data']
        
        # Save historical data as CSV if available
        if data['historical_data']:
            historical_df = pd.DataFrame(data['historical_data'])
            if 'date' in historical_df.columns:
                historical_df['date'] = pd.to_datetime(historical_df['date'])
                historical_df.set_index('date', inplace=True)
            
            csv_file = self.raw_data_path / f"{ticker}_historical_{timestamp}.csv"
            historical_df.to_csv(csv_file)
            self.logger.info(f"📊 Historical data saved to: {csv_file}")
        
        # Save news data as JSON if available
        if data['news']:
            news_file = self.raw_data_path / f"{ticker}_news_{timestamp}.json"
            with open(news_file, 'w') as f:
                json.dump(data['news'], f, indent=2, default=str)
            self.logger.info(f"📰 News data saved to: {news_file}")
    
    def get_latest_data(self, ticker: str = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest collected data for a ticker
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            Dict: Latest batch data or None
        """
        ticker = ticker or self.default_ticker
        
        # Find latest batch file
        batch_files = list(self.raw_data_path.glob(f"{ticker}_batch_*.json"))
        
        if not batch_files:
            return None
        
        latest_file = max(batch_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected data
        
        Returns:
            Dict: Data collection summary
        """
        summary = {
            'total_files': 0,
            'tickers': set(),
            'date_range': {'earliest': None, 'latest': None},
            'data_types': {
                'batch_files': 0,
                'historical_files': 0,
                'news_files': 0
            }
        }
        
        # Scan all files
        for file_path in self.raw_data_path.iterdir():
            if file_path.is_file():
                summary['total_files'] += 1
                
                # Extract ticker from filename
                filename = file_path.name
                if '_' in filename:
                    ticker = filename.split('_')[0]
                    summary['tickers'].add(ticker)
                
                # Categorize file types
                if 'batch' in filename:
                    summary['data_types']['batch_files'] += 1
                elif 'historical' in filename:
                    summary['data_types']['historical_files'] += 1
                elif 'news' in filename:
                    summary['data_types']['news_files'] += 1
                
                # Track date range
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if summary['date_range']['earliest'] is None or file_time < summary['date_range']['earliest']:
                    summary['date_range']['earliest'] = file_time
                if summary['date_range']['latest'] is None or file_time > summary['date_range']['latest']:
                    summary['date_range']['latest'] = file_time
        
        summary['tickers'] = list(summary['tickers'])
        
        return summary

def main():
    """Test the Polygon data collector"""
    collector = PolygonDataCollector()
    
    print("🧪 Testing Polygon Data Collector")
    print("=" * 40)
    
    # Collect batch data
    result = collector.collect_batch('AAPL')
    
    print(f"\n📊 Collection Results:")
    print(f"   Status: {'Success' if 'error' not in result['metadata'] else 'Failed'}")
    print(f"   Records: {result['metrics']['total_records']}")
    print(f"   API calls: {result['metrics']['api_calls_made']}")
    print(f"   Success rate: {result['metrics']['success_rate']:.2%}")
    print(f"   Duration: {result['metrics']['collection_duration']:.2f}s")
    
    # Get data summary
    summary = collector.get_data_summary()
    print(f"\n📁 Data Summary:")
    print(f"   Total files: {summary['total_files']}")
    print(f"   Tickers: {summary['tickers']}")
    print(f"   Batch files: {summary['data_types']['batch_files']}")

if __name__ == "__main__":
    main()
