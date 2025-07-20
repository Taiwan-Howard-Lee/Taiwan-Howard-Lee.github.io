#!/usr/bin/env python3
"""
Multi-Source Data Integrator
Unified framework for integrating multiple data sources into the RL trading pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class MultiSourceDataIntegrator:
    """
    Unified data integration framework for multiple sources
    Handles collection, validation, normalization, and combination
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the multi-source integrator"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # Initialize collectors
        self.collectors = {}
        self._initialize_collectors()
        
        # Data storage
        self.data_root = project_root / 'data'
        self.integrated_data_path = self.data_root / 'features' / 'combined'
        self.integrated_data_path.mkdir(parents=True, exist_ok=True)
        
        # Integration state
        self.integration_results = {
            'timestamp': datetime.now().isoformat(),
            'sources_processed': {},
            'unified_dataset': None,
            'quality_metrics': {},
            'errors': []
        }
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load integration configuration"""
        if config_path is None:
            config_path = project_root / 'config' / 'data_sources' / 'registry.json'
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "data_sources": {
                "yahoo_finance": {"enabled": True, "priority": 1},
                "alpha_vantage": {"enabled": True, "priority": 2},
                "polygon": {"enabled": False, "priority": 3}
            },
            "integration_priority": ["yahoo_finance", "alpha_vantage", "polygon"],
            "data_validation": {
                "required_fields": ["symbol", "date", "close"],
                "quality_thresholds": {"min_records": 50, "max_missing_ratio": 0.05}
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - INTEGRATOR - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_collectors(self):
        """Initialize available data collectors"""
        try:
            # Yahoo Finance Collector
            if self.config['data_sources'].get('yahoo_finance', {}).get('enabled', False):
                from scripts.create_yahoo_collector import YahooFinanceCollector
                self.collectors['yahoo_finance'] = YahooFinanceCollector()
                self.logger.info("✅ Yahoo Finance collector initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Yahoo Finance collector failed: {str(e)}")
        
        try:
            # Alpha Vantage Collector
            if self.config['data_sources'].get('alpha_vantage', {}).get('enabled', False):
                from src.data_pipeline.collectors.alpha_vantage_collector import AlphaVantageCollector
                self.collectors['alpha_vantage'] = AlphaVantageCollector()
                self.logger.info("✅ Alpha Vantage collector initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Alpha Vantage collector failed: {str(e)}")
        
        try:
            # Polygon Collector
            if self.config['data_sources'].get('polygon', {}).get('enabled', False):
                from src.data_collection.polygon_collector import PolygonCollector
                self.collectors['polygon'] = PolygonCollector()
                self.logger.info("✅ Polygon collector initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Polygon collector failed: {str(e)}")
        
        self.logger.info(f"🔧 Initialized {len(self.collectors)} data collectors")
    
    def register_new_source(self, source_name: str, collector_class, config: Dict = None):
        """
        Register a new data source
        
        Args:
            source_name (str): Name of the data source
            collector_class: Collector class instance
            config (Dict): Source-specific configuration
        """
        try:
            self.collectors[source_name] = collector_class
            
            # Update configuration
            if config:
                self.config['data_sources'][source_name] = config
            
            self.logger.info(f"✅ Registered new data source: {source_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register {source_name}: {str(e)}")
    
    def collect_from_all_sources(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all enabled sources
        
        Args:
            symbols (List[str]): List of symbols to collect
            **kwargs: Additional parameters for collectors
            
        Returns:
            Dict: Data from each source
        """
        self.logger.info(f"🚀 Starting multi-source data collection")
        self.logger.info(f"📊 Symbols: {symbols}")
        self.logger.info(f"🔧 Sources: {list(self.collectors.keys())}")
        
        collected_data = {}
        
        # Collect from each source based on priority
        priority_sources = self.config.get('integration_priority', list(self.collectors.keys()))
        
        for source_name in priority_sources:
            if source_name not in self.collectors:
                continue
            
            try:
                self.logger.info(f"📊 Collecting from {source_name}...")
                collector = self.collectors[source_name]
                
                # Different collection methods for different sources
                if source_name == 'yahoo_finance':
                    # Collect all symbols at once
                    result = collector.collect_all_data(period="1y")
                    if result.get('symbols_processed', 0) > 0:
                        # Load the combined dataset
                        combined_data = collector.create_combined_dataset()
                        if not combined_data.empty:
                            collected_data[source_name] = combined_data
                
                elif source_name == 'alpha_vantage':
                    # Collect with rate limiting
                    result = collector.collect_batch_data(symbols=symbols[:10])  # Limit for API
                    if result.get('symbols_processed', 0) > 0:
                        combined_data = collector.create_combined_dataset()
                        if not combined_data.empty:
                            collected_data[source_name] = combined_data
                
                elif source_name == 'polygon':
                    # Collect individual symbols
                    polygon_data = []
                    for symbol in symbols[:5]:  # Limit for rate limits
                        symbol_data = collector.get_aggregates(symbol, days=100)
                        if symbol_data is not None:
                            symbol_data['symbol'] = symbol
                            polygon_data.append(symbol_data)
                    
                    if polygon_data:
                        combined_data = pd.concat(polygon_data, ignore_index=False)
                        collected_data[source_name] = combined_data
                
                self.integration_results['sources_processed'][source_name] = {
                    'status': 'success',
                    'records': len(collected_data.get(source_name, [])),
                    'symbols': collected_data.get(source_name, pd.DataFrame()).get('symbol', pd.Series()).nunique() if source_name in collected_data else 0
                }
                
                self.logger.info(f"✅ {source_name}: {len(collected_data.get(source_name, []))} records collected")
                
            except Exception as e:
                self.logger.error(f"❌ {source_name} collection failed: {str(e)}")
                self.integration_results['sources_processed'][source_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                self.integration_results['errors'].append({
                    'source': source_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        self.logger.info(f"🎉 Multi-source collection completed: {len(collected_data)} sources")
        return collected_data
    
    def normalize_and_integrate(self, source_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Normalize data from different sources and integrate into unified dataset
        
        Args:
            source_data (Dict): Data from different sources
            
        Returns:
            pd.DataFrame: Unified, normalized dataset
        """
        self.logger.info("🔧 Starting data normalization and integration")
        
        normalized_datasets = []
        
        for source_name, data in source_data.items():
            try:
                self.logger.info(f"📊 Normalizing {source_name} data...")
                
                # Normalize column names and data types
                normalized_data = self._normalize_source_data(data, source_name)
                
                # Add source identifier
                normalized_data['data_source'] = source_name
                
                # Validate data quality
                if self._validate_data_quality(normalized_data, source_name):
                    normalized_datasets.append(normalized_data)
                    self.logger.info(f"✅ {source_name}: {len(normalized_data)} records normalized")
                else:
                    self.logger.warning(f"⚠️ {source_name}: Data quality validation failed")
                
            except Exception as e:
                self.logger.error(f"❌ {source_name} normalization failed: {str(e)}")
                self.integration_results['errors'].append({
                    'source': source_name,
                    'stage': 'normalization',
                    'error': str(e)
                })
        
        if not normalized_datasets:
            self.logger.error("❌ No valid datasets to integrate")
            return pd.DataFrame()
        
        # Combine all normalized datasets
        self.logger.info("🔗 Combining normalized datasets...")
        
        # Strategy: Use the highest priority source as base, fill gaps with other sources
        priority_sources = self.config.get('integration_priority', [])
        
        # Sort datasets by priority
        sorted_datasets = []
        for source in priority_sources:
            for dataset in normalized_datasets:
                if dataset['data_source'].iloc[0] == source:
                    sorted_datasets.append(dataset)
                    break
        
        # Add any remaining datasets
        for dataset in normalized_datasets:
            source = dataset['data_source'].iloc[0]
            if source not in priority_sources:
                sorted_datasets.append(dataset)
        
        # Combine with priority-based merging
        unified_data = self._priority_merge(sorted_datasets)
        
        # Calculate quality metrics
        self.integration_results['quality_metrics'] = self._calculate_quality_metrics(unified_data)
        
        self.logger.info(f"🎉 Integration completed: {len(unified_data)} unified records")
        self.logger.info(f"📊 Symbols: {unified_data['symbol'].nunique()}")
        self.logger.info(f"📅 Date range: {unified_data.index.min().date()} to {unified_data.index.max().date()}")
        
        return unified_data
    
    def _normalize_source_data(self, data: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Normalize data from a specific source to unified schema"""
        normalized = data.copy()
        
        # Ensure required columns exist
        required_columns = ['symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Handle different column naming conventions
        column_mappings = {
            'yahoo_finance': {
                'Adj Close': 'Adjusted_Close'
            },
            'alpha_vantage': {
                'Adjusted_Close': 'Adjusted_Close'
            },
            'polygon': {
                'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'
            }
        }
        
        # Apply source-specific mappings
        if source_name in column_mappings:
            normalized = normalized.rename(columns=column_mappings[source_name])
        
        # Ensure data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
        
        # Ensure datetime index
        if not isinstance(normalized.index, pd.DatetimeIndex):
            if 'date' in normalized.columns:
                normalized['date'] = pd.to_datetime(normalized['date'])
                normalized.set_index('date', inplace=True)
        
        # Sort by date
        normalized = normalized.sort_index()
        
        return normalized
    
    def _validate_data_quality(self, data: pd.DataFrame, source_name: str) -> bool:
        """Validate data quality against thresholds"""
        validation_config = self.config.get('data_validation', {})
        
        # Check minimum records
        min_records = validation_config.get('min_records', 50)
        if len(data) < min_records:
            self.logger.warning(f"⚠️ {source_name}: Insufficient records ({len(data)} < {min_records})")
            return False
        
        # Check required fields
        required_fields = validation_config.get('required_fields', ['symbol', 'close'])
        missing_fields = [field for field in required_fields if field.lower() not in [col.lower() for col in data.columns]]
        if missing_fields:
            self.logger.warning(f"⚠️ {source_name}: Missing required fields: {missing_fields}")
            return False
        
        # Check missing data ratio
        max_missing_ratio = validation_config.get('max_missing_ratio', 0.05)
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > max_missing_ratio:
            self.logger.warning(f"⚠️ {source_name}: High missing data ratio: {missing_ratio:.2%}")
            return False
        
        return True
    
    def _priority_merge(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge datasets with priority-based conflict resolution"""
        if not datasets:
            return pd.DataFrame()
        
        if len(datasets) == 1:
            return datasets[0]
        
        # Start with highest priority dataset
        unified = datasets[0].copy()
        
        # Merge additional datasets, filling gaps
        for dataset in datasets[1:]:
            # Align on symbol and date
            for symbol in dataset['symbol'].unique():
                symbol_data = dataset[dataset['symbol'] == symbol]
                
                # Fill missing dates for this symbol
                existing_symbol_data = unified[unified['symbol'] == symbol]
                
                if existing_symbol_data.empty:
                    # Add completely new symbol
                    unified = pd.concat([unified, symbol_data], ignore_index=False)
                else:
                    # Fill gaps in existing symbol data
                    missing_dates = symbol_data.index.difference(existing_symbol_data.index)
                    if len(missing_dates) > 0:
                        gap_data = symbol_data.loc[missing_dates]
                        unified = pd.concat([unified, gap_data], ignore_index=False)
        
        # Sort and clean
        unified = unified.sort_index()
        unified = unified.drop_duplicates(subset=['symbol'], keep='first')
        
        return unified
    
    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        if data.empty:
            return {}
        
        return {
            'total_records': len(data),
            'unique_symbols': data['symbol'].nunique(),
            'date_range_days': (data.index.max() - data.index.min()).days,
            'missing_data_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
            'data_sources_used': data['data_source'].nunique() if 'data_source' in data.columns else 1,
            'completeness_score': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        }
    
    def save_integrated_dataset(self, data: pd.DataFrame, filename: str = None) -> str:
        """Save the integrated dataset"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integrated_dataset_{timestamp}.csv"
        
        filepath = self.integrated_data_path / filename
        data.to_csv(filepath)
        
        self.logger.info(f"💾 Integrated dataset saved: {filepath}")
        return str(filepath)
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get comprehensive integration summary"""
        return {
            'integration_results': self.integration_results,
            'available_collectors': list(self.collectors.keys()),
            'configuration': self.config,
            'quality_metrics': self.integration_results.get('quality_metrics', {})
        }

def main():
    """Test the multi-source integrator"""
    print("🧪 Testing Multi-Source Data Integrator")
    print("=" * 42)
    
    # Initialize integrator
    integrator = MultiSourceDataIntegrator()
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Collect from all sources
    source_data = integrator.collect_from_all_sources(test_symbols)
    
    print(f"\n📊 Collection Results:")
    for source, data in source_data.items():
        print(f"   {source}: {len(data)} records, {data['symbol'].nunique()} symbols")
    
    # Integrate data
    if source_data:
        unified_data = integrator.normalize_and_integrate(source_data)
        
        if not unified_data.empty:
            # Save integrated dataset
            filepath = integrator.save_integrated_dataset(unified_data)
            
            print(f"\n🎉 Integration Success:")
            print(f"   📊 Total records: {len(unified_data)}")
            print(f"   🏢 Symbols: {unified_data['symbol'].nunique()}")
            print(f"   📅 Date range: {unified_data.index.min().date()} to {unified_data.index.max().date()}")
            print(f"   💾 Saved to: {filepath}")
            
            # Show quality metrics
            summary = integrator.get_integration_summary()
            quality = summary['quality_metrics']
            print(f"\n📈 Quality Metrics:")
            print(f"   Completeness: {quality.get('completeness_score', 0):.2%}")
            print(f"   Missing data: {quality.get('missing_data_ratio', 0):.2%}")
            print(f"   Sources used: {quality.get('data_sources_used', 0)}")

if __name__ == "__main__":
    main()
