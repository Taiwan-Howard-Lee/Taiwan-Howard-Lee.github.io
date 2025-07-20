#!/usr/bin/env python3
"""
Data Processor - Cleans and transforms raw data
Processes validated data into clean, analysis-ready format
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class DataProcessor:
    """
    Comprehensive data processor for the Apple ML Trading pipeline
    Cleans, transforms, and prepares data for analysis
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the data processor
        
        Args:
            config (Dict): Processing configuration
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Data paths
        self.data_root = project_root / 'data'
        self.raw_data_path = self.data_root / 'raw'
        self.processed_data_path = self.data_root / 'processed'
        self.processed_data_path.mkdir(exist_ok=True)
        
        # Processing results
        self.processing_results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'sources_processed': {},
            'summary': {
                'files_processed': 0,
                'records_processed': 0,
                'records_cleaned': 0,
                'processing_time': 0.0
            }
        }
    
    def _get_default_config(self) -> Dict:
        """Get default processing configuration"""
        return {
            'cleaning': {
                'remove_duplicates': True,
                'handle_missing': 'interpolate',  # 'drop', 'interpolate', 'forward_fill'
                'outlier_detection': True,
                'outlier_method': 'iqr',  # 'iqr', 'zscore'
                'outlier_threshold': 3.0
            },
            'transformation': {
                'normalize_prices': False,
                'calculate_returns': True,
                'resample_frequency': None,  # 'D', 'H', etc.
                'timezone': 'US/Eastern'
            },
            'validation': {
                'min_records': 10,
                'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume']
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processor"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - PROCESSOR - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_all_data(self) -> Dict[str, Any]:
        """
        Process all raw data sources
        
        Returns:
            Dict: Processing results summary
        """
        self.logger.info("⚙️ Starting data processing pipeline")
        start_time = datetime.now()
        
        # Process each data source
        for source_dir in self.raw_data_path.iterdir():
            if source_dir.is_dir():
                source_name = source_dir.name
                self.logger.info(f"📊 Processing {source_name} data...")
                
                source_result = self._process_source_data(source_dir, source_name)
                self.processing_results['sources_processed'][source_name] = source_result
                
                # Update summary
                self.processing_results['summary']['files_processed'] += source_result.get('files_processed', 0)
                self.processing_results['summary']['records_processed'] += source_result.get('records_processed', 0)
                self.processing_results['summary']['records_cleaned'] += source_result.get('records_cleaned', 0)
        
        # Calculate processing time
        end_time = datetime.now()
        self.processing_results['summary']['processing_time'] = (end_time - start_time).total_seconds()
        self.processing_results['status'] = 'completed'
        
        # Save processing report
        self._save_processing_report()
        
        self.logger.info(f"✅ Data processing completed in {self.processing_results['summary']['processing_time']:.2f} seconds")
        self.logger.info(f"📊 Processed {self.processing_results['summary']['records_processed']} records")
        
        return self.processing_results
    
    def _process_source_data(self, source_dir: Path, source_name: str) -> Dict[str, Any]:
        """
        Process data from a specific source
        
        Args:
            source_dir (Path): Source data directory
            source_name (str): Name of the data source
            
        Returns:
            Dict: Source processing results
        """
        result = {
            'source': source_name,
            'files_processed': 0,
            'records_processed': 0,
            'records_cleaned': 0,
            'output_files': []
        }
        
        if source_name == 'polygon':
            result.update(self._process_polygon_data(source_dir))
        elif source_name == 'trading_economics':
            result.update(self._process_trading_economics_data(source_dir))
        else:
            self.logger.warning(f"Unknown source: {source_name}")
        
        return result
    
    def _process_polygon_data(self, source_dir: Path) -> Dict[str, Any]:
        """Process Polygon.io data"""
        result = {'files_processed': 0, 'records_processed': 0, 'records_cleaned': 0, 'output_files': []}
        
        # Find all batch files
        batch_files = list(source_dir.glob('*_batch_*.json'))
        
        if not batch_files:
            self.logger.warning("No Polygon batch files found")
            return result
        
        # Process the latest batch file
        latest_batch = max(batch_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_batch, 'r') as f:
                batch_data = json.load(f)
            
            result['files_processed'] = 1
            
            # Extract and process historical data
            if 'data' in batch_data and 'historical_data' in batch_data['data']:
                historical_data = batch_data['data']['historical_data']
                
                if historical_data:
                    df = pd.DataFrame(historical_data)
                    result['records_processed'] = len(df)
                    
                    # Clean and process the data
                    cleaned_df = self._clean_ohlcv_data(df)
                    result['records_cleaned'] = len(cleaned_df)
                    
                    # Save processed data
                    ticker = batch_data['metadata'].get('ticker', 'AAPL')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    output_file = self.processed_data_path / f"{ticker}_processed_{timestamp}.csv"
                    cleaned_df.to_csv(output_file)
                    result['output_files'].append(str(output_file))
                    
                    self.logger.info(f"✅ Processed {len(cleaned_df)} records for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error processing Polygon data: {str(e)}")
        
        return result
    
    def _process_trading_economics_data(self, source_dir: Path) -> Dict[str, Any]:
        """Process Trading Economics data"""
        result = {'files_processed': 0, 'records_processed': 0, 'records_cleaned': 0, 'output_files': []}
        
        # Find JSON files
        json_files = list(source_dir.glob('*.json'))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                result['files_processed'] += 1
                
                # Process based on data structure
                if isinstance(data, list):
                    result['records_processed'] += len(data)
                elif isinstance(data, dict) and 'results' in data:
                    result['records_processed'] += len(data['results'])
                
                # For now, just copy the data (Trading Economics data is usually clean)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.processed_data_path / f"trading_economics_{timestamp}.json"
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                result['output_files'].append(str(output_file))
                result['records_cleaned'] = result['records_processed']
                
            except Exception as e:
                self.logger.error(f"Error processing Trading Economics file {json_file}: {str(e)}")
        
        return result
    
    def _clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        self.logger.info("🧹 Cleaning OHLCV data...")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Ensure proper column names
        column_mapping = {
            'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume',
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in cleaned_df.columns:
                cleaned_df.rename(columns={old_col: new_col}, inplace=True)
        
        # Handle date column
        if 'date' in cleaned_df.columns:
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
            cleaned_df.set_index('date', inplace=True)
        elif 't' in cleaned_df.columns:
            # Polygon timestamp format
            cleaned_df['date'] = pd.to_datetime(cleaned_df['t'], unit='ms')
            cleaned_df.set_index('date', inplace=True)
            cleaned_df.drop('t', axis=1, inplace=True)
        
        # Sort by date
        cleaned_df.sort_index(inplace=True)
        
        # Remove duplicates
        if self.config['cleaning']['remove_duplicates']:
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='last')]
            removed_duplicates = initial_count - len(cleaned_df)
            if removed_duplicates > 0:
                self.logger.info(f"🗑️ Removed {removed_duplicates} duplicate records")
        
        # Handle missing values
        if self.config['cleaning']['handle_missing'] == 'drop':
            cleaned_df.dropna(inplace=True)
        elif self.config['cleaning']['handle_missing'] == 'interpolate':
            cleaned_df.interpolate(method='linear', inplace=True)
        elif self.config['cleaning']['handle_missing'] == 'forward_fill':
            cleaned_df.fillna(method='ffill', inplace=True)
        
        # Outlier detection and removal
        if self.config['cleaning']['outlier_detection']:
            cleaned_df = self._remove_outliers(cleaned_df)
        
        # Calculate returns if requested
        if self.config['transformation']['calculate_returns']:
            if 'Close' in cleaned_df.columns:
                cleaned_df['Returns'] = cleaned_df['Close'].pct_change()
                cleaned_df['Log_Returns'] = np.log(cleaned_df['Close'] / cleaned_df['Close'].shift(1))
        
        # Remove any remaining NaN values
        cleaned_df.dropna(inplace=True)
        
        self.logger.info(f"✅ Data cleaning completed: {len(cleaned_df)} clean records")
        
        return cleaned_df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data"""
        if self.config['cleaning']['outlier_method'] == 'iqr':
            return self._remove_outliers_iqr(df)
        elif self.config['cleaning']['outlier_method'] == 'zscore':
            return self._remove_outliers_zscore(df)
        else:
            return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                df = df[~outliers]
                self.logger.info(f"🎯 Removed {outlier_count} outliers from {column}")
        
        return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        threshold = self.config['cleaning']['outlier_threshold']
        
        for column in numeric_columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                df = df[~outliers]
                self.logger.info(f"🎯 Removed {outlier_count} outliers from {column}")
        
        return df
    
    def _save_processing_report(self):
        """Save processing report to file"""
        reports_dir = self.data_root / 'processing_reports'
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"processing_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.processing_results, f, indent=2, default=str)
        
        self.logger.info(f"📄 Processing report saved to: {report_file}")

def main():
    """Test the data processor"""
    processor = DataProcessor()
    
    print("⚙️ Testing Data Processor")
    print("=" * 30)
    
    results = processor.process_all_data()
    
    print(f"\n📊 Processing Results:")
    print(f"   Status: {results['status']}")
    print(f"   Files Processed: {results['summary']['files_processed']}")
    print(f"   Records Processed: {results['summary']['records_processed']}")
    print(f"   Records Cleaned: {results['summary']['records_cleaned']}")
    print(f"   Processing Time: {results['summary']['processing_time']:.2f}s")

if __name__ == "__main__":
    main()
