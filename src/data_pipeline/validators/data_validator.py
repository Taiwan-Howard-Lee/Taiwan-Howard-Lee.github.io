#!/usr/bin/env python3
"""
Data Validator - Ensures data quality and integrity
Validates raw data from all sources before processing
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class DataValidator:
    """
    Comprehensive data validator for the Apple ML Trading pipeline
    Validates data quality, completeness, and consistency
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the data validator
        
        Args:
            config (Dict): Validation configuration
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Data paths
        self.data_root = project_root / 'data'
        self.raw_data_path = self.data_root / 'raw'
        
        # Validation results
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'pending',
            'sources': {},
            'summary': {
                'total_files': 0,
                'valid_files': 0,
                'invalid_files': 0,
                'warnings': 0,
                'errors': 0
            }
        }
    
    def _get_default_config(self) -> Dict:
        """Get default validation configuration"""
        return {
            'price_validation': {
                'min_price': 1.0,
                'max_price': 1000.0,
                'max_daily_change': 0.20  # 20% max daily change
            },
            'volume_validation': {
                'min_volume': 1000,
                'max_volume': 1000000000  # 1B shares
            },
            'date_validation': {
                'min_date': '2020-01-01',
                'max_future_days': 1  # Allow 1 day in future
            },
            'completeness': {
                'required_fields': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'max_missing_ratio': 0.05  # 5% max missing data
            },
            'consistency': {
                'check_ohlc_logic': True,  # High >= Open,Close,Low etc.
                'check_volume_positive': True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the validator"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - VALIDATOR - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_all_data(self) -> Dict[str, Any]:
        """
        Validate all data in the raw data directory
        
        Returns:
            Dict: Comprehensive validation results
        """
        self.logger.info("🔍 Starting comprehensive data validation")
        
        # Validate each data source
        for source_dir in self.raw_data_path.iterdir():
            if source_dir.is_dir():
                source_name = source_dir.name
                self.logger.info(f"📊 Validating {source_name} data...")
                
                source_results = self._validate_source_data(source_dir, source_name)
                self.validation_results['sources'][source_name] = source_results
                
                # Update summary
                self.validation_results['summary']['total_files'] += source_results['total_files']
                self.validation_results['summary']['valid_files'] += source_results['valid_files']
                self.validation_results['summary']['invalid_files'] += source_results['invalid_files']
                self.validation_results['summary']['warnings'] += len(source_results['warnings'])
                self.validation_results['summary']['errors'] += len(source_results['errors'])
        
        # Determine overall status
        if self.validation_results['summary']['errors'] == 0:
            if self.validation_results['summary']['warnings'] == 0:
                self.validation_results['overall_status'] = 'passed'
            else:
                self.validation_results['overall_status'] = 'passed_with_warnings'
        else:
            self.validation_results['overall_status'] = 'failed'
        
        # Save validation report
        self._save_validation_report()
        
        self.logger.info(f"✅ Validation completed: {self.validation_results['overall_status']}")
        self.logger.info(f"📊 Files: {self.validation_results['summary']['valid_files']}/{self.validation_results['summary']['total_files']} valid")
        
        return self.validation_results
    
    def _validate_source_data(self, source_dir: Path, source_name: str) -> Dict[str, Any]:
        """
        Validate data from a specific source
        
        Args:
            source_dir (Path): Source data directory
            source_name (str): Name of the data source
            
        Returns:
            Dict: Source validation results
        """
        results = {
            'source': source_name,
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'warnings': [],
            'errors': [],
            'file_results': {}
        }
        
        # Validate each file in the source directory
        for file_path in source_dir.iterdir():
            if file_path.is_file():
                results['total_files'] += 1
                
                file_result = self._validate_file(file_path, source_name)
                results['file_results'][file_path.name] = file_result
                
                if file_result['valid']:
                    results['valid_files'] += 1
                else:
                    results['invalid_files'] += 1
                
                results['warnings'].extend(file_result['warnings'])
                results['errors'].extend(file_result['errors'])
        
        return results
    
    def _validate_file(self, file_path: Path, source_name: str) -> Dict[str, Any]:
        """
        Validate a specific data file
        
        Args:
            file_path (Path): Path to the file
            source_name (str): Name of the data source
            
        Returns:
            Dict: File validation results
        """
        result = {
            'file': file_path.name,
            'valid': True,
            'warnings': [],
            'errors': [],
            'checks_performed': []
        }
        
        try:
            # Determine file type and validate accordingly
            if file_path.suffix == '.json':
                result.update(self._validate_json_file(file_path, source_name))
            elif file_path.suffix == '.csv':
                result.update(self._validate_csv_file(file_path, source_name))
            else:
                result['warnings'].append(f"Unknown file type: {file_path.suffix}")
            
            # File is invalid if it has errors
            if result['errors']:
                result['valid'] = False
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation failed: {str(e)}")
        
        return result
    
    def _validate_json_file(self, file_path: Path, source_name: str) -> Dict[str, Any]:
        """Validate JSON data file"""
        result = {'checks_performed': [], 'warnings': [], 'errors': []}
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            result['checks_performed'].append('json_structure')
            
            # Validate JSON structure based on source
            if source_name == 'polygon':
                result.update(self._validate_polygon_json(data, file_path.name))
            elif source_name == 'trading_economics':
                result.update(self._validate_trading_economics_json(data, file_path.name))
            
        except json.JSONDecodeError as e:
            result['errors'].append(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            result['errors'].append(f"JSON validation error: {str(e)}")
        
        return result
    
    def _validate_csv_file(self, file_path: Path, source_name: str) -> Dict[str, Any]:
        """Validate CSV data file"""
        result = {'checks_performed': [], 'warnings': [], 'errors': []}
        
        try:
            df = pd.read_csv(file_path)
            result['checks_performed'].append('csv_structure')
            
            # Basic CSV validation
            if df.empty:
                result['errors'].append("CSV file is empty")
                return result
            
            # Validate OHLCV data if present
            if self._is_ohlcv_data(df):
                result.update(self._validate_ohlcv_data(df, file_path.name))
            
            # Check for missing data
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_ratio > self.config['completeness']['max_missing_ratio']:
                result['warnings'].append(f"High missing data ratio: {missing_ratio:.2%}")
            
        except Exception as e:
            result['errors'].append(f"CSV validation error: {str(e)}")
        
        return result
    
    def _validate_polygon_json(self, data: Dict, filename: str) -> Dict[str, Any]:
        """Validate Polygon.io JSON data structure"""
        result = {'checks_performed': [], 'warnings': [], 'errors': []}
        
        # Check required top-level keys
        required_keys = ['metadata', 'data', 'metrics']
        for key in required_keys:
            if key not in data:
                result['errors'].append(f"Missing required key: {key}")
        
        result['checks_performed'].append('polygon_structure')
        
        # Validate metadata
        if 'metadata' in data:
            metadata = data['metadata']
            if 'ticker' not in metadata:
                result['errors'].append("Missing ticker in metadata")
            if 'collection_timestamp' not in metadata:
                result['warnings'].append("Missing collection timestamp")
        
        # Validate data section
        if 'data' in data:
            data_section = data['data']
            
            # Check historical data if present
            if 'historical_data' in data_section and data_section['historical_data']:
                historical_df = pd.DataFrame(data_section['historical_data'])
                if self._is_ohlcv_data(historical_df):
                    ohlcv_result = self._validate_ohlcv_data(historical_df, filename)
                    result['warnings'].extend(ohlcv_result['warnings'])
                    result['errors'].extend(ohlcv_result['errors'])
                    result['checks_performed'].extend(ohlcv_result['checks_performed'])
        
        return result
    
    def _validate_trading_economics_json(self, data: Dict, filename: str) -> Dict[str, Any]:
        """Validate Trading Economics JSON data structure"""
        result = {'checks_performed': [], 'warnings': [], 'errors': []}
        
        # Basic structure validation for Trading Economics data
        if isinstance(data, list):
            if not data:
                result['warnings'].append("Empty data array")
        elif isinstance(data, dict):
            if 'results' in data and not data['results']:
                result['warnings'].append("Empty results array")
        
        result['checks_performed'].append('trading_economics_structure')
        return result
    
    def _is_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains OHLCV data"""
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in df.columns for col in ohlcv_columns)
    
    def _validate_ohlcv_data(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Validate OHLCV data quality"""
        result = {'checks_performed': [], 'warnings': [], 'errors': []}
        
        # Price validation
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                # Check price range
                min_price = df[col].min()
                max_price = df[col].max()
                
                if min_price < self.config['price_validation']['min_price']:
                    result['errors'].append(f"{col} contains unrealistic low prices: {min_price}")
                
                if max_price > self.config['price_validation']['max_price']:
                    result['errors'].append(f"{col} contains unrealistic high prices: {max_price}")
                
                # Check for negative prices
                if (df[col] < 0).any():
                    result['errors'].append(f"{col} contains negative prices")
        
        result['checks_performed'].append('price_validation')
        
        # OHLC logic validation
        if self.config['consistency']['check_ohlc_logic']:
            # High should be >= Open, Close, Low
            if 'High' in df.columns and 'Low' in df.columns:
                if (df['High'] < df['Low']).any():
                    result['errors'].append("High price is less than Low price")
            
            # Check other OHLC relationships
            for col in ['Open', 'Close']:
                if col in df.columns and 'High' in df.columns:
                    if (df[col] > df['High']).any():
                        result['warnings'].append(f"{col} exceeds High price")
                
                if col in df.columns and 'Low' in df.columns:
                    if (df[col] < df['Low']).any():
                        result['warnings'].append(f"{col} below Low price")
        
        result['checks_performed'].append('ohlc_logic')
        
        # Volume validation
        if 'Volume' in df.columns:
            if self.config['consistency']['check_volume_positive']:
                if (df['Volume'] < 0).any():
                    result['errors'].append("Volume contains negative values")
            
            # Check volume range
            min_vol = df['Volume'].min()
            max_vol = df['Volume'].max()
            
            if min_vol < self.config['volume_validation']['min_volume']:
                result['warnings'].append(f"Very low volume detected: {min_vol}")
            
            if max_vol > self.config['volume_validation']['max_volume']:
                result['warnings'].append(f"Very high volume detected: {max_vol}")
        
        result['checks_performed'].append('volume_validation')
        
        return result
    
    def _save_validation_report(self):
        """Save validation report to file"""
        reports_dir = self.data_root / 'validation_reports'
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        self.logger.info(f"📄 Validation report saved to: {report_file}")

def main():
    """Test the data validator"""
    validator = DataValidator()
    
    print("🔍 Testing Data Validator")
    print("=" * 30)
    
    results = validator.validate_all_data()
    
    print(f"\n📊 Validation Results:")
    print(f"   Overall Status: {results['overall_status']}")
    print(f"   Total Files: {results['summary']['total_files']}")
    print(f"   Valid Files: {results['summary']['valid_files']}")
    print(f"   Invalid Files: {results['summary']['invalid_files']}")
    print(f"   Warnings: {results['summary']['warnings']}")
    print(f"   Errors: {results['summary']['errors']}")

if __name__ == "__main__":
    main()
