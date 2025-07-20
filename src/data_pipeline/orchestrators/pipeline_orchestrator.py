#!/usr/bin/env python3
"""
Data Pipeline Orchestrator - Manages the complete data flow
Coordinates collection, validation, processing, and feature engineering
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class DataPipelineOrchestrator:
    """
    Main orchestrator for the Apple ML Trading data pipeline
    Manages the complete flow from data collection to feature engineering
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline orchestrator
        
        Args:
            config_path (str): Path to pipeline configuration file
        """
        self.project_root = project_root
        self.config_path = config_path or "config/pipelines/default_pipeline.json"
        self.config = self._load_config()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Pipeline state
        self.pipeline_state = {
            'status': 'initialized',
            'current_stage': None,
            'start_time': None,
            'end_time': None,
            'stages_completed': [],
            'errors': [],
            'metrics': {}
        }
        
        # Data paths
        self.data_paths = {
            'raw': self.project_root / 'data' / 'raw',
            'processed': self.project_root / 'data' / 'processed', 
            'features': self.project_root / 'data' / 'features',
            'models': self.project_root / 'data' / 'models',
            'exports': self.project_root / 'data' / 'exports'
        }
        
        # Ensure data directories exist
        for path in self.data_paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load pipeline configuration"""
        config_file = self.project_root / self.config_path
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'pipeline': {
                    'name': 'apple_ml_trading_pipeline',
                    'version': '1.0.0',
                    'stages': [
                        'data_collection',
                        'data_validation', 
                        'data_processing',
                        'feature_engineering',
                        'data_export'
                    ]
                },
                'collection': {
                    'sources': ['polygon', 'trading_economics'],
                    'frequency': 'hourly',
                    'batch_size': 100
                },
                'processing': {
                    'clean_data': True,
                    'handle_missing': 'interpolate',
                    'outlier_detection': True
                },
                'features': {
                    'technical_indicators': True,
                    'economic_indicators': True,
                    'sentiment_analysis': False
                }
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup pipeline logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - PIPELINE - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = self.project_root / 'logs'
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / 'pipeline.log'
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def run_pipeline(self, stages: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete data pipeline
        
        Args:
            stages (List[str]): Specific stages to run (default: all)
            
        Returns:
            Dict: Pipeline execution results
        """
        self.pipeline_state['status'] = 'running'
        self.pipeline_state['start_time'] = datetime.now()
        
        stages_to_run = stages or self.config['pipeline']['stages']
        
        self.logger.info(f"🚀 Starting pipeline: {self.config['pipeline']['name']}")
        self.logger.info(f"📊 Stages to run: {stages_to_run}")
        
        try:
            for stage in stages_to_run:
                self.pipeline_state['current_stage'] = stage
                self.logger.info(f"▶️ Starting stage: {stage}")
                
                stage_result = self._run_stage(stage)
                
                if stage_result['success']:
                    self.pipeline_state['stages_completed'].append(stage)
                    self.logger.info(f"✅ Stage completed: {stage}")
                else:
                    self.pipeline_state['errors'].append({
                        'stage': stage,
                        'error': stage_result['error'],
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.error(f"❌ Stage failed: {stage} - {stage_result['error']}")
                    
                    if self.config.get('pipeline', {}).get('stop_on_error', True):
                        break
            
            self.pipeline_state['status'] = 'completed'
            self.pipeline_state['end_time'] = datetime.now()
            
            # Calculate metrics
            duration = self.pipeline_state['end_time'] - self.pipeline_state['start_time']
            self.pipeline_state['metrics'] = {
                'duration_seconds': duration.total_seconds(),
                'stages_completed': len(self.pipeline_state['stages_completed']),
                'stages_failed': len(self.pipeline_state['errors']),
                'success_rate': len(self.pipeline_state['stages_completed']) / len(stages_to_run)
            }
            
            self.logger.info(f"🎉 Pipeline completed in {duration.total_seconds():.2f} seconds")
            self.logger.info(f"📊 Success rate: {self.pipeline_state['metrics']['success_rate']:.2%}")
            
        except Exception as e:
            self.pipeline_state['status'] = 'failed'
            self.pipeline_state['end_time'] = datetime.now()
            self.logger.error(f"💥 Pipeline failed: {str(e)}")
            
            self.pipeline_state['errors'].append({
                'stage': 'pipeline',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        # Save pipeline state
        self._save_pipeline_state()
        
        return self.pipeline_state
    
    def _run_stage(self, stage: str) -> Dict[str, Any]:
        """
        Run a specific pipeline stage
        
        Args:
            stage (str): Stage name to run
            
        Returns:
            Dict: Stage execution result
        """
        try:
            if stage == 'data_collection':
                return self._run_data_collection()
            elif stage == 'data_validation':
                return self._run_data_validation()
            elif stage == 'data_processing':
                return self._run_data_processing()
            elif stage == 'feature_engineering':
                return self._run_feature_engineering()
            elif stage == 'data_export':
                return self._run_data_export()
            else:
                return {
                    'success': False,
                    'error': f'Unknown stage: {stage}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _run_data_collection(self) -> Dict[str, Any]:
        """Run data collection stage"""
        self.logger.info("📊 Running data collection...")
        
        # Import collectors
        try:
            from src.data_pipeline.collectors.polygon_collector import PolygonDataCollector
            from src.data_pipeline.collectors.trading_economics_collector import TradingEconomicsDataCollector
            
            results = {}
            
            # Collect from Polygon.io
            if 'polygon' in self.config['collection']['sources']:
                polygon_collector = PolygonDataCollector()
                polygon_data = polygon_collector.collect_batch()
                results['polygon'] = polygon_data
                self.logger.info(f"✅ Polygon data collected: {len(polygon_data) if polygon_data else 0} records")
            
            # Collect from Trading Economics
            if 'trading_economics' in self.config['collection']['sources']:
                te_collector = TradingEconomicsDataCollector()
                te_data = te_collector.collect_batch()
                results['trading_economics'] = te_data
                self.logger.info(f"✅ Trading Economics data collected: {len(te_data) if te_data else 0} records")
            
            return {'success': True, 'data': results}
            
        except Exception as e:
            return {'success': False, 'error': f'Data collection failed: {str(e)}'}
    
    def _run_data_validation(self) -> Dict[str, Any]:
        """Run data validation stage"""
        self.logger.info("🔍 Running data validation...")
        
        try:
            from src.data_pipeline.validators.data_validator import DataValidator
            
            validator = DataValidator()
            validation_results = validator.validate_all_data()
            
            self.logger.info(f"✅ Data validation completed: {validation_results['summary']}")
            return {'success': True, 'validation': validation_results}
            
        except Exception as e:
            return {'success': False, 'error': f'Data validation failed: {str(e)}'}
    
    def _run_data_processing(self) -> Dict[str, Any]:
        """Run data processing stage"""
        self.logger.info("⚙️ Running data processing...")
        
        try:
            from src.data_pipeline.processors.data_processor import DataProcessor
            
            processor = DataProcessor()
            processing_results = processor.process_all_data()
            
            self.logger.info(f"✅ Data processing completed: {processing_results['summary']}")
            return {'success': True, 'processing': processing_results}
            
        except Exception as e:
            return {'success': False, 'error': f'Data processing failed: {str(e)}'}
    
    def _run_feature_engineering(self) -> Dict[str, Any]:
        """Run feature engineering stage"""
        self.logger.info("🔧 Running feature engineering...")
        
        try:
            from src.feature_engineering.technical_indicators import TechnicalIndicators
            
            # Load processed data
            processed_data_path = self.data_paths['processed'] / 'aapl_processed.csv'
            if processed_data_path.exists():
                data = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
                
                # Generate features
                indicators = TechnicalIndicators(data)
                features = indicators.calculate_all_indicators()
                
                # Save features
                features_path = self.data_paths['features'] / f'aapl_features_{datetime.now().strftime("%Y%m%d")}.csv'
                features.to_csv(features_path)
                
                self.logger.info(f"✅ Feature engineering completed: {len(features.columns)} features generated")
                return {'success': True, 'features': len(features.columns)}
            else:
                return {'success': False, 'error': 'No processed data found'}
                
        except Exception as e:
            return {'success': False, 'error': f'Feature engineering failed: {str(e)}'}
    
    def _run_data_export(self) -> Dict[str, Any]:
        """Run data export stage"""
        self.logger.info("📤 Running data export...")
        
        try:
            # Export final dataset
            features_dir = self.data_paths['features']
            latest_features = max(features_dir.glob('aapl_features_*.csv'), key=os.path.getctime)
            
            if latest_features:
                # Copy to exports
                export_path = self.data_paths['exports'] / f'aapl_final_{datetime.now().strftime("%Y%m%d")}.csv'
                import shutil
                shutil.copy2(latest_features, export_path)
                
                self.logger.info(f"✅ Data exported to: {export_path}")
                return {'success': True, 'export_path': str(export_path)}
            else:
                return {'success': False, 'error': 'No features data found for export'}
                
        except Exception as e:
            return {'success': False, 'error': f'Data export failed: {str(e)}'}
    
    def _save_pipeline_state(self):
        """Save pipeline state to file"""
        state_file = self.project_root / 'logs' / f'pipeline_state_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)
        
        self.logger.info(f"📁 Pipeline state saved to: {state_file}")

def main():
    """Run the pipeline orchestrator"""
    orchestrator = DataPipelineOrchestrator()
    result = orchestrator.run_pipeline()
    
    print(f"\n🎯 Pipeline Result:")
    print(f"   Status: {result['status']}")
    print(f"   Stages completed: {len(result['stages_completed'])}")
    print(f"   Errors: {len(result['errors'])}")
    
    if result.get('metrics'):
        print(f"   Duration: {result['metrics']['duration_seconds']:.2f} seconds")
        print(f"   Success rate: {result['metrics']['success_rate']:.2%}")

if __name__ == "__main__":
    main()
