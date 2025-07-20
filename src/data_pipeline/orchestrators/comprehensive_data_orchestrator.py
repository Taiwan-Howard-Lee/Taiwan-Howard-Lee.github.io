#!/usr/bin/env python3
"""
Comprehensive Data Collection Orchestrator
8-hour continuous collection session with intelligent API key management
"""

import asyncio
import threading
import time
import signal
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data_pipeline.utils.multi_api_key_manager import MultiAPIKeyManager
from src.data_pipeline.integrators.multi_source_integrator import MultiSourceDataIntegrator

@dataclass
class CollectionSession:
    """Data collection session configuration and state"""
    session_id: str
    start_time: datetime
    duration_hours: int = 8
    target_symbols: List[str] = None
    data_sources: List[str] = None
    status: str = "initialized"  # initialized, running, paused, completed, failed
    total_records_collected: int = 0
    total_api_calls: int = 0
    errors_encountered: int = 0
    last_checkpoint: datetime = None
    
    def __post_init__(self):
        if self.target_symbols is None:
            self.target_symbols = [
                # Major Tech
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                # Major Finance
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B',
                # Major Healthcare
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO',
                # Major Consumer
                'HD', 'MCD', 'NKE', 'COST', 'WMT', 'PG',
                # Major Energy
                'XOM', 'CVX', 'COP', 'EOG',
                # Major ETFs
                'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO'
            ]
        
        if self.data_sources is None:
            self.data_sources = [
                'eodhd', 'finnhub', 'financial_modeling_prep', 
                'yahoo_finance', 'alpha_vantage', 'polygon'
            ]

class ComprehensiveDataOrchestrator:
    """
    Production-ready 8-hour data collection orchestrator
    Maximizes data acquisition with intelligent API key management
    """
    
    def __init__(self, session_config: CollectionSession = None):
        """Initialize the comprehensive data orchestrator"""
        self.session = session_config or CollectionSession(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now()
        )
        
        # Data storage (initialize first)
        self.session_data_root = project_root / 'data' / 'orchestrator_sessions' / self.session.session_id
        self.session_data_root.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.key_manager = MultiAPIKeyManager()
        self.integrator = MultiSourceDataIntegrator()
        self.logger = self._setup_logger()

        # Session management
        self.is_running = False
        self.should_stop = False
        self.checkpoint_interval = 300  # 5 minutes
        self.status_report_interval = 1800  # 30 minutes
        
        # Collection state
        self.collection_stats = {
            'sources_stats': {},
            'symbol_stats': {},
            'hourly_progress': [],
            'api_usage': {},
            'data_quality_metrics': {}
        }
        
        # Rate limiting and scheduling
        self.source_priorities = {
            'eodhd': {'priority': 1, 'max_concurrent': 3, 'delay_between_calls': 1.0},
            'finnhub': {'priority': 2, 'max_concurrent': 2, 'delay_between_calls': 1.0},
            'financial_modeling_prep': {'priority': 3, 'max_concurrent': 1, 'delay_between_calls': 2.0},
            'yahoo_finance': {'priority': 4, 'max_concurrent': 5, 'delay_between_calls': 0.5},
            'alpha_vantage': {'priority': 5, 'max_concurrent': 1, 'delay_between_calls': 12.0},
            'polygon': {'priority': 6, 'max_concurrent': 1, 'delay_between_calls': 12.0}
        }
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - ORCHESTRATOR - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = self.session_data_root / f"{self.session.session_id}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True
    
    def start_8_hour_session(self) -> Dict[str, Any]:
        """
        Start the 8-hour comprehensive data collection session
        
        Returns:
            Dict: Session results and statistics
        """
        self.logger.info("🚀 Starting 8-hour comprehensive data collection session")
        self.logger.info(f"📊 Session ID: {self.session.session_id}")
        self.logger.info(f"🎯 Target symbols: {len(self.session.target_symbols)}")
        self.logger.info(f"🔧 Data sources: {len(self.session.data_sources)}")
        
        self.session.status = "running"
        self.is_running = True
        session_start = time.time()
        session_end = session_start + (self.session.duration_hours * 3600)
        
        try:
            # Initialize all collectors
            self._initialize_collectors()
            
            # Start background monitoring threads
            checkpoint_thread = threading.Thread(target=self._checkpoint_worker, daemon=True)
            status_thread = threading.Thread(target=self._status_reporter, daemon=True)
            
            checkpoint_thread.start()
            status_thread.start()
            
            # Main collection loop
            collection_cycle = 0
            while time.time() < session_end and not self.should_stop:
                collection_cycle += 1
                cycle_start = time.time()
                
                self.logger.info(f"🔄 Starting collection cycle {collection_cycle}")
                
                # Execute collection cycle
                cycle_results = self._execute_collection_cycle(collection_cycle)
                
                # Update session statistics
                self._update_session_stats(cycle_results)
                
                # Calculate next cycle timing
                cycle_duration = time.time() - cycle_start
                remaining_time = session_end - time.time()
                
                self.logger.info(f"✅ Cycle {collection_cycle} completed in {cycle_duration:.1f}s")
                self.logger.info(f"⏱️ Remaining session time: {remaining_time/3600:.1f} hours")
                
                # Adaptive delay based on API limits and remaining time
                if remaining_time > 0:
                    optimal_delay = self._calculate_optimal_delay(remaining_time, cycle_duration)
                    if optimal_delay > 0:
                        self.logger.info(f"⏸️ Waiting {optimal_delay:.1f}s before next cycle")
                        time.sleep(optimal_delay)
            
            # Session completion
            self.session.status = "completed" if not self.should_stop else "interrupted"
            self.is_running = False
            
            # Final checkpoint and summary
            final_results = self._generate_final_summary()
            self._save_session_checkpoint(final_results)
            
            self.logger.info("🎉 8-hour data collection session completed!")
            return final_results
            
        except Exception as e:
            self.session.status = "failed"
            self.is_running = False
            self.logger.error(f"❌ Session failed: {str(e)}")
            
            # Save error state
            error_results = self._generate_error_summary(str(e))
            self._save_session_checkpoint(error_results)
            
            raise e
    
    def _initialize_collectors(self):
        """Initialize all data source collectors"""
        self.logger.info("🔧 Initializing data source collectors...")
        
        # Test API key availability
        for source in self.session.data_sources:
            available_keys = self.key_manager.get_available_keys(source)
            self.logger.info(f"🔑 {source}: {len(available_keys)} API keys available")
            
            self.collection_stats['api_usage'][source] = {
                'available_keys': len(available_keys),
                'requests_made': 0,
                'errors': 0,
                'last_used': None
            }
        
        # Initialize integrator with all sources
        self.integrator._initialize_collectors()
        
        self.logger.info(f"✅ Initialized {len(self.integrator.collectors)} collectors")
    
    def _execute_collection_cycle(self, cycle_number: int) -> Dict[str, Any]:
        """
        Execute a single collection cycle across all sources
        
        Args:
            cycle_number (int): Current cycle number
            
        Returns:
            Dict: Cycle results and statistics
        """
        cycle_results = {
            'cycle_number': cycle_number,
            'start_time': datetime.now().isoformat(),
            'sources_processed': {},
            'total_records': 0,
            'total_api_calls': 0,
            'errors': []
        }
        
        # Determine symbols for this cycle (rotate through all symbols)
        symbols_per_cycle = min(10, len(self.session.target_symbols))
        start_idx = ((cycle_number - 1) * symbols_per_cycle) % len(self.session.target_symbols)
        end_idx = start_idx + symbols_per_cycle
        
        if end_idx > len(self.session.target_symbols):
            cycle_symbols = (self.session.target_symbols[start_idx:] + 
                           self.session.target_symbols[:end_idx - len(self.session.target_symbols)])
        else:
            cycle_symbols = self.session.target_symbols[start_idx:end_idx]
        
        self.logger.info(f"📊 Cycle {cycle_number} symbols: {cycle_symbols}")
        
        # Collect from sources in priority order
        for source in sorted(self.session.data_sources, 
                           key=lambda x: self.source_priorities.get(x, {}).get('priority', 999)):
            
            if self.should_stop:
                break
            
            try:
                self.logger.info(f"📊 Collecting from {source}...")
                source_start = time.time()
                
                # Get source-specific results
                source_results = self._collect_from_source(source, cycle_symbols)
                
                source_duration = time.time() - source_start
                
                if source_results:
                    cycle_results['sources_processed'][source] = {
                        'records_collected': len(source_results.get('data', [])),
                        'api_calls': source_results.get('api_calls', 0),
                        'duration': source_duration,
                        'success': True
                    }
                    
                    cycle_results['total_records'] += len(source_results.get('data', []))
                    cycle_results['total_api_calls'] += source_results.get('api_calls', 0)
                    
                    # Save source data incrementally
                    self._save_source_data(source, source_results, cycle_number)
                    
                    self.logger.info(f"✅ {source}: {len(source_results.get('data', []))} records in {source_duration:.1f}s")
                else:
                    cycle_results['sources_processed'][source] = {
                        'records_collected': 0,
                        'api_calls': 0,
                        'duration': source_duration,
                        'success': False
                    }
                    self.logger.warning(f"⚠️ {source}: No data collected")
                
                # Respect rate limits between sources
                source_config = self.source_priorities.get(source, {})
                delay = source_config.get('delay_between_calls', 1.0)
                time.sleep(delay)
                
            except Exception as e:
                self.logger.error(f"❌ {source} collection failed: {str(e)}")
                cycle_results['errors'].append({
                    'source': source,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update error statistics
                if source in self.collection_stats['api_usage']:
                    self.collection_stats['api_usage'][source]['errors'] += 1
        
        cycle_results['end_time'] = datetime.now().isoformat()
        cycle_results['duration'] = (datetime.fromisoformat(cycle_results['end_time']) - 
                                   datetime.fromisoformat(cycle_results['start_time'])).total_seconds()
        
        return cycle_results
    
    def _collect_from_source(self, source: str, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """
        Collect data from a specific source with API key management
        
        Args:
            source (str): Data source name
            symbols (List[str]): Symbols to collect
            
        Returns:
            Dict: Collection results
        """
        if source not in self.integrator.collectors:
            self.logger.warning(f"⚠️ Collector not available for {source}")
            return None
        
        collector = self.integrator.collectors[source]
        
        try:
            if source == 'eodhd':
                # Collect comprehensive data from EODHD
                results = {'data': [], 'api_calls': 0}
                for symbol in symbols[:5]:  # Limit for rate limits
                    symbol_data = collector.collect_comprehensive_data(f"{symbol}.US")
                    if symbol_data and symbol_data.get('data'):
                        results['data'].append(symbol_data)
                        results['api_calls'] += 8  # Estimate based on endpoints
                return results
                
            elif source == 'finnhub':
                # Collect comprehensive data from Finnhub
                results = {'data': [], 'api_calls': 0}
                for symbol in symbols[:3]:  # Limit for rate limits
                    symbol_data = collector.collect_comprehensive_data(symbol)
                    if symbol_data and symbol_data.get('data'):
                        results['data'].append(symbol_data)
                        results['api_calls'] += 8  # Estimate based on endpoints
                return results
                
            elif source == 'financial_modeling_prep':
                # Collect comprehensive data from FMP
                results = {'data': [], 'api_calls': 0}
                for symbol in symbols[:2]:  # Limit for rate limits
                    symbol_data = collector.collect_comprehensive_data(symbol)
                    if symbol_data and symbol_data.get('data'):
                        results['data'].append(symbol_data)
                        results['api_calls'] += 8  # Estimate based on endpoints
                return results
                
            elif source == 'yahoo_finance':
                # Collect from Yahoo Finance
                yahoo_results = collector.collect_all_data(period="3mo")
                if yahoo_results.get('symbols_processed', 0) > 0:
                    combined_data = collector.create_combined_dataset()
                    return {
                        'data': [combined_data] if not combined_data.empty else [],
                        'api_calls': yahoo_results.get('symbols_processed', 0)
                    }
                
            elif source == 'alpha_vantage':
                # Collect from Alpha Vantage with rate limiting
                av_results = collector.collect_batch_data(symbols=symbols[:3])
                if av_results.get('symbols_processed', 0) > 0:
                    combined_data = collector.create_combined_dataset()
                    return {
                        'data': [combined_data] if not combined_data.empty else [],
                        'api_calls': av_results.get('symbols_processed', 0)
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ {source} collection error: {str(e)}")
            return None

    def _save_source_data(self, source: str, data: Dict[str, Any], cycle: int):
        """Save collected data incrementally"""
        try:
            source_dir = self.session_data_root / source
            source_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{source}_cycle_{cycle}_{timestamp}.json"
            filepath = source_dir / filename

            # Convert any DataFrames to JSON-serializable format
            serializable_data = self._make_json_serializable(data)

            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)

            self.logger.debug(f"💾 Saved {source} data to {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save {source} data: {str(e)}")

    def _make_json_serializable(self, obj):
        """Convert complex objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj

    def _update_session_stats(self, cycle_results: Dict[str, Any]):
        """Update session-wide statistics"""
        self.session.total_records_collected += cycle_results['total_records']
        self.session.total_api_calls += cycle_results['total_api_calls']
        self.session.errors_encountered += len(cycle_results['errors'])

        # Update hourly progress
        current_hour = int((time.time() - self.session.start_time.timestamp()) / 3600)

        if len(self.collection_stats['hourly_progress']) <= current_hour:
            self.collection_stats['hourly_progress'].extend([
                {'hour': i, 'records': 0, 'api_calls': 0, 'errors': 0}
                for i in range(len(self.collection_stats['hourly_progress']), current_hour + 1)
            ])

        if current_hour < len(self.collection_stats['hourly_progress']):
            self.collection_stats['hourly_progress'][current_hour]['records'] += cycle_results['total_records']
            self.collection_stats['hourly_progress'][current_hour]['api_calls'] += cycle_results['total_api_calls']
            self.collection_stats['hourly_progress'][current_hour]['errors'] += len(cycle_results['errors'])

        # Update source statistics
        for source, stats in cycle_results['sources_processed'].items():
            if source not in self.collection_stats['sources_stats']:
                self.collection_stats['sources_stats'][source] = {
                    'total_records': 0,
                    'total_api_calls': 0,
                    'total_errors': 0,
                    'success_rate': 0.0,
                    'avg_duration': 0.0,
                    'cycles_processed': 0
                }

            source_stat = self.collection_stats['sources_stats'][source]
            source_stat['total_records'] += stats['records_collected']
            source_stat['total_api_calls'] += stats['api_calls']
            source_stat['cycles_processed'] += 1

            if not stats['success']:
                source_stat['total_errors'] += 1

            # Update averages
            source_stat['success_rate'] = (
                (source_stat['cycles_processed'] - source_stat['total_errors']) /
                source_stat['cycles_processed']
            )

            # Update API usage tracking
            if source in self.collection_stats['api_usage']:
                self.collection_stats['api_usage'][source]['requests_made'] += stats['api_calls']
                self.collection_stats['api_usage'][source]['last_used'] = datetime.now().isoformat()

    def _calculate_optimal_delay(self, remaining_time: float, last_cycle_duration: float) -> float:
        """Calculate optimal delay between cycles"""
        # Base delay to respect API limits
        base_delay = 60  # 1 minute minimum

        # Adjust based on remaining time and cycle efficiency
        if remaining_time > 3600:  # More than 1 hour remaining
            return base_delay
        elif remaining_time > 1800:  # More than 30 minutes remaining
            return base_delay * 0.75
        elif remaining_time > 600:  # More than 10 minutes remaining
            return base_delay * 0.5
        else:  # Less than 10 minutes remaining
            return base_delay * 0.25

    def _checkpoint_worker(self):
        """Background worker for periodic checkpoints"""
        while self.is_running:
            try:
                time.sleep(self.checkpoint_interval)
                if self.is_running:
                    self._save_session_checkpoint()
                    self.session.last_checkpoint = datetime.now()
                    self.logger.info("💾 Session checkpoint saved")
            except Exception as e:
                self.logger.error(f"❌ Checkpoint failed: {str(e)}")

    def _status_reporter(self):
        """Background worker for periodic status reports"""
        while self.is_running:
            try:
                time.sleep(self.status_report_interval)
                if self.is_running:
                    self._generate_status_report()
            except Exception as e:
                self.logger.error(f"❌ Status report failed: {str(e)}")

    def _generate_status_report(self):
        """Generate and log periodic status report"""
        elapsed_time = (datetime.now() - self.session.start_time).total_seconds()
        elapsed_hours = elapsed_time / 3600

        self.logger.info("📊 === STATUS REPORT ===")
        self.logger.info(f"⏱️ Elapsed time: {elapsed_hours:.1f} hours")
        self.logger.info(f"📊 Total records collected: {self.session.total_records_collected:,}")
        self.logger.info(f"📞 Total API calls made: {self.session.total_api_calls:,}")
        self.logger.info(f"❌ Errors encountered: {self.session.errors_encountered}")

        # Source-wise statistics
        for source, stats in self.collection_stats['sources_stats'].items():
            self.logger.info(f"   {source}: {stats['total_records']:,} records, "
                           f"{stats['success_rate']:.1%} success rate")

        # API usage statistics
        api_stats = self.key_manager.get_usage_stats()
        for source, stats in api_stats.items():
            if stats['active_keys'] > 0:
                self.logger.info(f"🔑 {source}: {stats['active_keys']} keys active")

        self.logger.info("📊 === END REPORT ===")

    def _save_session_checkpoint(self, additional_data: Dict = None):
        """Save session checkpoint"""
        checkpoint_data = {
            'session': asdict(self.session),
            'collection_stats': self.collection_stats,
            'timestamp': datetime.now().isoformat()
        }

        if additional_data:
            checkpoint_data.update(additional_data)

        checkpoint_file = self.session_data_root / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {str(e)}")

    def _generate_final_summary(self) -> Dict[str, Any]:
        """Generate final session summary"""
        session_duration = (datetime.now() - self.session.start_time).total_seconds()

        summary = {
            'session_summary': {
                'session_id': self.session.session_id,
                'duration_hours': session_duration / 3600,
                'status': self.session.status,
                'total_records_collected': self.session.total_records_collected,
                'total_api_calls': self.session.total_api_calls,
                'errors_encountered': self.session.errors_encountered,
                'records_per_hour': self.session.total_records_collected / (session_duration / 3600),
                'api_calls_per_hour': self.session.total_api_calls / (session_duration / 3600)
            },
            'source_performance': self.collection_stats['sources_stats'],
            'hourly_breakdown': self.collection_stats['hourly_progress'],
            'api_usage_final': self.key_manager.get_usage_stats(),
            'data_quality_metrics': self._calculate_data_quality_metrics()
        }

        return summary

    def _generate_error_summary(self, error_message: str) -> Dict[str, Any]:
        """Generate error summary for failed sessions"""
        return {
            'session_id': self.session.session_id,
            'status': 'failed',
            'error_message': error_message,
            'partial_results': self.collection_stats,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_data_quality_metrics(self) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        total_sources = len(self.session.data_sources)
        successful_sources = len([s for s in self.collection_stats['sources_stats'].values()
                                if s['success_rate'] > 0.5])

        return {
            'source_success_rate': successful_sources / total_sources if total_sources > 0 else 0,
            'overall_error_rate': (self.session.errors_encountered /
                                 max(self.session.total_api_calls, 1)),
            'data_completeness': min(1.0, self.session.total_records_collected / 10000),  # Target 10k records
            'api_efficiency': (self.session.total_records_collected /
                             max(self.session.total_api_calls, 1))
        }

    def resume_session(self, checkpoint_file: str) -> Dict[str, Any]:
        """Resume a session from checkpoint"""
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Restore session state
            session_data = checkpoint_data['session']
            self.session = CollectionSession(**session_data)
            self.collection_stats = checkpoint_data['collection_stats']

            self.logger.info(f"🔄 Resuming session {self.session.session_id}")
            self.logger.info(f"📊 Previous progress: {self.session.total_records_collected:,} records")

            # Continue session
            remaining_time = (self.session.start_time +
                            timedelta(hours=self.session.duration_hours) -
                            datetime.now()).total_seconds()

            if remaining_time > 0:
                self.session.duration_hours = remaining_time / 3600
                return self.start_8_hour_session()
            else:
                self.logger.warning("⚠️ Session time already expired")
                return self._generate_final_summary()

        except Exception as e:
            self.logger.error(f"❌ Failed to resume session: {str(e)}")
            raise e

def main():
    """Test the comprehensive data orchestrator"""
    print("🧪 Testing Comprehensive Data Orchestrator")
    print("=" * 50)

    # Create test session (shorter duration for testing)
    test_session = CollectionSession(
        session_id=f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=datetime.now(),
        duration_hours=0.5,  # 30 minutes for testing
        target_symbols=['AAPL', 'MSFT', 'GOOGL']  # Smaller set for testing
    )

    orchestrator = ComprehensiveDataOrchestrator(test_session)

    try:
        results = orchestrator.start_8_hour_session()

        print(f"\n🎉 Test Session Completed!")
        print(f"📊 Records collected: {results['session_summary']['total_records_collected']:,}")
        print(f"📞 API calls made: {results['session_summary']['total_api_calls']:,}")
        print(f"⏱️ Duration: {results['session_summary']['duration_hours']:.2f} hours")
        print(f"📈 Records per hour: {results['session_summary']['records_per_hour']:.0f}")

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main()
