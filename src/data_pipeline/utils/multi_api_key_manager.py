#!/usr/bin/env python3
"""
Multi-API Key Manager
Manages multiple API keys per data source for increased rate limits and redundancy
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class MultiAPIKeyManager:
    """
    Manages multiple API keys per data source
    Provides automatic key rotation, rate limiting, and failover
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the multi-API key manager"""
        self.config_path = config_path or project_root / 'config' / 'data_sources' / 'registry.json'
        self.logger = self._setup_logger()
        
        # Load configuration
        self.config = self._load_config()
        
        # Key usage tracking
        self.key_usage = defaultdict(lambda: defaultdict(int))  # source -> key -> usage_count
        self.key_last_used = defaultdict(lambda: defaultdict(float))  # source -> key -> timestamp
        self.key_errors = defaultdict(lambda: defaultdict(int))  # source -> key -> error_count
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Rate limiting
        self.rate_limits = {}
        self._initialize_rate_limits()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - API_KEY_MANAGER - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> Dict:
        """Load data source configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            return {"data_sources": {}}
    
    def _initialize_rate_limits(self):
        """Initialize rate limiting for each data source"""
        for source_name, source_config in self.config.get('data_sources', {}).items():
            rate_limit = source_config.get('rate_limit', '1_per_second')
            self.rate_limits[source_name] = self._parse_rate_limit(rate_limit)
    
    def _parse_rate_limit(self, rate_limit_str: str) -> Dict[str, float]:
        """Parse rate limit string into usable format"""
        if 'unlimited' in rate_limit_str.lower():
            return {'requests': float('inf'), 'period': 1.0}

        if 'varies' in rate_limit_str.lower():
            return {'requests': 1000, 'period': 86400.0}  # Default: 1000 per day

        # Parse formats like "500_per_day", "5_per_minute", "1_per_second"
        parts = rate_limit_str.lower().split('_per_')
        if len(parts) != 2:
            return {'requests': 1, 'period': 1.0}

        try:
            requests = int(parts[0])
        except ValueError:
            return {'requests': 1, 'period': 1.0}

        period_str = parts[1].replace('_per_key', '')

        period_map = {
            'second': 1.0,
            'minute': 60.0,
            'hour': 3600.0,
            'day': 86400.0
        }

        period = period_map.get(period_str, 1.0)
        return {'requests': requests, 'period': period}
    
    def get_available_keys(self, source_name: str) -> List[Dict[str, Any]]:
        """
        Get all available API keys for a data source
        
        Args:
            source_name (str): Name of the data source
            
        Returns:
            List[Dict]: Available API keys with metadata
        """
        source_config = self.config.get('data_sources', {}).get(source_name, {})
        api_keys = source_config.get('api_keys', [])
        
        available_keys = []
        for key_config in api_keys:
            if key_config.get('status') == 'active' and not key_config['key'].startswith('SLOT_'):
                available_keys.append(key_config)
        
        return available_keys
    
    def get_next_key(self, source_name: str) -> Optional[str]:
        """
        Get the next available API key for a data source
        Uses round-robin with rate limiting and error tracking
        
        Args:
            source_name (str): Name of the data source
            
        Returns:
            str: API key to use, or None if no keys available
        """
        with self.lock:
            available_keys = self.get_available_keys(source_name)
            
            if not available_keys:
                self.logger.warning(f"⚠️ No available API keys for {source_name}")
                return None
            
            # Find the best key to use
            best_key = None
            best_score = float('-inf')
            
            current_time = time.time()
            rate_limit = self.rate_limits.get(source_name, {'requests': 1, 'period': 1.0})
            
            for key_config in available_keys:
                key = key_config['key']
                
                # Skip keys with too many recent errors
                if self.key_errors[source_name][key] > 5:
                    continue
                
                # Check rate limiting
                last_used = self.key_last_used[source_name][key]
                time_since_last = current_time - last_used
                
                if time_since_last < (rate_limit['period'] / rate_limit['requests']):
                    continue  # Still rate limited
                
                # Calculate score (prefer less used keys with fewer errors)
                usage_count = self.key_usage[source_name][key]
                error_count = self.key_errors[source_name][key]
                
                score = -usage_count - (error_count * 10) + time_since_last
                
                if score > best_score:
                    best_score = score
                    best_key = key
            
            if best_key:
                # Update usage tracking
                self.key_usage[source_name][best_key] += 1
                self.key_last_used[source_name][best_key] = current_time
                
                self.logger.debug(f"🔑 Using API key for {source_name}: {best_key[:8]}...")
                return best_key
            
            self.logger.warning(f"⚠️ All API keys for {source_name} are rate limited or have errors")
            return None
    
    def report_key_error(self, source_name: str, api_key: str, error_type: str = 'general'):
        """
        Report an error with a specific API key
        
        Args:
            source_name (str): Name of the data source
            api_key (str): API key that had an error
            error_type (str): Type of error
        """
        with self.lock:
            self.key_errors[source_name][api_key] += 1
            self.logger.warning(f"⚠️ API key error for {source_name}: {error_type}")
            
            # If too many errors, temporarily disable the key
            if self.key_errors[source_name][api_key] > 10:
                self.logger.error(f"❌ API key for {source_name} disabled due to excessive errors")
    
    def report_key_success(self, source_name: str, api_key: str):
        """
        Report successful use of an API key
        
        Args:
            source_name (str): Name of the data source
            api_key (str): API key that was successful
        """
        with self.lock:
            # Reset error count on success
            if self.key_errors[source_name][api_key] > 0:
                self.key_errors[source_name][api_key] = max(0, self.key_errors[source_name][api_key] - 1)
    
    def add_api_key(self, source_name: str, api_key: str, tier: str = 'free', description: str = '') -> bool:
        """
        Add a new API key to a data source
        
        Args:
            source_name (str): Name of the data source
            api_key (str): New API key to add
            tier (str): Tier of the API key (free, premium, etc.)
            description (str): Description of the key
            
        Returns:
            bool: True if key was added successfully
        """
        try:
            # Find an available slot
            source_config = self.config.get('data_sources', {}).get(source_name, {})
            api_keys = source_config.get('api_keys', [])
            
            # Find first available slot
            for i, key_config in enumerate(api_keys):
                if key_config['key'].startswith('SLOT_') and key_config['status'] == 'available':
                    # Update the slot
                    api_keys[i] = {
                        'key': api_key,
                        'status': 'active',
                        'tier': tier,
                        'requests_per_day': key_config.get('requests_per_day', 1000),
                        'description': description or f"API key slot {i+1}"
                    }
                    
                    # Save updated configuration
                    with open(self.config_path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                    
                    self.logger.info(f"✅ Added API key to {source_name} slot {i+1}")
                    return True
            
            self.logger.error(f"❌ No available slots for {source_name}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add API key: {str(e)}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all API keys"""
        stats = {}
        
        for source_name in self.config.get('data_sources', {}):
            available_keys = self.get_available_keys(source_name)
            
            source_stats = {
                'total_keys': len(available_keys),
                'active_keys': len([k for k in available_keys if k['status'] == 'active']),
                'key_details': []
            }
            
            for key_config in available_keys:
                key = key_config['key']
                key_stats = {
                    'key_preview': key[:8] + '...' if len(key) > 8 else key,
                    'tier': key_config.get('tier', 'unknown'),
                    'usage_count': self.key_usage[source_name][key],
                    'error_count': self.key_errors[source_name][key],
                    'last_used': self.key_last_used[source_name][key],
                    'status': 'healthy' if self.key_errors[source_name][key] < 5 else 'degraded'
                }
                source_stats['key_details'].append(key_stats)
            
            stats[source_name] = source_stats
        
        return stats
    
    def reset_error_counts(self, source_name: str = None):
        """Reset error counts for all keys or a specific source"""
        with self.lock:
            if source_name:
                self.key_errors[source_name].clear()
                self.logger.info(f"🔄 Reset error counts for {source_name}")
            else:
                self.key_errors.clear()
                self.logger.info("🔄 Reset all error counts")

def main():
    """Test the multi-API key manager"""
    print("🧪 Testing Multi-API Key Manager")
    print("=" * 35)
    
    manager = MultiAPIKeyManager()
    
    # Show current configuration
    print("\n📊 Current API Key Configuration:")
    for source_name in manager.config.get('data_sources', {}):
        available_keys = manager.get_available_keys(source_name)
        print(f"\n🔧 {source_name}:")
        print(f"   Available keys: {len(available_keys)}")
        
        for i, key_config in enumerate(available_keys):
            key_preview = key_config['key'][:8] + '...' if len(key_config['key']) > 8 else key_config['key']
            print(f"   Slot {i+1}: {key_preview} ({key_config.get('tier', 'unknown')} tier)")
    
    # Test key rotation
    print(f"\n🔄 Testing Key Rotation:")
    for source_name in ['eodhd', 'alpha_vantage']:
        print(f"\n{source_name}:")
        for i in range(3):
            key = manager.get_next_key(source_name)
            if key:
                key_preview = key[:8] + '...' if len(key) > 8 else key
                print(f"   Request {i+1}: {key_preview}")
            else:
                print(f"   Request {i+1}: No key available")
    
    # Show usage stats
    print(f"\n📈 Usage Statistics:")
    stats = manager.get_usage_stats()
    for source_name, source_stats in stats.items():
        print(f"\n{source_name}:")
        print(f"   Total keys: {source_stats['total_keys']}")
        print(f"   Active keys: {source_stats['active_keys']}")

if __name__ == "__main__":
    main()
