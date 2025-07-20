#!/usr/bin/env python3
"""
Data Source Integration Template
Template for integrating new data sources into the pipeline
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

class DataSourceCollector(ABC):
    """Abstract base class for all data source collectors"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.source_name = self.__class__.__name__.lower().replace('collector', '')
    
    @abstractmethod
    def collect_data(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        """Collect data for given symbols"""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate collected data quality"""
        pass
    
    def normalize_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data to unified schema"""
        # Implement schema normalization
        return data
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        return []
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limiting information"""
        return {"requests_per_minute": 60, "daily_limit": None}

# Example implementation
class NewDataSourceCollector(DataSourceCollector):
    """Template for new data source collector"""
    
    def collect_data(self, symbols: List[str], **kwargs) -> pd.DataFrame:
        # Implement data collection logic
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        # Implement validation logic
        return True
