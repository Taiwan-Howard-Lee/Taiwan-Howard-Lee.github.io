#!/usr/bin/env python3
"""
Complete Codebase Organization Script
Organizes the Apple ML Trading system for optimal maintainability and scalability
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def organize_codebase():
    """Complete codebase organization"""
    
    print("🏗️ Apple ML Trading - Complete Codebase Organization")
    print("=" * 55)
    print("📁 Organizing for optimal maintainability and scalability...")
    print()
    
    project_root = Path.cwd()
    
    # Define the ideal directory structure
    ideal_structure = {
        # Core application code
        'src/': {
            'data_pipeline/': {
                'collectors/': 'Data collection from various sources',
                'processors/': 'Data cleaning and transformation',
                'validators/': 'Data quality validation',
                'orchestrators/': 'Pipeline workflow management',
                'integrators/': 'Multi-source data integration'
            },
            'feature_engineering/': {
                'technical_indicators/': 'Technical analysis indicators',
                'economic_indicators/': 'Economic and fundamental data',
                'sentiment_analysis/': 'News and social sentiment',
                'sector_analysis/': 'Sector rotation and breadth'
            },
            'models/': {
                'rl_agents/': 'Reinforcement learning agents',
                'ml_models/': 'Traditional ML models',
                'ensemble/': 'Model combination strategies',
                'optimization/': 'Parameter optimization'
            },
            'backtesting/': {
                'engines/': 'Backtesting frameworks',
                'metrics/': 'Performance calculation',
                'visualization/': 'Results visualization'
            },
            'risk_metrics/': {
                'portfolio/': 'Portfolio risk analysis',
                'market/': 'Market risk assessment',
                'correlation/': 'Asset correlation analysis'
            },
            'utils/': {
                'data_utils/': 'Data manipulation utilities',
                'math_utils/': 'Mathematical calculations',
                'api_utils/': 'API interaction helpers'
            }
        },
        
        # Configuration management
        'config/': {
            'data_sources/': 'Data source configurations',
            'pipelines/': 'Pipeline configurations',
            'models/': 'Model configurations',
            'environments/': 'Environment-specific settings',
            'schemas/': 'Data schema definitions'
        },
        
        # Data storage (organized by source and processing stage)
        'data/': {
            'raw/': {
                'yahoo_finance/': 'Yahoo Finance data',
                'alpha_vantage/': 'Alpha Vantage data',
                'polygon/': 'Polygon.io data',
                'trading_economics/': 'Trading Economics data',
                'fred/': 'Federal Reserve Economic Data',
                'news/': 'News and sentiment data',
                'alternative/': 'Alternative data sources'
            },
            'processed/': {
                'daily/': 'Daily processed data',
                'intraday/': 'Intraday processed data',
                'fundamental/': 'Fundamental data',
                'economic/': 'Economic indicators'
            },
            'features/': {
                'technical/': 'Technical indicators',
                'fundamental/': 'Fundamental features',
                'economic/': 'Economic features',
                'sentiment/': 'Sentiment features',
                'combined/': 'Multi-source combined features'
            },
            'models/': {
                'trained/': 'Trained model artifacts',
                'checkpoints/': 'Training checkpoints',
                'metadata/': 'Model metadata'
            },
            'exports/': {
                'datasets/': 'Final datasets for analysis',
                'reports/': 'Generated reports',
                'visualizations/': 'Charts and graphs'
            },
            'cache/': 'Temporary cached data',
            'logs/': 'Data processing logs'
        },
        
        # Automation scripts
        'scripts/': {
            'data_collection/': 'Data collection automation',
            'pipeline/': 'Pipeline management',
            'backtesting/': 'Backtesting automation',
            'deployment/': 'Deployment scripts',
            'monitoring/': 'System monitoring',
            'maintenance/': 'System maintenance'
        },
        
        # Testing framework
        'tests/': {
            'unit/': {
                'data_pipeline/': 'Data pipeline unit tests',
                'models/': 'Model unit tests',
                'utils/': 'Utility unit tests'
            },
            'integration/': {
                'data_sources/': 'Data source integration tests',
                'pipelines/': 'Pipeline integration tests',
                'end_to_end/': 'Complete system tests'
            },
            'performance/': 'Performance and load tests',
            'fixtures/': 'Test data and fixtures'
        },
        
        # Documentation
        'docs/': {
            'user_guides/': 'User documentation',
            'api/': 'API documentation',
            'architecture/': 'System architecture docs',
            'data_sources/': 'Data source documentation',
            'deployment/': 'Deployment guides',
            'examples/': 'Usage examples'
        },
        
        # Web dashboard
        'dashboard/': {
            'components/': 'UI components',
            'pages/': 'Dashboard pages',
            'assets/': 'Static assets',
            'api/': 'Dashboard API endpoints'
        },
        
        # Jupyter notebooks
        'notebooks/': {
            'exploration/': 'Data exploration',
            'analysis/': 'Analysis notebooks',
            'research/': 'Research and experiments',
            'examples/': 'Usage examples'
        }
    }
    
    # Create directory structure
    print("📁 Creating organized directory structure...")
    created_dirs = 0
    
    def create_structure(base_path: Path, structure: dict, level: int = 0):
        nonlocal created_dirs
        for name, content in structure.items():
            dir_path = base_path / name
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs += 1
            
            # Create __init__.py for Python packages
            if name.endswith('/') and 'src/' in str(dir_path):
                init_file = dir_path / '__init__.py'
                if not init_file.exists():
                    init_file.touch()
            
            # Create README.md with description
            if isinstance(content, str):
                readme_file = dir_path / 'README.md'
                if not readme_file.exists():
                    with open(readme_file, 'w') as f:
                        f.write(f"# {name.rstrip('/')}\n\n{content}\n")
            
            # Recursively create subdirectories
            elif isinstance(content, dict):
                create_structure(dir_path, content, level + 1)
            
            print(f"   {'  ' * level}✅ {name}")
    
    create_structure(project_root, ideal_structure)
    
    print(f"\n📊 Created {created_dirs} directories")
    
    # Create configuration files for data source integration
    print(f"\n⚙️ Creating data source integration configurations...")
    
    # Data source registry
    data_sources_config = {
        "data_sources": {
            "yahoo_finance": {
                "type": "free",
                "rate_limit": "unlimited",
                "data_types": ["daily", "intraday", "historical"],
                "symbols_supported": "all_major_exchanges",
                "features": ["ohlcv", "dividends", "splits"],
                "collector_class": "YahooFinanceCollector",
                "enabled": True
            },
            "alpha_vantage": {
                "type": "freemium",
                "rate_limit": "500_per_day",
                "data_types": ["daily", "intraday", "fundamental"],
                "symbols_supported": "global",
                "features": ["ohlcv", "technical_indicators", "fundamental"],
                "collector_class": "AlphaVantageCollector",
                "api_key_required": True,
                "enabled": True
            },
            "polygon": {
                "type": "freemium",
                "rate_limit": "5_per_minute",
                "data_types": ["daily", "intraday", "options"],
                "symbols_supported": "us_markets",
                "features": ["ohlcv", "options", "news"],
                "collector_class": "PolygonCollector",
                "api_key_required": True,
                "enabled": True
            },
            "trading_economics": {
                "type": "premium",
                "rate_limit": "varies",
                "data_types": ["economic", "calendar", "indicators"],
                "symbols_supported": "global_economics",
                "features": ["economic_indicators", "calendar_events"],
                "collector_class": "TradingEconomicsCollector",
                "api_key_required": True,
                "enabled": False
            },
            "fred": {
                "type": "free",
                "rate_limit": "120_per_minute",
                "data_types": ["economic", "monetary"],
                "symbols_supported": "us_economic_data",
                "features": ["fed_data", "economic_indicators"],
                "collector_class": "FREDCollector",
                "api_key_required": True,
                "enabled": False
            }
        },
        "integration_priority": [
            "yahoo_finance",
            "alpha_vantage", 
            "polygon",
            "fred",
            "trading_economics"
        ],
        "data_validation": {
            "required_fields": ["symbol", "date", "close"],
            "quality_thresholds": {
                "min_records": 50,
                "max_missing_ratio": 0.05,
                "price_range_check": True
            }
        }
    }
    
    config_file = project_root / 'config' / 'data_sources' / 'registry.json'
    with open(config_file, 'w') as f:
        json.dump(data_sources_config, f, indent=2)
    
    print(f"   ✅ Data source registry: {config_file}")
    
    # Schema definitions
    schema_config = {
        "unified_schema": {
            "company_level": {
                "required_fields": [
                    "symbol", "date", "open", "high", "low", "close", "volume"
                ],
                "technical_indicators": {
                    "volume": ["volume_sma_20", "volume_ratio", "obv", "vwap"],
                    "momentum": ["rsi_14", "macd_line", "stochastic_k", "williams_r"],
                    "trend": ["sma_20", "sma_50", "ema_12", "bb_upper", "bb_lower"],
                    "volatility": ["atr_14", "historical_vol_20", "bb_width"],
                    "support_resistance": ["pivot_point", "support_1", "resistance_1"]
                },
                "derived_features": [
                    "daily_return", "forward_return_1d", "price_vs_sma20",
                    "bb_position", "price_change", "gap"
                ]
            },
            "market_level": {
                "breadth_indicators": [
                    "advance_decline_ratio", "new_highs_lows", "up_down_volume"
                ],
                "sector_rotation": [
                    "sector_relative_strength", "sector_momentum", "sector_correlation"
                ],
                "economic_context": [
                    "fed_funds_rate", "inflation_rate", "gdp_growth", "unemployment"
                ]
            }
        },
        "data_types": {
            "price_data": "float64",
            "volume_data": "int64", 
            "indicators": "float64",
            "categorical": "string",
            "dates": "datetime64[ns]"
        }
    }
    
    schema_file = project_root / 'config' / 'schemas' / 'unified_schema.json'
    with open(schema_file, 'w') as f:
        json.dump(schema_config, f, indent=2)
    
    print(f"   ✅ Unified schema: {schema_file}")
    
    # Create data source integration template
    integration_template = '''#!/usr/bin/env python3
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
'''
    
    template_file = project_root / 'src' / 'data_pipeline' / 'integrators' / 'data_source_template.py'
    with open(template_file, 'w') as f:
        f.write(integration_template)
    
    print(f"   ✅ Integration template: {template_file}")
    
    # Create organization summary
    summary = {
        "organization_date": datetime.now().isoformat(),
        "directories_created": created_dirs,
        "structure_type": "production_ready",
        "key_improvements": [
            "Separated data sources by type and processing stage",
            "Created modular collector architecture",
            "Implemented unified schema framework",
            "Added comprehensive testing structure",
            "Organized configuration management",
            "Prepared for multi-source integration"
        ],
        "ready_for_integration": [
            "New data sources via template",
            "Additional technical indicators",
            "Economic data feeds",
            "News and sentiment data",
            "Alternative data sources"
        ]
    }
    
    summary_file = project_root / 'docs' / 'architecture' / 'organization_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Codebase organization completed!")
    print(f"📊 Summary:")
    print(f"   📁 Directories created: {created_dirs}")
    print(f"   ⚙️ Configuration files: 3")
    print(f"   📋 Templates created: 1")
    print(f"   📄 Documentation: Updated")
    
    print(f"\n🚀 Ready for data source integration:")
    print(f"   1. Add new collectors using the template")
    print(f"   2. Configure data sources in registry.json")
    print(f"   3. Implement schema normalization")
    print(f"   4. Add validation rules")
    print(f"   5. Test integration with existing pipeline")
    
    print(f"\n📋 Next Steps:")
    print(f"   • Share your new data sources")
    print(f"   • I'll create specific collectors")
    print(f"   • Integrate into unified pipeline")
    print(f"   • Test with RL trading agent")

if __name__ == "__main__":
    organize_codebase()
