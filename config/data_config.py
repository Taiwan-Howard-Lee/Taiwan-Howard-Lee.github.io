"""
Data Configuration - Configuration settings for data collection and processing
"""

from typing import Dict, List
import os

# Main ticker configuration
TICKER_CONFIG = {
    'primary_ticker': 'AAPL',
    'company_name': 'Apple Inc.',
    'sector': 'Technology',
    'exchange': 'NASDAQ'
}

# Market context tickers
MARKET_TICKERS = {
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ Trust',
    'VIX': 'CBOE Volatility Index',
    'DXY': 'US Dollar Index',
    '^TNX': '10-Year Treasury Note Yield',
    '^VIX': 'CBOE Volatility Index'
}

# Sector ETFs for context
SECTOR_TICKERS = {
    'XLK': 'Technology Select Sector SPDR Fund',
    'XLF': 'Financial Select Sector SPDR Fund',
    'XLE': 'Energy Select Sector SPDR Fund',
    'XLV': 'Health Care Select Sector SPDR Fund',
    'XLI': 'Industrial Select Sector SPDR Fund'
}

# Data collection settings
DATA_CONFIG = {
    'default_period': '5y',
    'intraday_period': '1mo',
    'intervals': ['1d', '1h', '5m', '1m'],
    'max_retries': 3,
    'retry_delay': 1.0,  # seconds
    'rate_limit_delay': 0.1,  # seconds between API calls
}

# File paths
PATHS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'external_data': 'data/external',
    'influxdb_data': 'data/influxdb',
    'models': 'models',
    'logs': 'logs',
    'config': 'config'
}

# Ensure directories exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# InfluxDB configuration
INFLUXDB_CONFIG = {
    'url': 'http://localhost:8086',
    'token': '',  # Set this in environment variable INFLUXDB_TOKEN
    'org': 'apple_trading',
    'bucket': 'financial_data',
    'measurement_price': 'stock_price',
    'measurement_features': 'features',
    'measurement_predictions': 'predictions'
}

# Economic indicators from FRED
FRED_INDICATORS = {
    'GDP': 'GDP',
    'UNRATE': 'Unemployment Rate',
    'CPIAUCSL': 'Consumer Price Index',
    'FEDFUNDS': 'Federal Funds Rate',
    'DGS10': '10-Year Treasury Constant Maturity Rate',
    'DGS2': '2-Year Treasury Constant Maturity Rate',
    'DEXUSEU': 'US / Euro Foreign Exchange Rate',
    'DEXJPUS': 'Japan / US Foreign Exchange Rate'
}

# Trading Economics API Configuration
TRADING_ECONOMICS_CONFIG = {
    'api_key': '50439e96184c4b1:7008dwvh5w03yxa',
    'base_url': 'https://api.tradingeconomics.com',
    'available_countries': ['mexico', 'sweden', 'new-zealand', 'thailand'],
    'default_country': 'mexico',
    'update_frequency': 3600,  # seconds (1 hour)
    'cache_duration': 1800     # seconds (30 minutes)
}

# Polygon.io API Configuration
POLYGON_CONFIG = {
    'api_key': 'YpK43xQz3xo0hRVS2l6u8lqwJPSn_Tgf',
    'base_url': 'https://api.polygon.io',
    'rate_limit': 5,           # requests per minute (free tier)
    'request_interval': 12,    # seconds between requests
    'retry_attempts': 3,
    'timeout': 30
}

# Continuous Collection Configuration
CONTINUOUS_COLLECTION_CONFIG = {
    'data_directory': 'data/continuous_collection',
    'session_save_interval': 10,  # save every N requests
    'default_collection_hours': 4,
    'max_collection_hours': 24,
    'priority_tickers': ['AAPL', 'SPY', 'QQQ', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
    'collection_strategy': {
        'core_apple_data': 0.4,      # 40% of requests
        'historical_data': 0.3,      # 30% of requests
        'related_tickers': 0.2,      # 20% of requests
        'general_news': 0.1          # 10% of requests
    }
}

# Technical indicators configuration
TECHNICAL_INDICATORS_CONFIG = {
    'moving_average_periods': [5, 10, 20, 50, 100, 200],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'atr_period': 14,
    'stoch_k_period': 14,
    'stoch_d_period': 3,
    'williams_r_period': 14,
    'cci_period': 14,
    'adx_period': 14,
    'aroon_period': 14
}

# Feature engineering settings
FEATURE_CONFIG = {
    'lookback_periods': [5, 10, 20, 50],
    'return_periods': [1, 5, 10, 20],
    'volatility_windows': [10, 20, 50],
    'volume_windows': [10, 20, 50],
    'correlation_windows': [20, 50, 100]
}

# Model training configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'scoring_metric': 'accuracy',
    'class_weight': 'balanced'
}

# Target variable configuration
TARGET_CONFIG = {
    'horizons': [1, 3, 5, 10],  # days ahead to predict
    'thresholds': [0.01, 0.02, 0.03],  # return thresholds for classification
    'default_horizon': 5,
    'default_threshold': 0.02
}

# Risk management settings
RISK_CONFIG = {
    'var_confidence_levels': [0.95, 0.99],
    'max_position_size': 0.1,  # 10% of portfolio
    'stop_loss': 0.05,  # 5% stop loss
    'max_drawdown': 0.15,  # 15% maximum drawdown
    'risk_free_rate': 0.02  # 2% annual risk-free rate
}

# Backtesting configuration
BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'transaction_cost': 0.001,  # 0.1% per trade
    'slippage': 0.0005,  # 0.05% slippage
    'benchmark': 'SPY',
    'rebalance_frequency': 'daily'
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'refresh_interval': 300,  # 5 minutes
    'cache_ttl': 300,  # 5 minutes
    'max_data_points': 1000,
    'default_chart_height': 600
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Environment variables
def get_env_var(var_name: str, default_value: str = "") -> str:
    """Get environment variable with default value."""
    return os.getenv(var_name, default_value)

# Update InfluxDB token from environment
INFLUXDB_CONFIG['token'] = get_env_var('INFLUXDB_TOKEN', '')

# Alpha Vantage API key (if using)
ALPHA_VANTAGE_API_KEY = get_env_var('ALPHA_VANTAGE_API_KEY', '')

# News API keys (if using)
NEWS_API_CONFIG = {
    'google_news_api_key': get_env_var('GOOGLE_NEWS_API_KEY', ''),
    'news_api_key': get_env_var('NEWS_API_KEY', ''),
    'reddit_client_id': get_env_var('REDDIT_CLIENT_ID', ''),
    'reddit_client_secret': get_env_var('REDDIT_CLIENT_SECRET', ''),
    'reddit_user_agent': get_env_var('REDDIT_USER_AGENT', 'AppleMLTrading/1.0')
}

# Validation functions
def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check required paths exist
    for name, path in PATHS.items():
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create {name} directory {path}: {e}")
    
    # Validate numeric settings
    if DATA_CONFIG['max_retries'] < 1:
        errors.append("max_retries must be at least 1")
    
    if RISK_CONFIG['max_position_size'] <= 0 or RISK_CONFIG['max_position_size'] > 1:
        errors.append("max_position_size must be between 0 and 1")
    
    if BACKTEST_CONFIG['initial_capital'] <= 0:
        errors.append("initial_capital must be positive")
    
    return errors

# Run validation on import
_validation_errors = validate_config()
if _validation_errors:
    print("Configuration validation errors:")
    for error in _validation_errors:
        print(f"  - {error}")
