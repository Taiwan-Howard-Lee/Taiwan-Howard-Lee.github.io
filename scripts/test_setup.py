#!/usr/bin/env python3
"""
Test Setup Script - Verify that the project structure and basic functionality work
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from data_collection.apple_collector import AppleDataCollector
        print("✅ AppleDataCollector imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AppleDataCollector: {e}")
        return False
    
    try:
        from feature_engineering.technical_indicators import TechnicalIndicators
        print("✅ TechnicalIndicators imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TechnicalIndicators: {e}")
        return False
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        import config.data_config as config
        print("✅ Configuration imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import configuration: {e}")
        return False
    
    return True


def test_data_collection():
    """Test basic data collection functionality."""
    print("\n📊 Testing data collection...")
    
    try:
        from data_collection.apple_collector import AppleDataCollector
        
        collector = AppleDataCollector()
        print("✅ AppleDataCollector initialized")
        
        # Test fetching small amount of data
        data = collector.fetch_daily_data(period="5d")
        
        if data is not None and not data.empty:
            print(f"✅ Successfully fetched {len(data)} days of data")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
            return True
        else:
            print("❌ No data returned from fetch_daily_data")
            return False
            
    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        return False


def test_technical_indicators():
    """Test technical indicators calculation."""
    print("\n📈 Testing technical indicators...")
    
    try:
        from data_collection.apple_collector import AppleDataCollector
        from feature_engineering.technical_indicators import TechnicalIndicators
        
        # Get some sample data
        collector = AppleDataCollector()
        data = collector.fetch_daily_data(period="3mo")
        
        if data is None or data.empty:
            print("❌ No data available for technical indicators test")
            return False
        
        # Calculate indicators
        indicators = TechnicalIndicators(data)
        print("✅ TechnicalIndicators initialized")
        
        # Test moving averages
        ma_indicators = indicators.moving_averages([20, 50])
        if not ma_indicators.empty:
            print(f"✅ Moving averages calculated: {list(ma_indicators.columns)}")
        else:
            print("❌ Moving averages calculation failed")
            return False
        
        # Test momentum indicators
        momentum_indicators = indicators.momentum_indicators()
        if not momentum_indicators.empty:
            print(f"✅ Momentum indicators calculated: {list(momentum_indicators.columns)}")
        else:
            print("❌ Momentum indicators calculation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False


def test_directory_structure():
    """Test that all required directories exist."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed', 
        'data/external',
        'data/influxdb',
        'src/data_collection',
        'src/feature_engineering',
        'src/models/traditional',
        'src/models/deep_learning',
        'src/models/ensemble',
        'src/backtesting',
        'src/utils',
        'src/risk_metrics',
        'dashboard/pages',
        'dashboard/components',
        'dashboard/assets',
        'notebooks',
        'config',
        'tests',
        'logs',
        'scripts'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print(f"✅ All {len(required_dirs)} required directories exist")
        return True


def test_dependencies():
    """Test that key dependencies are available."""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('yfinance', 'yf'),
        ('plotly.graph_objects', 'go'),
        ('streamlit', 'st')
    ]
    
    optional_packages = [
        ('talib', None),
        ('pandas_ta', 'ta'),
        ('influxdb_client', None),
        ('sklearn', None),
        ('textblob', None)
    ]
    
    # Test required packages
    missing_required = []
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package} available")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} not available")
    
    # Test optional packages
    missing_optional = []
    for package, alias in optional_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package} available (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️  {package} not available (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {missing_optional}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    return True


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")

    try:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        import config.data_config as config
        
        # Test key configuration values
        assert config.TICKER_CONFIG['primary_ticker'] == 'AAPL'
        assert len(config.MARKET_TICKERS) > 0
        assert len(config.TECHNICAL_INDICATORS_CONFIG) > 0
        
        print("✅ Configuration loaded successfully")
        print(f"   Primary ticker: {config.TICKER_CONFIG['primary_ticker']}")
        print(f"   Market tickers: {len(config.MARKET_TICKERS)}")
        print(f"   Data paths configured: {len(config.PATHS)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Apple ML Trading System Setup Test")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("Imports", test_imports),
        ("Data Collection", test_data_collection),
        ("Technical Indicators", test_technical_indicators)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Install any missing optional dependencies")
        print("2. Run the dashboard: streamlit run dashboard/app.py")
        print("3. Start implementing the features from TEST_TIMELINE.md")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
