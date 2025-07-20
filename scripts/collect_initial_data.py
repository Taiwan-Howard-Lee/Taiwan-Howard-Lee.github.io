#!/usr/bin/env python3
"""
Initial Data Collection Script - Collect and save initial dataset for Apple ML Trading System
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

def collect_apple_data():
    """Collect Apple stock data."""
    print("📊 Collecting Apple (AAPL) stock data...")
    
    try:
        from data_collection.apple_collector import AppleDataCollector
        
        collector = AppleDataCollector()
        
        # Collect different timeframes
        datasets = {
            'daily_5y': collector.fetch_daily_data(period="5y"),
            'daily_1y': collector.fetch_daily_data(period="1y"),
            'intraday_1mo': collector.fetch_intraday_data(period="1mo", interval="1h")
        }
        
        for name, data in datasets.items():
            if data is not None and not data.empty:
                filename = f"aapl_{name}_{datetime.now().strftime('%Y%m%d')}.csv"
                if collector.save_to_csv(data, filename):
                    print(f"✅ Saved {name}: {len(data)} records to {filename}")
                else:
                    print(f"❌ Failed to save {name}")
            else:
                print(f"❌ No data collected for {name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error collecting Apple data: {e}")
        return False


def collect_market_data():
    """Collect market context data."""
    print("\n📈 Collecting market context data...")
    
    try:
        from data_collection.apple_collector import AppleDataCollector
        
        collector = AppleDataCollector()
        
        # Collect market data
        market_data = collector.fetch_market_data(period="2y")
        
        if market_data:
            for ticker, data in market_data.items():
                if data is not None and not data.empty:
                    filename = f"{ticker.lower()}_2y_{datetime.now().strftime('%Y%m%d')}.csv"
                    if collector.save_to_csv(data, filename):
                        print(f"✅ Saved {ticker}: {len(data)} records to {filename}")
                    else:
                        print(f"❌ Failed to save {ticker}")
                else:
                    print(f"❌ No data collected for {ticker}")
        
        # Collect sector ETFs
        sector_data = collector.fetch_sector_etfs(period="2y")
        
        if sector_data:
            for ticker, data in sector_data.items():
                if data is not None and not data.empty:
                    filename = f"{ticker.lower()}_sector_2y_{datetime.now().strftime('%Y%m%d')}.csv"
                    if collector.save_to_csv(data, filename):
                        print(f"✅ Saved {ticker}: {len(data)} records to {filename}")
                    else:
                        print(f"❌ Failed to save {ticker}")
                else:
                    print(f"❌ No data collected for {ticker}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error collecting market data: {e}")
        return False


def calculate_initial_indicators():
    """Calculate technical indicators for the collected data."""
    print("\n📊 Calculating technical indicators...")
    
    try:
        from data_collection.apple_collector import AppleDataCollector
        from feature_engineering.technical_indicators import TechnicalIndicators
        
        # Load recent Apple data
        collector = AppleDataCollector()
        data = collector.fetch_daily_data(period="2y")
        
        if data is None or data.empty:
            print("❌ No Apple data available for indicator calculation")
            return False
        
        # Calculate indicators
        indicators = TechnicalIndicators(data)
        all_indicators = indicators.calculate_all_indicators()
        
        if not all_indicators.empty:
            # Combine price data with indicators
            combined_data = pd.concat([data, all_indicators], axis=1)
            
            # Save combined dataset
            filename = f"aapl_with_indicators_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join("data", "processed", filename)
            
            # Ensure processed directory exists
            os.makedirs("data/processed", exist_ok=True)
            
            combined_data.to_csv(filepath)
            print(f"✅ Saved combined dataset with {len(all_indicators.columns)} indicators to {filename}")
            print(f"   Total features: {len(combined_data.columns)}")
            print(f"   Date range: {combined_data.index[0].date()} to {combined_data.index[-1].date()}")
            
            return True
        else:
            print("❌ No indicators calculated")
            return False
            
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return False


def get_company_info():
    """Get and save Apple company information."""
    print("\n🏢 Collecting company information...")
    
    try:
        from data_collection.apple_collector import AppleDataCollector
        
        collector = AppleDataCollector()
        company_info = collector.get_company_info()
        
        if company_info:
            # Save company info as JSON
            import json
            
            filename = f"aapl_company_info_{datetime.now().strftime('%Y%m%d')}.json"
            filepath = os.path.join("data", "external", filename)
            
            # Ensure external directory exists
            os.makedirs("data/external", exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(company_info, f, indent=2, default=str)
            
            print(f"✅ Saved company information to {filename}")
            print(f"   Company: {company_info.get('longName', 'N/A')}")
            print(f"   Sector: {company_info.get('sector', 'N/A')}")
            print(f"   Market Cap: ${company_info.get('marketCap', 'N/A'):,}")
            print(f"   Employees: {company_info.get('fullTimeEmployees', 'N/A'):,}")
            
            return True
        else:
            print("❌ No company information retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Error collecting company info: {e}")
        return False


def create_data_summary():
    """Create a summary of collected data."""
    print("\n📋 Creating data summary...")
    
    try:
        data_dir = "data/raw"
        if not os.path.exists(data_dir):
            print("❌ No raw data directory found")
            return False
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ No CSV files found in raw data directory")
            return False
        
        summary = []
        total_records = 0
        
        for filename in csv_files:
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                records = len(df)
                total_records += records
                
                summary.append({
                    'filename': filename,
                    'records': records,
                    'columns': len(df.columns),
                    'start_date': df.index.min().strftime('%Y-%m-%d') if not df.empty else 'N/A',
                    'end_date': df.index.max().strftime('%Y-%m-%d') if not df.empty else 'N/A',
                    'size_mb': round(os.path.getsize(filepath) / (1024*1024), 2)
                })
                
            except Exception as e:
                print(f"⚠️  Could not read {filename}: {e}")
        
        if summary:
            print(f"✅ Data collection summary:")
            print(f"   Total files: {len(summary)}")
            print(f"   Total records: {total_records:,}")
            
            for item in summary:
                print(f"   📄 {item['filename']}: {item['records']:,} records, "
                      f"{item['columns']} columns, {item['size_mb']} MB")
            
            return True
        else:
            print("❌ No valid data files found")
            return False
            
    except Exception as e:
        print(f"❌ Error creating data summary: {e}")
        return False


def main():
    """Main data collection workflow."""
    print("🚀 Starting Initial Data Collection for Apple ML Trading System")
    print("=" * 70)
    
    tasks = [
        ("Apple Stock Data", collect_apple_data),
        ("Market Context Data", collect_market_data),
        ("Company Information", get_company_info),
        ("Technical Indicators", calculate_initial_indicators),
        ("Data Summary", create_data_summary)
    ]
    
    results = {}
    
    for task_name, task_func in tasks:
        try:
            results[task_name] = task_func()
        except Exception as e:
            print(f"❌ {task_name} failed: {e}")
            results[task_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DATA COLLECTION SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for task_name, result in results.items():
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{status} {task_name}")
    
    print(f"\nOverall: {passed}/{total} tasks completed successfully")
    
    if passed == total:
        print("\n🎉 Initial data collection completed successfully!")
        print("\nNext steps:")
        print("1. Review the collected data in the data/ directory")
        print("2. Run the dashboard: streamlit run dashboard/app.py")
        print("3. Continue with feature engineering from TEST_TIMELINE.md")
    else:
        print(f"\n⚠️  {total - passed} tasks failed. Check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
