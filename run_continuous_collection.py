#!/usr/bin/env python3
"""
Launcher for continuous data collection
Usage: python3 run_continuous_collection.py --hours 4
"""

import sys
import os

# Add src to path
sys.path.append('src')

from data_collection.continuous_collector import ContinuousCollector

def main():
    """Run continuous collection with user-friendly interface"""
    
    print("🍎 Apple ML Trading - Continuous Data Collector")
    print("=" * 50)
    print("📊 Maximizing Polygon.io API usage at 5 requests/minute")
    print("🌍 Including Trading Economics global data")
    print()
    
    # Get user input
    try:
        hours = input("⏰ How many hours to run collection? (default: 4): ").strip()
        hours = float(hours) if hours else 4.0
        
        if hours <= 0:
            print("❌ Hours must be positive")
            return
        
        if hours > 24:
            confirm = input(f"⚠️  Running for {hours} hours. Continue? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Collection cancelled")
                return
        
    except ValueError:
        print("❌ Invalid input. Using default 4 hours.")
        hours = 4.0
    
    print(f"\n🚀 Starting {hours} hour collection session...")
    print(f"📈 Expected requests: {int(hours * 60 * 5)}")
    print(f"💾 Data will be saved to: data/continuous_collection/")
    print(f"📝 Logs will be written to: data/continuous_collection/continuous_collection.log")
    print()
    
    # Confirm start
    input("Press Enter to start collection (Ctrl+C to cancel)...")
    
    # Run collection
    try:
        collector = ContinuousCollector()
        data = collector.run_continuous_collection(hours)
        
        print("\n🎉 Collection completed successfully!")
        print(f"📊 Final Summary:")
        print(f"   📈 Ticker details: {len(data['ticker_details'])}")
        print(f"   📊 Daily data points: {len(data['daily_data'])}")
        print(f"   📰 News articles: {len(data['news_articles'])}")
        print(f"   💰 Dividends: {len(data['dividends'])}")
        print(f"   🌍 Economic indicators: {len(data['economic_indicators'])}")
        print(f"   💱 Currency pairs: {len(data['currency_data'])}")
        print(f"   📝 Total API calls: {len(data['collection_log'])}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Collection interrupted by user")
        print("💾 Partial data has been saved")
    except Exception as e:
        print(f"\n❌ Collection failed: {str(e)}")

if __name__ == "__main__":
    main()
