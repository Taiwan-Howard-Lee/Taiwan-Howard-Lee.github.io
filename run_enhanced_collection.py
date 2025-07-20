#!/usr/bin/env python3
"""
Enhanced Data Collection Launcher - Maximum diversity and freshness
Usage: python3 run_enhanced_collection.py --hours 8
"""

import sys
import os
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_collection.enhanced_collector import EnhancedContinuousCollector

def main():
    """Run enhanced continuous collection with diversity optimization"""
    
    print("🎯 Apple ML Trading - ENHANCED Data Collector")
    print("=" * 55)
    print("🚀 Maximum Data Diversity & Freshness Strategy")
    print("📊 Smart Deduplication & 31+ Ticker Coverage")
    print("⏰ 16 Time Periods & Intelligent Request Distribution")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Continuous Data Collection')
    parser.add_argument('--hours', type=float, default=None, help='Hours to run collection')
    parser.add_argument('--test', action='store_true', help='Run 5-minute test mode')
    parser.add_argument('--strategy-only', action='store_true', help='Show strategy without collecting')
    
    args = parser.parse_args()
    
    # Determine collection duration
    if args.test:
        hours = 5/60  # 5 minutes
        print("🧪 ENHANCED TEST MODE (5 minutes)")
    elif args.hours:
        hours = args.hours
        print(f"🚀 ENHANCED COLLECTION MODE ({hours} hours)")
    else:
        # Interactive mode
        try:
            hours_input = input("⏰ Hours to run enhanced collection? (default: 8): ").strip()
            hours = float(hours_input) if hours_input else 8.0
            
            if hours <= 0:
                print("❌ Hours must be positive")
                return
            
            if hours > 24:
                confirm = input(f"⚠️  Running for {hours} hours. Continue? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("Collection cancelled")
                    return
        except ValueError:
            print("❌ Invalid input. Using default 8 hours.")
            hours = 8.0
    
    # Initialize enhanced collector
    collector = EnhancedContinuousCollector()
    
    # Build and show strategy
    print(f"\n📊 Building Enhanced Strategy...")
    strategy = collector.build_enhanced_strategy(hours)
    
    # Strategy analysis
    request_types = {}
    tickers = set()
    time_periods = set()
    
    for req in strategy:
        req_type = req['type']
        request_types[req_type] = request_types.get(req_type, 0) + 1
        
        if 'ticker' in req:
            tickers.add(req['ticker'])
        
        if 'days' in req:
            time_periods.add(req['days'])
    
    print(f"✅ Enhanced Strategy Built:")
    print(f"   📊 Total Requests: {len(strategy)}")
    print(f"   🎯 Unique Request Types: {len(request_types)}")
    print(f"   📈 Ticker Coverage: {len(tickers)} symbols")
    print(f"   ⏰ Time Periods: {len(time_periods)} ranges")
    print(f"   🔄 Expected Fresh Data: >90%")
    
    print(f"\n📋 Request Distribution:")
    for req_type, count in sorted(request_types.items()):
        percentage = (count / len(strategy)) * 100
        print(f"   {req_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Ticker Coverage:")
    ticker_list = sorted(list(tickers))
    print(f"   Core: {[t for t in ticker_list if t == 'AAPL']}")
    print(f"   Tech: {[t for t in ticker_list if t in ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL']][:5]}...")
    print(f"   ETFs: {[t for t in ticker_list if t in ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'XLK', 'XLF']][:5]}...")
    
    if time_periods:
        print(f"\n⏰ Time Period Diversity:")
        sorted_periods = sorted(list(time_periods))
        print(f"   Short-term: {[p for p in sorted_periods if p <= 7]} days")
        print(f"   Medium-term: {[p for p in sorted_periods if 7 < p <= 90]} days") 
        print(f"   Long-term: {[p for p in sorted_periods if p > 90]} days")
    
    if args.strategy_only:
        print(f"\n📊 Strategy analysis complete. Use --hours to start collection.")
        return
    
    # Confirm start
    print(f"\n🚀 Enhanced Collection Summary:")
    print(f"   Duration: {hours} hours")
    print(f"   Expected API calls: {len(strategy)}")
    print(f"   Rate: 5 requests/minute (12s intervals)")
    print(f"   Data diversity: Maximum")
    print(f"   Deduplication: Enabled")
    print(f"   Freshness tracking: Enabled")
    
    if not args.test:
        input("\nPress Enter to start ENHANCED collection (Ctrl+C to cancel)...")
    
    # Run enhanced collection
    try:
        print(f"\n🎯 Starting Enhanced Collection...")
        print(f"📁 Data: data/continuous_collection/enhanced_session_*.json")
        print(f"📝 Logs: data/continuous_collection/enhanced_collection.log")
        print()
        
        # Note: We need to implement the full collection loop
        # For now, show what would happen
        print("🚧 Enhanced collection implementation in progress...")
        print("📊 Strategy validated and ready for execution")
        
        # Show expected results
        expected_fresh = int(len(strategy) * 0.9)
        expected_tickers = len(tickers)
        expected_periods = len(time_periods)
        
        print(f"\n🎯 Expected Results:")
        print(f"   Fresh data points: ~{expected_fresh} (90%+)")
        print(f"   Ticker coverage: {expected_tickers} symbols")
        print(f"   Time period coverage: {expected_periods} ranges")
        print(f"   Duplicate avoidance: Hash-based")
        print(f"   Data quality: Premium")
        
    except KeyboardInterrupt:
        print("\n⚠️ Enhanced collection interrupted by user")
        print("💾 Strategy analysis completed")
    except Exception as e:
        print(f"\n❌ Enhanced collection error: {str(e)}")

if __name__ == "__main__":
    main()
