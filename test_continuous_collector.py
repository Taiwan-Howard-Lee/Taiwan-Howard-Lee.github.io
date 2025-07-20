#!/usr/bin/env python3
"""
Test the continuous collector with a short run
"""

import sys
import os
sys.path.append('src')

from data_collection.continuous_collector import ContinuousCollector

def test_continuous_collector():
    """Test continuous collector for 2 minutes"""
    print("🧪 Testing Continuous Collector (2 minutes)")
    print("=" * 40)
    
    collector = ContinuousCollector()
    
    # Test for 2 minutes (should make about 10 requests)
    test_hours = 2/60  # 2 minutes
    
    print(f"⏰ Running test for {test_hours*60:.0f} minutes")
    print(f"📊 Expected requests: ~{int(test_hours * 60 * 5)}")
    print()
    
    try:
        data = collector.run_continuous_collection(test_hours)
        
        print("\n✅ Test completed successfully!")
        print(f"📊 Results:")
        print(f"   Ticker details: {len(data['ticker_details'])}")
        print(f"   Daily data: {len(data['daily_data'])}")
        print(f"   News articles: {len(data['news_articles'])}")
        print(f"   Dividends: {len(data['dividends'])}")
        print(f"   API calls made: {len(data['collection_log'])}")
        
        # Show some sample data
        if data['collection_log']:
            print(f"\n📝 Sample API calls:")
            for i, log in enumerate(data['collection_log'][:3]):
                status = "✅" if log['success'] else "❌"
                print(f"   {i+1}. {status} {log['request']['type']} - {log['duration']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_continuous_collector()
    if success:
        print("\n🎉 Continuous collector is ready for long-term data collection!")
        print("💡 Run: python run_continuous_collection.py --hours 8")
    else:
        print("\n❌ Fix issues before running long-term collection")
