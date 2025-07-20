#!/usr/bin/env python3
"""
Test the Trading Economics Collector
"""

import sys
import os
sys.path.append('src')

from data_collection.trading_economics_collector import TradingEconomicsCollector
import json

def test_collector():
    """Test the Trading Economics collector"""
    print("🚀 Testing Trading Economics Collector")
    print("=" * 50)
    
    # Initialize collector
    collector = TradingEconomicsCollector()
    
    # Test economic indicators
    print("\n📊 Testing Economic Indicators...")
    econ_data = collector.get_economic_indicators('mexico')
    if econ_data is not None:
        print(f"✅ Retrieved {len(econ_data)} economic indicators")
        print("\nTop 5 indicators:")
        for _, row in econ_data.head().iterrows():
            print(f"  📈 {row['Category']}: {row.get('Unit', '')} (Updated: {row['LatestValueDate']})")
    else:
        print("❌ Failed to retrieve economic indicators")
    
    # Test currency data
    print("\n💱 Testing Currency Data...")
    currency_data = collector.get_currency_data()
    if currency_data is not None:
        print(f"✅ Retrieved {len(currency_data)} currency pairs")
        print("\nCurrency pairs:")
        for _, row in currency_data.iterrows():
            symbol = row['Symbol']
            rate = row['Last']
            change = row['DailyPercentualChange']
            print(f"  💱 {symbol}: {rate} ({change:+.2f}%)" if pd.notna(change) else f"  💱 {symbol}: {rate}")
    else:
        print("❌ Failed to retrieve currency data")
    
    # Test commodity data
    print("\n🛢️ Testing Commodity Data...")
    commodity_data = collector.get_commodity_data()
    if commodity_data is not None:
        print(f"✅ Retrieved {len(commodity_data)} commodities")
        if len(commodity_data) > 0:
            print("\nCommodities:")
            for _, row in commodity_data.head().iterrows():
                symbol = row.get('Symbol', 'Unknown')
                price = row['Last']
                change = row['DailyPercentualChange']
                print(f"  🛢️ {symbol}: {price} ({change:+.2f}%)" if pd.notna(change) else f"  🛢️ {symbol}: {price}")
    else:
        print("❌ Failed to retrieve commodity data")
    
    # Test market summary
    print("\n🌍 Testing Market Summary...")
    summary = collector.get_market_summary()
    if summary:
        print("✅ Market summary generated successfully")
        print(f"📊 Economic indicators: {len(summary['economic_indicators'])}")
        print(f"💱 Currency pairs: {len(summary['currencies'])}")
        print(f"🛢️ Commodities: {len(summary['commodities'])}")
        print(f"📈 Market sentiment: {summary['market_sentiment']}")
        
        # Show sample data
        print("\nSample economic indicators:")
        for key, value in list(summary['economic_indicators'].items())[:3]:
            print(f"  📊 {key}: {value['ticker']} ({value['unit']})")
    else:
        print("❌ Failed to generate market summary")
    
    # Test saving data
    print("\n💾 Testing Data Save...")
    filename = collector.save_market_data()
    if filename:
        print(f"✅ Data saved to {filename}")
        
        # Verify file exists and show size
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"📁 File size: {size} bytes")
            
            # Show sample of saved data
            with open(filename, 'r') as f:
                data = json.load(f)
                print(f"📊 Saved data contains {len(data)} sections")
        else:
            print("❌ Saved file not found")
    else:
        print("❌ Failed to save data")
    
    print("\n" + "=" * 50)
    print("🎉 Trading Economics Collector testing completed!")
    
    return summary

if __name__ == "__main__":
    import pandas as pd
    results = test_collector()
