#!/usr/bin/env python3
"""
Test the Polygon.io Collector
"""

import sys
import os
sys.path.append('src')

from data_collection.polygon_collector import PolygonCollector
import json
import pandas as pd

def test_polygon_collector():
    """Test the Polygon.io collector"""
    print("🚀 Testing Polygon.io Collector")
    print("=" * 50)
    
    # Initialize collector
    collector = PolygonCollector()
    
    # Test ticker details
    print("\n🍎 Testing Ticker Details...")
    ticker_details = collector.get_ticker_details('AAPL')
    if ticker_details:
        print(f"✅ Ticker Details Retrieved:")
        print(f"   Name: {ticker_details.get('name')}")
        print(f"   Market: {ticker_details.get('market')}")
        print(f"   Type: {ticker_details.get('type')}")
        print(f"   Currency: {ticker_details.get('currency_name')}")
        print(f"   Active: {ticker_details.get('active')}")
    else:
        print("❌ Failed to retrieve ticker details")
    
    # Test previous close
    print("\n📊 Testing Previous Close...")
    prev_close = collector.get_previous_close('AAPL')
    if prev_close:
        print(f"✅ Previous Close Data:")
        print(f"   Date: {prev_close.get('date')}")
        print(f"   Open: ${prev_close.get('o'):.2f}")
        print(f"   High: ${prev_close.get('h'):.2f}")
        print(f"   Low: ${prev_close.get('l'):.2f}")
        print(f"   Close: ${prev_close.get('c'):.2f}")
        print(f"   Volume: {prev_close.get('v'):,.0f}")
    else:
        print("❌ Failed to retrieve previous close")
    
    # Test aggregates
    print("\n📈 Testing Historical Aggregates...")
    aggregates = collector.get_aggregates('AAPL', days=7)
    if aggregates is not None:
        print(f"✅ Retrieved {len(aggregates)} days of historical data")
        print("\nLast 3 days:")
        for date, row in aggregates.tail(3).iterrows():
            print(f"   {date.strftime('%Y-%m-%d')}: O=${row['Open']:.2f} H=${row['High']:.2f} L=${row['Low']:.2f} C=${row['Close']:.2f} V={row['Volume']:,.0f}")
    else:
        print("❌ Failed to retrieve aggregates")
    
    # Test news
    print("\n📰 Testing News Data...")
    news = collector.get_news('AAPL', limit=3)
    if news:
        print(f"✅ Retrieved {len(news)} news articles")
        for i, article in enumerate(news[:3], 1):
            print(f"   {i}. {article.get('published_date')}: {article.get('title_short')}")
    else:
        print("❌ Failed to retrieve news")
    
    # Test dividends
    print("\n💰 Testing Dividends...")
    dividends = collector.get_dividends('AAPL')
    if dividends:
        print(f"✅ Retrieved {len(dividends)} dividend records")
        for div in dividends[:2]:  # Show first 2
            print(f"   ${div.get('cash_amount')} on {div.get('ex_dividend_date')} (Pay: {div.get('pay_date')})")
    else:
        print("❌ Failed to retrieve dividends")
    
    # Test market summary
    print("\n🌍 Testing Market Summary...")
    summary = collector.get_market_summary('AAPL')
    if summary:
        print("✅ Market summary generated successfully")
        print(f"📊 Ticker: {summary['ticker']}")
        print(f"🏢 Company: {summary['ticker_details'].get('name', 'N/A')}")
        
        current = summary.get('current_data', {})
        if current:
            print(f"💰 Current Price: ${current.get('close', 0):.2f}")
            print(f"📊 Volume: {current.get('volume', 0):,.0f}")
            print(f"📅 Date: {current.get('date', 'N/A')}")
        
        historical_data = summary.get('historical_data')
        if historical_data:
            print(f"📈 Historical Days: {len(historical_data)}")
        else:
            print(f"📈 Historical Days: 0 (data unavailable)")
        print(f"📰 News Articles: {len(summary.get('news', []))}")
        print(f"💰 Dividend Records: {len(summary.get('dividends', []))}")
    else:
        print("❌ Failed to generate market summary")
    
    # Test saving data
    print("\n💾 Testing Data Save...")
    filename = collector.save_market_data('AAPL')
    if filename:
        print(f"✅ Data saved to {filename}")
        
        # Verify file exists and show size
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"📁 File size: {size:,} bytes")
            
            # Show sample of saved data
            with open(filename, 'r') as f:
                data = json.load(f)
                print(f"📊 Saved data sections: {list(data.keys())}")
        else:
            print("❌ Saved file not found")
    else:
        print("❌ Failed to save data")
    
    print("\n" + "=" * 50)
    print("🎉 Polygon.io Collector testing completed!")
    
    return summary

def compare_data_sources():
    """Compare Polygon.io with Trading Economics data"""
    print("\n🔄 Comparing Data Sources")
    print("=" * 30)
    
    # Get Polygon.io data
    polygon_collector = PolygonCollector()
    polygon_data = polygon_collector.get_previous_close('AAPL')
    
    if polygon_data:
        print(f"📊 Polygon.io (Real US Market Data):")
        print(f"   AAPL Close: ${polygon_data.get('c'):.2f}")
        print(f"   AAPL Volume: {polygon_data.get('v'):,.0f}")
        print(f"   Date: {polygon_data.get('date')}")
    
    # Compare with dashboard mock data
    print(f"\n📊 Current Dashboard (Mock Data):")
    print(f"   AAPL Close: $227.20")
    print(f"   AAPL Volume: 28,567,234")
    print(f"   Date: 2024-07-20")
    
    if polygon_data:
        price_diff = abs(polygon_data.get('c', 0) - 227.20)
        print(f"\n🔍 Analysis:")
        print(f"   Price Difference: ${price_diff:.2f}")
        print(f"   Recommendation: Update dashboard with Polygon.io real data")
    
    return polygon_data

if __name__ == "__main__":
    # Test collector
    summary = test_polygon_collector()
    
    # Compare data sources
    comparison = compare_data_sources()
