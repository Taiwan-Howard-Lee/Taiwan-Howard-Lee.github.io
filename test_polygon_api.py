#!/usr/bin/env python3
"""
Test Polygon.io API integration for Apple ML Trading Dashboard
"""

import requests
import json
from datetime import datetime, timedelta

# API Configuration
API_KEY = 'YpK43xQz3xo0hRVS2l6u8lqwJPSn_Tgf'
BASE_URL = 'https://api.polygon.io'

def test_basic_connection():
    """Test basic API connection with dividends endpoint"""
    print("🔍 Testing Polygon.io API connection...")
    
    try:
        url = f'{BASE_URL}/v3/reference/dividends'
        headers = {'Authorization': f'Bearer {API_KEY}'}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Connection successful!")
            print(f"📊 Status: {data.get('status')}")
            print(f"📋 Results count: {len(data.get('results', []))}")
            
            # Show sample dividend data
            if data.get('results'):
                sample = data['results'][0]
                print(f"📈 Sample dividend: {sample.get('ticker')} - ${sample.get('cash_amount')} on {sample.get('ex_dividend_date')}")
                return True
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_apple_stock_data():
    """Test Apple (AAPL) specific data"""
    print("\n🍎 Testing Apple (AAPL) stock data...")
    
    try:
        # Get AAPL ticker details
        url = f'{BASE_URL}/v3/reference/tickers/AAPL'
        headers = {'Authorization': f'Bearer {API_KEY}'}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', {})
            
            print(f"✅ AAPL Ticker Details:")
            print(f"   Name: {results.get('name')}")
            print(f"   Market: {results.get('market')}")
            print(f"   Type: {results.get('type')}")
            print(f"   Currency: {results.get('currency_name')}")
            print(f"   Active: {results.get('active')}")
            
            return True
        else:
            print(f"❌ AAPL data error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ AAPL data error: {e}")
        return False

def test_market_data():
    """Test market data endpoints"""
    print("\n📈 Testing market data...")
    
    try:
        # Get previous close for AAPL
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = f'{BASE_URL}/v2/aggs/ticker/AAPL/prev'
        headers = {'Authorization': f'Bearer {API_KEY}'}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            if results:
                result = results[0]
                print(f"✅ AAPL Previous Close:")
                print(f"   Open: ${result.get('o')}")
                print(f"   High: ${result.get('h')}")
                print(f"   Low: ${result.get('l')}")
                print(f"   Close: ${result.get('c')}")
                print(f"   Volume: {result.get('v'):,}")
                print(f"   Date: {datetime.fromtimestamp(result.get('t')/1000).strftime('%Y-%m-%d')}")
                
                return result
            else:
                print("⚠️ No market data available")
                return None
        else:
            print(f"❌ Market data error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Market data error: {e}")
        return None

def test_aggregates_data():
    """Test aggregates (OHLCV) data"""
    print("\n📊 Testing aggregates data...")
    
    try:
        # Get last 5 days of AAPL data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        url = f'{BASE_URL}/v2/aggs/ticker/AAPL/range/1/day/{start_date}/{end_date}'
        headers = {'Authorization': f'Bearer {API_KEY}'}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            print(f"✅ AAPL Aggregates ({len(results)} days):")
            for result in results[-3:]:  # Show last 3 days
                date = datetime.fromtimestamp(result.get('t')/1000).strftime('%Y-%m-%d')
                print(f"   {date}: O=${result.get('o'):.2f} H=${result.get('h'):.2f} L=${result.get('l'):.2f} C=${result.get('c'):.2f} V={result.get('v'):,}")
            
            return results
        else:
            print(f"❌ Aggregates error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Aggregates error: {e}")
        return None

def test_news_data():
    """Test news data"""
    print("\n📰 Testing news data...")
    
    try:
        url = f'{BASE_URL}/v2/reference/news'
        headers = {'Authorization': f'Bearer {API_KEY}'}
        params = {
            'ticker': 'AAPL',
            'limit': 5
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            print(f"✅ AAPL News ({len(results)} articles):")
            for article in results[:3]:  # Show first 3
                title = article.get('title', 'No title')[:60] + '...'
                published = article.get('published_utc', '')[:10]
                print(f"   📰 {published}: {title}")
            
            return results
        else:
            print(f"❌ News error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ News error: {e}")
        return None

def cross_examine_with_trading_economics():
    """Cross-examine Polygon.io data with Trading Economics"""
    print("\n🔄 Cross-examining data sources...")
    
    # Get Polygon.io AAPL data
    polygon_data = test_market_data()
    
    if polygon_data:
        polygon_close = polygon_data.get('c')
        polygon_volume = polygon_data.get('v')
        
        print(f"\n📊 Data Comparison:")
        print(f"   Polygon.io AAPL Close: ${polygon_close}")
        print(f"   Polygon.io AAPL Volume: {polygon_volume:,}")
        
        # Compare with our existing data (from index.html)
        dashboard_close = 227.20
        dashboard_volume = 28567234
        
        print(f"   Dashboard AAPL Close: ${dashboard_close}")
        print(f"   Dashboard AAPL Volume: {dashboard_volume:,}")
        
        # Calculate differences
        price_diff = abs(polygon_close - dashboard_close) if polygon_close else 0
        volume_diff = abs(polygon_volume - dashboard_volume) if polygon_volume else 0
        
        print(f"\n🔍 Analysis:")
        print(f"   Price difference: ${price_diff:.2f}")
        print(f"   Volume difference: {volume_diff:,}")
        
        if price_diff < 5:  # Within $5
            print(f"   ✅ Price data is consistent")
        else:
            print(f"   ⚠️ Price data shows significant difference")
        
        return {
            'polygon_close': polygon_close,
            'polygon_volume': polygon_volume,
            'price_diff': price_diff,
            'volume_diff': volume_diff
        }
    
    return None

def main():
    """Run all Polygon.io API tests"""
    print("🚀 Testing Polygon.io API Integration")
    print("=" * 50)
    
    # Test basic connection
    if not test_basic_connection():
        print("❌ Basic connection failed. Check API key and internet connection.")
        return
    
    # Test different data types
    test_apple_stock_data()
    market_data = test_market_data()
    aggregates_data = test_aggregates_data()
    news_data = test_news_data()
    
    # Cross-examine with existing data
    comparison = cross_examine_with_trading_economics()
    
    print("\n" + "=" * 50)
    print("🎉 Polygon.io API testing completed!")
    
    if market_data and aggregates_data:
        print(f"✅ Successfully retrieved comprehensive AAPL data")
        print(f"📊 Ready for integration into Apple ML Trading Dashboard")
        print(f"🔄 Data cross-examination completed")
    
    return {
        'market_data': market_data,
        'aggregates_data': aggregates_data,
        'news_data': news_data,
        'comparison': comparison
    }

if __name__ == "__main__":
    results = main()
