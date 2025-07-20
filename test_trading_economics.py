#!/usr/bin/env python3
"""
Test Trading Economics API integration
"""

import requests
import json
from datetime import datetime, timedelta

# API Configuration
API_KEY = '50439e96184c4b1:7008dwvh5w03yxa'
BASE_URL = 'https://api.tradingeconomics.com'

def test_basic_connection():
    """Test basic API connection with available free countries"""
    print("🔍 Testing basic API connection...")

    # Free tier countries: Sweden, Mexico, New Zealand, Thailand
    free_countries = ['mexico', 'sweden', 'new-zealand', 'thailand']

    for country in free_countries:
        try:
            url = f'{BASE_URL}/country/{country}?c={API_KEY}'
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Connection successful with {country.title()}!")
                print(f"📊 Retrieved {len(data)} data points for {country.title()}")

                # Show sample data
                if data:
                    sample = data[0] if isinstance(data, list) else data
                    print(f"📋 Sample data keys: {list(sample.keys())[:10]}")
                    return True, country
            else:
                print(f"⚠️ {country}: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ {country} error: {e}")

    return False, None

def test_economic_indicators(country='mexico'):
    """Test economic indicators for available free countries"""
    print(f"\n🏦 Testing economic indicators for {country.title()}...")

    results = {}

    try:
        # Get all indicators for the country
        url = f'{BASE_URL}/indicators/{country}?c={API_KEY}'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieved {len(data)} indicators for {country.title()}")

            # Show first 10 indicators
            for i, indicator in enumerate(data[:10]):
                category = indicator.get('Category', 'Unknown')
                value = indicator.get('LatestValue', 'N/A')
                date = indicator.get('LatestValueDate', 'N/A')
                unit = indicator.get('Unit', '')

                results[category] = {
                    'value': value,
                    'date': date,
                    'unit': unit
                }
                print(f"📊 {category}: {value} {unit} ({date})")

        else:
            print(f"❌ Indicators error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Indicators error: {e}")

    return results

def test_market_data():
    """Test market data endpoints"""
    print("\n📈 Testing market data...")
    
    try:
        # Test stock market indices
        url = f'{BASE_URL}/markets/index/united-states?c={API_KEY}'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Market indices: Retrieved {len(data)} indices")
            
            # Show major indices
            major_indices = ['S&P 500', 'NASDAQ', 'Dow Jones']
            for item in data[:10]:  # Show first 10
                name = item.get('Symbol', 'Unknown')
                value = item.get('Last', 'N/A')
                change = item.get('DailyChange', 'N/A')
                print(f"📊 {name}: {value} ({change:+.2f})" if isinstance(change, (int, float)) else f"📊 {name}: {value}")
                
        else:
            print(f"❌ Market data error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Market data error: {e}")

def test_currency_data():
    """Test currency exchange rates"""
    print("\n💱 Testing currency data...")
    
    try:
        url = f'{BASE_URL}/markets/currency?c={API_KEY}'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Currency data: Retrieved {len(data)} currency pairs")
            
            # Show major USD pairs
            usd_pairs = [item for item in data if 'USD' in item.get('Symbol', '')][:5]
            for pair in usd_pairs:
                symbol = pair.get('Symbol', 'Unknown')
                rate = pair.get('Last', 'N/A')
                change = pair.get('DailyChange', 'N/A')
                print(f"💱 {symbol}: {rate} ({change:+.4f})" if isinstance(change, (int, float)) else f"💱 {symbol}: {rate}")
                
        else:
            print(f"❌ Currency data error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Currency data error: {e}")

def test_commodity_data():
    """Test commodity prices"""
    print("\n🛢️ Testing commodity data...")
    
    try:
        url = f'{BASE_URL}/markets/commodities?c={API_KEY}'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Commodity data: Retrieved {len(data)} commodities")
            
            # Show key commodities
            key_commodities = ['Gold', 'Oil', 'Silver', 'Copper']
            for item in data[:10]:
                symbol = item.get('Symbol', 'Unknown')
                price = item.get('Last', 'N/A')
                change = item.get('DailyChange', 'N/A')
                if any(commodity.lower() in symbol.lower() for commodity in key_commodities):
                    print(f"🛢️ {symbol}: {price} ({change:+.2f})" if isinstance(change, (int, float)) else f"🛢️ {symbol}: {price}")
                
        else:
            print(f"❌ Commodity data error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Commodity data error: {e}")

def main():
    """Run all API tests"""
    print("🚀 Testing Trading Economics API Integration")
    print("=" * 50)
    
    # Test basic connection
    if not test_basic_connection():
        print("❌ Basic connection failed. Check API key and internet connection.")
        return
    
    # Test different data types
    economic_data = test_economic_indicators()
    test_market_data()
    test_currency_data()
    test_commodity_data()
    
    print("\n" + "=" * 50)
    print("🎉 Trading Economics API testing completed!")
    
    if economic_data:
        print(f"✅ Successfully retrieved {len(economic_data)} economic indicators")
        print("📊 Ready for integration into Apple ML Trading Dashboard")
    
    return economic_data

if __name__ == "__main__":
    results = main()
