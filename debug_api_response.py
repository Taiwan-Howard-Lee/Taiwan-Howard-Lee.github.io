#!/usr/bin/env python3
"""
Debug API response structure
"""

import requests
import json

API_KEY = '50439e96184c4b1:7008dwvh5w03yxa'
BASE_URL = 'https://api.tradingeconomics.com'

def debug_api_structure():
    """Debug the actual API response structure"""
    
    # Test economic indicators
    print("🔍 Debugging economic indicators structure...")
    url = f'{BASE_URL}/indicators/mexico?c={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            print(f"📊 Sample indicator structure:")
            sample = data[0]
            print(json.dumps(sample, indent=2))
            print(f"\n📋 Available keys: {list(sample.keys())}")
        else:
            print("❌ No data returned")
    else:
        print(f"❌ API Error: {response.status_code}")
    
    # Test currency data
    print("\n🔍 Debugging currency data structure...")
    url = f'{BASE_URL}/markets/currency?c={API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            print(f"💱 Sample currency structure:")
            sample = data[0]
            print(json.dumps(sample, indent=2))
            print(f"\n📋 Available keys: {list(sample.keys())}")
        else:
            print("❌ No currency data returned")
    else:
        print(f"❌ Currency API Error: {response.status_code}")

if __name__ == "__main__":
    debug_api_structure()
