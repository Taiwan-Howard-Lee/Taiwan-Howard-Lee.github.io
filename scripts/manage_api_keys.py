#!/usr/bin/env python3
"""
API Key Management Script
Interactive script to manage multiple API keys for all data sources
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_pipeline.utils.multi_api_key_manager import MultiAPIKeyManager

class APIKeyManager:
    """Interactive API key management interface"""
    
    def __init__(self):
        self.manager = MultiAPIKeyManager()
        
    def show_current_status(self):
        """Display current API key status for all data sources"""
        print("\n" + "="*60)
        print("🔑 CURRENT API KEY STATUS")
        print("="*60)
        
        stats = self.manager.get_usage_stats()
        
        for source_name, source_stats in stats.items():
            print(f"\n📊 {source_name.upper()}:")
            print(f"   Total slots: 3")
            print(f"   Active keys: {source_stats['active_keys']}")
            print(f"   Available slots: {3 - source_stats['active_keys']}")
            
            for i, key_detail in enumerate(source_stats['key_details']):
                status_emoji = "✅" if key_detail['status'] == 'healthy' else "⚠️"
                print(f"   Slot {i+1}: {status_emoji} {key_detail['key_preview']} ({key_detail['tier']} tier)")
                print(f"           Usage: {key_detail['usage_count']} requests, {key_detail['error_count']} errors")
            
            # Show empty slots
            empty_slots = 3 - len(source_stats['key_details'])
            for i in range(empty_slots):
                slot_num = len(source_stats['key_details']) + i + 1
                print(f"   Slot {slot_num}: 🔓 AVAILABLE - Ready for API key")
    
    def add_api_key_interactive(self):
        """Interactive API key addition"""
        print("\n" + "="*60)
        print("➕ ADD NEW API KEY")
        print("="*60)
        
        # Show available data sources
        data_sources = list(self.manager.config.get('data_sources', {}).keys())
        print("\nAvailable data sources:")
        for i, source in enumerate(data_sources, 1):
            print(f"   {i}. {source}")
        
        try:
            choice = int(input(f"\nSelect data source (1-{len(data_sources)}): ")) - 1
            if choice < 0 or choice >= len(data_sources):
                print("❌ Invalid choice")
                return
            
            source_name = data_sources[choice]
            
            # Get API key
            api_key = input(f"\nEnter API key for {source_name}: ").strip()
            if not api_key:
                print("❌ API key cannot be empty")
                return
            
            # Get tier
            print("\nAPI key tier:")
            print("   1. Free")
            print("   2. Premium")
            print("   3. Enterprise")
            
            tier_choice = input("Select tier (1-3, default: 1): ").strip() or "1"
            tier_map = {"1": "free", "2": "premium", "3": "enterprise"}
            tier = tier_map.get(tier_choice, "free")
            
            # Get description
            description = input(f"Description (optional): ").strip() or f"{source_name} API key"
            
            # Add the key
            success = self.manager.add_api_key(source_name, api_key, tier, description)
            
            if success:
                print(f"✅ Successfully added API key to {source_name}")
            else:
                print(f"❌ Failed to add API key to {source_name}")
                
        except ValueError:
            print("❌ Invalid input")
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
    
    def test_api_keys(self):
        """Test all API keys"""
        print("\n" + "="*60)
        print("🧪 TESTING API KEYS")
        print("="*60)
        
        # Test each data source
        for source_name in self.manager.config.get('data_sources', {}):
            print(f"\n🔧 Testing {source_name}:")
            
            available_keys = self.manager.get_available_keys(source_name)
            if not available_keys:
                print("   ❌ No API keys available")
                continue
            
            for i in range(3):  # Test key rotation
                key = self.manager.get_next_key(source_name)
                if key:
                    key_preview = key[:8] + '...' if len(key) > 8 else key
                    print(f"   Request {i+1}: ✅ {key_preview}")
                    self.manager.report_key_success(source_name, key)
                else:
                    print(f"   Request {i+1}: ❌ No key available")
    
    def show_rate_limits(self):
        """Show rate limits for all data sources"""
        print("\n" + "="*60)
        print("⏱️ RATE LIMITS & CAPACITY")
        print("="*60)
        
        for source_name, source_config in self.manager.config.get('data_sources', {}).items():
            print(f"\n📊 {source_name.upper()}:")
            
            rate_limit = source_config.get('rate_limit', 'unknown')
            print(f"   Rate limit: {rate_limit}")
            
            available_keys = self.manager.get_available_keys(source_name)
            active_keys = len(available_keys)
            
            if 'per_day' in rate_limit:
                daily_limit = int(rate_limit.split('_')[0]) * active_keys
                print(f"   Daily capacity: {daily_limit:,} requests ({active_keys} keys)")
            elif 'per_minute' in rate_limit:
                minute_limit = int(rate_limit.split('_')[0]) * active_keys
                print(f"   Per-minute capacity: {minute_limit} requests ({active_keys} keys)")
            elif 'unlimited' in rate_limit:
                print(f"   Unlimited capacity ({active_keys} keys)")
            
            print(f"   Available slots: {3 - active_keys}")
    
    def show_upgrade_recommendations(self):
        """Show upgrade recommendations"""
        print("\n" + "="*60)
        print("💎 UPGRADE RECOMMENDATIONS")
        print("="*60)
        
        recommendations = {
            'eodhd': {
                'current': 'Free tier (20,000 requests/day)',
                'upgrade': 'All-In-One ($79.99/month)',
                'benefits': ['Fundamentals data', 'Technical indicators', 'Intraday data', 'Options data'],
                'capacity_boost': '3x with 3 API keys = 180,000 requests/day'
            },
            'alpha_vantage': {
                'current': 'Free tier (500 requests/day)',
                'upgrade': 'Premium ($49.99/month)',
                'benefits': ['25 requests/minute', 'Real-time data', 'Extended history'],
                'capacity_boost': '3x with 3 API keys = 1,500 requests/day'
            },
            'polygon': {
                'current': 'Free tier (5 requests/minute)',
                'upgrade': 'Starter ($99/month)',
                'benefits': ['100 requests/minute', 'Real-time data', 'Options data'],
                'capacity_boost': '3x with 3 API keys = 15 requests/minute'
            }
        }
        
        for source_name, rec in recommendations.items():
            print(f"\n📊 {source_name.upper()}:")
            print(f"   Current: {rec['current']}")
            print(f"   Recommended: {rec['upgrade']}")
            print(f"   Benefits: {', '.join(rec['benefits'])}")
            print(f"   With 3 keys: {rec['capacity_boost']}")
    
    def main_menu(self):
        """Main interactive menu"""
        while True:
            print("\n" + "="*60)
            print("🔑 API KEY MANAGEMENT SYSTEM")
            print("="*60)
            print("1. 📊 Show current API key status")
            print("2. ➕ Add new API key")
            print("3. 🧪 Test API keys")
            print("4. ⏱️ Show rate limits & capacity")
            print("5. 💎 Show upgrade recommendations")
            print("6. 🚪 Exit")
            
            try:
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == '1':
                    self.show_current_status()
                elif choice == '2':
                    self.add_api_key_interactive()
                elif choice == '3':
                    self.test_api_keys()
                elif choice == '4':
                    self.show_rate_limits()
                elif choice == '5':
                    self.show_upgrade_recommendations()
                elif choice == '6':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main function"""
    print("🚀 Starting API Key Management System...")
    
    try:
        manager = APIKeyManager()
        manager.main_menu()
    except Exception as e:
        print(f"❌ Failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
