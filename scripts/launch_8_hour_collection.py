#!/usr/bin/env python3
"""
8-Hour Data Collection Session Launcher
Production-ready launcher for comprehensive data collection
"""

import sys
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_pipeline.orchestrators.comprehensive_data_orchestrator import (
    ComprehensiveDataOrchestrator, CollectionSession
)

def create_production_session() -> CollectionSession:
    """Create a production-ready 8-hour collection session"""
    return CollectionSession(
        session_id=f"production_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=datetime.now(),
        duration_hours=8,
        target_symbols=[
            # Major Tech (High Priority)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            
            # Major Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B', 'V', 'MA',
            
            # Major Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'BMY',
            
            # Major Consumer
            'HD', 'MCD', 'NKE', 'COST', 'WMT', 'PG', 'KO', 'PEP',
            
            # Major Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX',
            
            # Major Industrial
            'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT',
            
            # Major ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'GLD', 'TLT',
            
            # Sector ETFs
            'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE'
        ],
        data_sources=[
            'eodhd',                    # Priority 1: Comprehensive data
            'finnhub',                  # Priority 2: Real-time + fundamentals
            'financial_modeling_prep',  # Priority 3: Deep fundamentals
            'yahoo_finance',            # Priority 4: Reliable historical
            'alpha_vantage',           # Priority 5: Technical indicators
            'polygon'                  # Priority 6: High-quality market data
        ]
    )

def create_test_session() -> CollectionSession:
    """Create a test session (30 minutes)"""
    return CollectionSession(
        session_id=f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=datetime.now(),
        duration_hours=0.5,  # 30 minutes
        target_symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        data_sources=['eodhd', 'finnhub', 'financial_modeling_prep', 'yahoo_finance']
    )

def create_focused_session() -> CollectionSession:
    """Create a focused session (2 hours, high-value data only)"""
    return CollectionSession(
        session_id=f"focused_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        start_time=datetime.now(),
        duration_hours=2,
        target_symbols=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'JPM', 'BAC', 'JNJ', 'PFE', 'HD', 'MCD', 'XOM', 'CVX'
        ],
        data_sources=['eodhd', 'finnhub', 'financial_modeling_prep']
    )

def print_session_preview(session: CollectionSession):
    """Print session configuration preview"""
    print("🎯 SESSION CONFIGURATION")
    print("=" * 50)
    print(f"📊 Session ID: {session.session_id}")
    print(f"⏱️ Duration: {session.duration_hours} hours")
    print(f"🏢 Target symbols: {len(session.target_symbols)}")
    print(f"🔧 Data sources: {len(session.data_sources)}")
    print(f"🚀 Start time: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏁 End time: {(session.start_time + timedelta(hours=session.duration_hours)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 Symbols to collect:")
    for i, symbol in enumerate(session.target_symbols):
        if i % 10 == 0:
            print()
        print(f"{symbol:>6}", end=" ")
    
    print(f"\n\n🔧 Data sources (in priority order):")
    for i, source in enumerate(session.data_sources, 1):
        print(f"   {i}. {source}")
    
    print("\n" + "=" * 50)

def estimate_collection_capacity(session: CollectionSession):
    """Estimate collection capacity for the session"""
    print("📈 ESTIMATED COLLECTION CAPACITY")
    print("=" * 50)
    
    # API limits per source (with 3 keys each)
    api_limits = {
        'eodhd': {'daily': 60000, 'hourly': 2500},
        'finnhub': {'daily': 86400, 'hourly': 10800},  # 60/min * 60 * 24, 180/min * 60
        'financial_modeling_prep': {'daily': 750, 'hourly': 31},
        'yahoo_finance': {'daily': float('inf'), 'hourly': float('inf')},
        'alpha_vantage': {'daily': 1500, 'hourly': 62},
        'polygon': {'daily': 21600, 'hourly': 900}  # 15/min * 60 * 24, 15/min * 60
    }
    
    total_daily_capacity = 0
    total_hourly_capacity = 0
    
    for source in session.data_sources:
        if source in api_limits:
            daily = api_limits[source]['daily']
            hourly = api_limits[source]['hourly']
            
            if daily != float('inf'):
                total_daily_capacity += daily
            if hourly != float('inf'):
                total_hourly_capacity += hourly
            
            print(f"🔧 {source}:")
            print(f"   Daily capacity: {daily:,} requests" if daily != float('inf') else "   Daily capacity: Unlimited")
            print(f"   Hourly capacity: {hourly:,} requests" if hourly != float('inf') else "   Hourly capacity: Unlimited")
    
    session_capacity = min(total_daily_capacity, total_hourly_capacity * session.duration_hours)
    
    print(f"\n📊 SESSION ESTIMATES:")
    print(f"   Total API capacity: {session_capacity:,} requests")
    print(f"   Estimated records: {session_capacity * 0.8:,.0f} (80% efficiency)")
    print(f"   Records per symbol: {(session_capacity * 0.8) / len(session.target_symbols):,.0f}")
    print(f"   Records per hour: {(session_capacity * 0.8) / session.duration_hours:,.0f}")
    
    print("\n" + "=" * 50)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Launch 8-hour data collection session")
    parser.add_argument('--mode', choices=['production', 'test', 'focused'], 
                       default='production', help='Collection mode')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint file')
    parser.add_argument('--preview', action='store_true', help='Preview session without starting')
    parser.add_argument('--estimate', action='store_true', help='Show capacity estimates')
    
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE DATA COLLECTION ORCHESTRATOR")
    print("=" * 60)
    
    # Create session based on mode
    if args.mode == 'production':
        session = create_production_session()
    elif args.mode == 'test':
        session = create_test_session()
    elif args.mode == 'focused':
        session = create_focused_session()
    
    # Show session preview
    print_session_preview(session)
    
    # Show capacity estimates if requested
    if args.estimate:
        estimate_collection_capacity(session)
    
    # Preview mode - just show configuration and exit
    if args.preview:
        print("👀 Preview mode - session configuration shown above")
        return
    
    # Confirmation prompt for production mode
    if args.mode == 'production':
        print("\n⚠️  PRODUCTION MODE - This will run for 8 hours!")
        print("   - Make sure your system can run uninterrupted")
        print("   - Ensure stable internet connection")
        print("   - Monitor disk space for data storage")
        
        confirm = input("\n🤔 Continue with production session? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("❌ Session cancelled by user")
            return
    
    # Initialize orchestrator
    orchestrator = ComprehensiveDataOrchestrator(session)
    
    try:
        # Resume from checkpoint if specified
        if args.resume:
            print(f"🔄 Resuming from checkpoint: {args.resume}")
            results = orchestrator.resume_session(args.resume)
        else:
            # Start new session
            print(f"\n🚀 Starting {args.mode} data collection session...")
            print("   Press Ctrl+C to gracefully stop the session")
            results = orchestrator.start_8_hour_session()
        
        # Display final results
        print("\n🎉 SESSION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        summary = results['session_summary']
        print(f"📊 Session ID: {summary['session_id']}")
        print(f"⏱️ Duration: {summary['duration_hours']:.2f} hours")
        print(f"📈 Records collected: {summary['total_records_collected']:,}")
        print(f"📞 API calls made: {summary['total_api_calls']:,}")
        print(f"❌ Errors: {summary['errors_encountered']}")
        print(f"📊 Records/hour: {summary['records_per_hour']:.0f}")
        print(f"🔧 API efficiency: {results['data_quality_metrics']['api_efficiency']:.2f} records/call")
        
        # Source performance
        print(f"\n📊 SOURCE PERFORMANCE:")
        for source, stats in results['source_performance'].items():
            print(f"   {source}: {stats['total_records']:,} records, "
                  f"{stats['success_rate']:.1%} success")
        
        # Save final summary
        summary_file = Path(f"session_summary_{summary['session_id']}.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Full results saved to: {summary_file}")
        
    except KeyboardInterrupt:
        print("\n🛑 Session interrupted by user")
        print("💾 Data collected so far has been saved")
        
    except Exception as e:
        print(f"\n❌ Session failed: {str(e)}")
        print("💾 Partial data may have been saved")
        sys.exit(1)

if __name__ == "__main__":
    main()
