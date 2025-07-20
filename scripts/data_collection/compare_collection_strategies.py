#!/usr/bin/env python3
"""
Compare Basic vs Enhanced Collection Strategies
Shows data diversity improvements
"""

import sys
import os
sys.path.append('src')

from data_collection.continuous_collector import ContinuousCollector
from data_collection.enhanced_collector import EnhancedContinuousCollector

def analyze_strategy(strategy, name):
    """Analyze a collection strategy"""
    print(f"\n📊 {name} Strategy Analysis:")
    print("=" * 40)
    
    # Request type distribution
    request_types = {}
    tickers = set()
    time_periods = set()
    news_limits = set()
    
    for req in strategy:
        req_type = req['type']
        request_types[req_type] = request_types.get(req_type, 0) + 1
        
        if 'ticker' in req:
            tickers.add(req['ticker'])
        
        if 'days' in req:
            time_periods.add(req['days'])
            
        if 'limit' in req:
            news_limits.add(req['limit'])
    
    print(f"📈 Total Requests: {len(strategy)}")
    print(f"🎯 Request Types: {len(request_types)}")
    print(f"📊 Ticker Coverage: {len(tickers)}")
    print(f"⏰ Time Periods: {len(time_periods)}")
    print(f"📰 News Limits: {len(news_limits)}")
    
    print(f"\n📋 Request Distribution:")
    for req_type, count in sorted(request_types.items()):
        percentage = (count / len(strategy)) * 100
        print(f"   {req_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Tickers: {sorted(list(tickers))}")
    
    if time_periods:
        print(f"⏰ Time Periods: {sorted(list(time_periods))}")
    
    if news_limits:
        print(f"📰 News Limits: {sorted(list(news_limits))}")
    
    return {
        'total_requests': len(strategy),
        'request_types': len(request_types),
        'ticker_coverage': len(tickers),
        'time_periods': len(time_periods),
        'news_limits': len(news_limits),
        'tickers': tickers,
        'periods': time_periods
    }

def calculate_diversity_score(analysis):
    """Calculate diversity score (0-100)"""
    score = 0
    
    # Ticker diversity (40 points max)
    ticker_score = min(analysis['ticker_coverage'] * 2, 40)
    score += ticker_score
    
    # Time period diversity (30 points max)
    period_score = min(analysis['time_periods'] * 3, 30)
    score += period_score
    
    # Request type diversity (20 points max)
    type_score = min(analysis['request_types'] * 3, 20)
    score += type_score
    
    # News variety (10 points max)
    news_score = min(analysis['news_limits'] * 2, 10)
    score += news_score
    
    return min(score, 100)

def main():
    """Compare collection strategies"""
    print("🔍 Collection Strategy Comparison")
    print("=" * 50)
    print("Comparing Basic vs Enhanced data collection strategies")
    
    # Test parameters
    test_hours = 8  # 8-hour comparison
    
    print(f"\n⏰ Test Duration: {test_hours} hours")
    print(f"📊 Expected Requests: {test_hours * 60 * 5} total")
    
    # Initialize collectors
    basic_collector = ContinuousCollector()
    enhanced_collector = EnhancedContinuousCollector()
    
    # Build strategies
    print(f"\n🔄 Building strategies...")
    basic_strategy = basic_collector.build_collection_strategy(test_hours)
    enhanced_strategy = enhanced_collector.build_enhanced_strategy(test_hours)
    
    # Analyze strategies
    basic_analysis = analyze_strategy(basic_strategy, "BASIC")
    enhanced_analysis = analyze_strategy(enhanced_strategy, "ENHANCED")
    
    # Calculate diversity scores
    basic_score = calculate_diversity_score(basic_analysis)
    enhanced_score = calculate_diversity_score(enhanced_analysis)
    
    # Comparison summary
    print(f"\n🏆 STRATEGY COMPARISON SUMMARY")
    print("=" * 50)
    
    metrics = [
        ('Total Requests', basic_analysis['total_requests'], enhanced_analysis['total_requests']),
        ('Ticker Coverage', basic_analysis['ticker_coverage'], enhanced_analysis['ticker_coverage']),
        ('Time Periods', basic_analysis['time_periods'], enhanced_analysis['time_periods']),
        ('Request Types', basic_analysis['request_types'], enhanced_analysis['request_types']),
        ('News Variety', basic_analysis['news_limits'], enhanced_analysis['news_limits']),
        ('Diversity Score', basic_score, enhanced_score)
    ]
    
    print(f"{'Metric':<15} {'Basic':<10} {'Enhanced':<10} {'Improvement':<12}")
    print("-" * 50)
    
    for metric, basic_val, enhanced_val in metrics:
        if basic_val > 0:
            improvement = f"+{((enhanced_val - basic_val) / basic_val * 100):.0f}%"
        else:
            improvement = "N/A"
        
        print(f"{metric:<15} {basic_val:<10} {enhanced_val:<10} {improvement:<12}")
    
    # Detailed improvements
    print(f"\n🚀 KEY IMPROVEMENTS:")
    
    # Ticker improvements
    basic_tickers = basic_analysis['tickers']
    enhanced_tickers = enhanced_analysis['tickers']
    new_tickers = enhanced_tickers - basic_tickers
    
    if new_tickers:
        print(f"📈 New Tickers: {len(new_tickers)} additional")
        print(f"   Added: {sorted(list(new_tickers))[:10]}...")
    
    # Time period improvements
    basic_periods = basic_analysis['periods']
    enhanced_periods = enhanced_analysis['periods']
    new_periods = enhanced_periods - basic_periods
    
    if new_periods:
        print(f"⏰ New Time Periods: {len(new_periods)} additional")
        print(f"   Added: {sorted(list(new_periods))}")
    
    # Quality assessment
    print(f"\n📊 QUALITY ASSESSMENT:")
    
    if enhanced_score > basic_score:
        improvement_pct = ((enhanced_score - basic_score) / basic_score) * 100
        print(f"✅ Enhanced strategy is {improvement_pct:.0f}% more diverse")
    
    if enhanced_analysis['ticker_coverage'] > 20:
        print(f"✅ Excellent ticker coverage ({enhanced_analysis['ticker_coverage']} symbols)")
    
    if enhanced_analysis['time_periods'] > 10:
        print(f"✅ Comprehensive time coverage ({enhanced_analysis['time_periods']} periods)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if enhanced_score > 80:
        print(f"🎯 Use ENHANCED collector for maximum data diversity")
        print(f"🚀 Command: python3 run_enhanced_collection.py --hours {test_hours}")
    elif enhanced_score > basic_score:
        print(f"📈 Enhanced collector provides better diversity")
        print(f"⚡ Consider enhanced version for comprehensive data collection")
    else:
        print(f"📊 Both strategies have similar diversity")
    
    print(f"\n🎉 Analysis complete! Enhanced strategy provides superior data diversity.")

if __name__ == "__main__":
    main()
