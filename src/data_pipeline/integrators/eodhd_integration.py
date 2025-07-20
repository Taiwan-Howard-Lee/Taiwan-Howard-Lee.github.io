#!/usr/bin/env python3
"""
EODHD Integration Module
Integrates EODHD data with the existing multi-source framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data_pipeline.collectors.eodhd_collector import EODHDCollector
from src.data_pipeline.integrators.multi_source_integrator import MultiSourceDataIntegrator

class EODHDIntegration:
    """
    EODHD integration with multi-source framework
    Provides normalized data for the RL trading system
    """
    
    def __init__(self, api_key: str = "687c985a3deee0.98552733"):
        """Initialize EODHD integration"""
        self.collector = EODHDCollector(api_key)
        self.logger = self._setup_logger()
        
        # Data normalization mappings
        self.column_mappings = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'adjusted_close': 'Adjusted_Close',
            'volume': 'Volume'
        }
        
        # Feature mappings for RL system
        self.feature_mappings = {
            'price_features': ['Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume'],
            'fundamental_features': [
                'MarketCapitalization', 'PERatio', 'PEGRatio', 'BookValue',
                'DividendYield', 'EarningsShare', 'ProfitMargin', 'OperatingMarginTTM',
                'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'Beta'
            ],
            'technical_features': [
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14', 'macd_line',
                'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle'
            ],
            'sentiment_features': ['sentiment_score', 'news_count', 'sentiment_trend'],
            'volume_features': ['volume_sma_20', 'volume_ratio', 'volume_trend']
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - EODHD_INTEGRATION - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def normalize_eod_data(self, eod_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize EOD data to unified schema
        
        Args:
            eod_data (pd.DataFrame): Raw EOD data from EODHD
            
        Returns:
            pd.DataFrame: Normalized data
        """
        if eod_data is None or eod_data.empty:
            return pd.DataFrame()
        
        normalized = eod_data.copy()
        
        # Rename columns to match unified schema
        for old_col, new_col in self.column_mappings.items():
            if old_col in normalized.columns:
                normalized = normalized.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in normalized.columns:
                self.logger.warning(f"Missing required column: {col}")
        
        # Convert data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
        
        # Add data source identifier
        normalized['data_source'] = 'eodhd'
        
        return normalized
    
    def extract_fundamental_features(self, fundamentals: Dict) -> Dict[str, float]:
        """
        Extract key fundamental features for RL system
        
        Args:
            fundamentals (Dict): Raw fundamental data
            
        Returns:
            Dict: Extracted features
        """
        features = {}
        
        try:
            # General company data
            if 'General' in fundamentals:
                general = fundamentals['General']
                features['market_cap'] = general.get('MarketCapitalization', 0)
                features['sector'] = general.get('Sector', 'Unknown')
                features['industry'] = general.get('Industry', 'Unknown')
            
            # Financial highlights
            if 'Highlights' in fundamentals:
                highlights = fundamentals['Highlights']
                features['pe_ratio'] = highlights.get('PERatio', 0)
                features['peg_ratio'] = highlights.get('PEGRatio', 0)
                features['book_value'] = highlights.get('BookValue', 0)
                features['dividend_yield'] = highlights.get('DividendYield', 0)
                features['earnings_share'] = highlights.get('EarningsShare', 0)
                features['profit_margin'] = highlights.get('ProfitMargin', 0)
                features['operating_margin'] = highlights.get('OperatingMarginTTM', 0)
                features['roa'] = highlights.get('ReturnOnAssetsTTM', 0)
                features['roe'] = highlights.get('ReturnOnEquityTTM', 0)
            
            # Valuation metrics
            if 'Valuation' in fundamentals:
                valuation = fundamentals['Valuation']
                features['trailing_pe'] = valuation.get('TrailingPE', 0)
                features['forward_pe'] = valuation.get('ForwardPE', 0)
                features['price_to_sales'] = valuation.get('PriceSalesTTM', 0)
                features['price_to_book'] = valuation.get('PriceBookMRQ', 0)
            
            # Technical metrics
            if 'Technicals' in fundamentals:
                technicals = fundamentals['Technicals']
                features['beta'] = technicals.get('Beta', 1.0)
                features['52_week_high'] = technicals.get('52WeekHigh', 0)
                features['52_week_low'] = technicals.get('52WeekLow', 0)
                features['50_day_ma'] = technicals.get('50DayMA', 0)
                features['200_day_ma'] = technicals.get('200DayMA', 0)
            
        except Exception as e:
            self.logger.error(f"Error extracting fundamental features: {str(e)}")
        
        return features
    
    def extract_sentiment_features(self, sentiment_data: Dict, news_data: List[Dict] = None) -> Dict[str, float]:
        """
        Extract sentiment features for RL system
        
        Args:
            sentiment_data (Dict): Sentiment data
            news_data (List[Dict]): News data
            
        Returns:
            Dict: Sentiment features
        """
        features = {}
        
        try:
            if sentiment_data:
                # Calculate average sentiment
                sentiment_scores = []
                for symbol, data in sentiment_data.items():
                    if isinstance(data, list):
                        for entry in data:
                            if 'normalized' in entry:
                                sentiment_scores.append(entry['normalized'])
                
                if sentiment_scores:
                    features['sentiment_score'] = np.mean(sentiment_scores)
                    features['sentiment_std'] = np.std(sentiment_scores)
                    features['sentiment_trend'] = sentiment_scores[-1] - sentiment_scores[0] if len(sentiment_scores) > 1 else 0
                else:
                    features['sentiment_score'] = 0
                    features['sentiment_std'] = 0
                    features['sentiment_trend'] = 0
            
            # News count and recency
            if news_data:
                features['news_count'] = len(news_data)
                
                # Calculate news recency (days since last news)
                if news_data:
                    latest_news = max(news_data, key=lambda x: x.get('date', ''))
                    latest_date = pd.to_datetime(latest_news.get('date', ''))
                    days_since = (datetime.now() - latest_date).days
                    features['news_recency'] = days_since
                else:
                    features['news_recency'] = 999
            else:
                features['news_count'] = 0
                features['news_recency'] = 999
        
        except Exception as e:
            self.logger.error(f"Error extracting sentiment features: {str(e)}")
            features = {'sentiment_score': 0, 'sentiment_std': 0, 'sentiment_trend': 0, 'news_count': 0, 'news_recency': 999}
        
        return features
    
    def create_rl_features(self, comprehensive_data: Dict) -> pd.DataFrame:
        """
        Create feature set for RL trading system
        
        Args:
            comprehensive_data (Dict): Comprehensive data from EODHD
            
        Returns:
            pd.DataFrame: Feature dataset for RL system
        """
        self.logger.info(f"🔧 Creating RL features for {comprehensive_data.get('symbol', 'Unknown')}")
        
        # Start with EOD data as base
        eod_data = comprehensive_data['data'].get('eod')
        if eod_data is None or eod_data.empty:
            self.logger.error("No EOD data available for feature creation")
            return pd.DataFrame()
        
        # Normalize EOD data
        features_df = self.normalize_eod_data(eod_data)
        
        # Add fundamental features (broadcast to all dates)
        fundamentals = comprehensive_data['data'].get('fundamentals')
        if fundamentals:
            fund_features = self.extract_fundamental_features(fundamentals)
            for feature_name, feature_value in fund_features.items():
                if isinstance(feature_value, (int, float)):
                    features_df[feature_name] = feature_value
        
        # Add sentiment features (broadcast to all dates)
        sentiment_data = comprehensive_data['data'].get('sentiment')
        news_data = comprehensive_data['data'].get('news')
        sentiment_features = self.extract_sentiment_features(sentiment_data, news_data)
        for feature_name, feature_value in sentiment_features.items():
            features_df[feature_name] = feature_value
        
        # Add technical indicators
        technical_indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands']
        for indicator in technical_indicators:
            tech_data = comprehensive_data['data'].get(f'technical_{indicator}')
            if tech_data is not None and not tech_data.empty:
                # Merge technical data with main features
                tech_data_clean = tech_data.select_dtypes(include=[np.number])
                for col in tech_data_clean.columns:
                    if col != 'symbol':
                        features_df[f'{indicator}_{col}'] = tech_data_clean[col]
        
        # Add volume-based features
        if 'Volume' in features_df.columns:
            features_df['volume_sma_20'] = features_df['Volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_sma_20']
            features_df['volume_trend'] = features_df['Volume'].pct_change(5)
        
        # Add price-based features
        if 'Close' in features_df.columns:
            features_df['price_change'] = features_df['Close'].pct_change()
            features_df['price_volatility'] = features_df['price_change'].rolling(window=20).std()
            features_df['price_momentum'] = features_df['Close'].pct_change(5)
        
        # Add dividend information
        dividends = comprehensive_data['data'].get('dividends')
        if dividends is not None and not dividends.empty:
            # Create dividend yield feature
            latest_dividend = dividends['dividend'].iloc[-1] if 'dividend' in dividends.columns else 0
            latest_price = features_df['Close'].iloc[-1] if 'Close' in features_df.columns else 1
            dividend_yield = (latest_dividend * 4) / latest_price  # Annualized
            features_df['dividend_yield_calculated'] = dividend_yield
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        self.logger.info(f"✅ RL features created: {len(features_df)} records, {len(features_df.columns)} features")
        
        return features_df
    
    def integrate_with_multi_source(self, symbols: List[str], integrator: MultiSourceDataIntegrator) -> Dict[str, Any]:
        """
        Integrate EODHD data with multi-source framework
        
        Args:
            symbols (List[str]): Symbols to collect
            integrator (MultiSourceDataIntegrator): Multi-source integrator instance
            
        Returns:
            Dict: Integration results
        """
        self.logger.info(f"🔗 Integrating EODHD with multi-source framework")
        self.logger.info(f"📊 Symbols: {symbols}")
        
        # Collect comprehensive EODHD data
        eodhd_results = self.collector.collect_batch_comprehensive(symbols, max_symbols=len(symbols))
        
        # Convert to format compatible with multi-source integrator
        eodhd_datasets = {}
        
        for symbol, symbol_data in eodhd_results['results'].items():
            # Create RL features
            rl_features = self.create_rl_features(symbol_data)
            
            if not rl_features.empty:
                eodhd_datasets[symbol] = rl_features
        
        # Register EODHD as a data source in the integrator
        integrator.register_new_source(
            source_name='eodhd',
            collector_class=self.collector,
            config={
                'enabled': True,
                'priority': 1,  # High priority due to comprehensive data
                'data_types': ['eod', 'fundamentals', 'news', 'sentiment', 'technical'],
                'rate_limit': '1_per_second'
            }
        )
        
        # Combine EODHD datasets
        if eodhd_datasets:
            combined_eodhd = pd.concat(eodhd_datasets.values(), ignore_index=False)
            combined_eodhd['data_source'] = 'eodhd'
            
            # Integrate with other sources
            source_data = {'eodhd': combined_eodhd}
            unified_data = integrator.normalize_and_integrate(source_data)
            
            integration_results = {
                'eodhd_results': eodhd_results,
                'unified_data': unified_data,
                'integration_summary': integrator.get_integration_summary()
            }
            
            self.logger.info(f"🎉 EODHD integration completed")
            self.logger.info(f"   📊 EODHD records: {len(combined_eodhd)}")
            self.logger.info(f"   🔗 Unified records: {len(unified_data)}")
            
            return integration_results
        
        else:
            self.logger.error("❌ No EODHD data available for integration")
            return {}

def main():
    """Test EODHD integration"""
    print("🧪 Testing EODHD Integration")
    print("=" * 35)
    
    # Initialize integration
    eodhd_integration = EODHDIntegration()
    
    # Initialize multi-source integrator
    multi_integrator = MultiSourceDataIntegrator()
    
    # Test symbols
    test_symbols = ['AAPL.US', 'MSFT.US']
    
    # Run integration
    results = eodhd_integration.integrate_with_multi_source(test_symbols, multi_integrator)
    
    if results:
        print(f"\n📊 Integration Results:")
        print(f"   EODHD symbols: {results['eodhd_results']['symbols_processed']}")
        print(f"   Unified records: {len(results['unified_data'])}")
        print(f"   Features: {len(results['unified_data'].columns)}")
        
        # Show sample features
        if not results['unified_data'].empty:
            print(f"\n🔧 Sample Features:")
            for col in results['unified_data'].columns[:10]:
                print(f"     {col}")

if __name__ == "__main__":
    main()
