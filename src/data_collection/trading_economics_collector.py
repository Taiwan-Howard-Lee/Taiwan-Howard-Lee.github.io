"""
Trading Economics Data Collector - Economic indicators and market context data
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import time
import json

class TradingEconomicsCollector:
    """
    Collector for economic indicators, market data, and macroeconomic context
    from Trading Economics API to enhance Apple ML Trading analysis.
    """
    
    def __init__(self, api_key: str = "50439e96184c4b1:7008dwvh5w03yxa"):
        """
        Initialize Trading Economics collector.
        
        Args:
            api_key (str): Trading Economics API key
        """
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.logger = self._setup_logger()
        
        # Available countries for free tier
        self.available_countries = ['mexico', 'sweden', 'new-zealand', 'thailand']
        
        # Key economic indicators that affect tech stocks
        self.key_indicators = {
            'gdp': 'GDP Growth Rate',
            'inflation': 'Inflation Rate', 
            'unemployment': 'Unemployment Rate',
            'interest_rate': 'Interest Rate',
            'consumer_confidence': 'Consumer Confidence',
            'business_confidence': 'Business Confidence',
            'manufacturing': 'Manufacturing PMI',
            'retail_sales': 'Retail Sales',
            'industrial_production': 'Industrial Production'
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the collector."""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make API request with error handling.
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Additional parameters
            
        Returns:
            Dict: API response data or None if error
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            
            # Add API key to parameters
            if params is None:
                params = {}
            params['c'] = self.api_key
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Request error: {str(e)}")
            return None
    
    def get_economic_indicators(self, country: str = 'mexico') -> Optional[pd.DataFrame]:
        """
        Get economic indicators for a country.
        
        Args:
            country (str): Country name (mexico, sweden, new-zealand, thailand)
            
        Returns:
            pd.DataFrame: Economic indicators data
        """
        if country not in self.available_countries:
            self.logger.warning(f"Country {country} not available. Using mexico instead.")
            country = 'mexico'
        
        self.logger.info(f"Fetching economic indicators for {country}")
        
        data = self._make_request(f"indicators/{country}")
        
        if data:
            df = pd.DataFrame(data)

            # Clean and process data
            df['LatestValueDate'] = pd.to_datetime(df['LatestValueDate'], errors='coerce')

            # Note: Economic indicators endpoint doesn't include values, only metadata
            # We'll need to fetch individual indicator values separately if needed

            self.logger.info(f"Retrieved {len(df)} economic indicators metadata")
            return df
        
        return None
    
    def get_currency_data(self) -> Optional[pd.DataFrame]:
        """
        Get currency exchange rates.
        
        Returns:
            pd.DataFrame: Currency data
        """
        self.logger.info("Fetching currency data")
        
        data = self._make_request("markets/currency")
        
        if data:
            df = pd.DataFrame(data)
            
            # Focus on USD pairs relevant to Apple (US company)
            usd_pairs = df[df['Symbol'].str.contains('USD', na=False)].copy()
            
            # Clean data
            usd_pairs['Last'] = pd.to_numeric(usd_pairs['Last'], errors='coerce')
            usd_pairs['DailyChange'] = pd.to_numeric(usd_pairs['DailyChange'], errors='coerce')
            usd_pairs['DailyPercentualChange'] = pd.to_numeric(usd_pairs['DailyPercentualChange'], errors='coerce')
            
            self.logger.info(f"Retrieved {len(usd_pairs)} USD currency pairs")
            return usd_pairs
        
        return None
    
    def get_commodity_data(self) -> Optional[pd.DataFrame]:
        """
        Get commodity prices.
        
        Returns:
            pd.DataFrame: Commodity data
        """
        self.logger.info("Fetching commodity data")
        
        data = self._make_request("markets/commodities")
        
        if data:
            df = pd.DataFrame(data)
            
            # Clean data
            df['Last'] = pd.to_numeric(df['Last'], errors='coerce')
            df['DailyChange'] = pd.to_numeric(df['DailyChange'], errors='coerce')
            df['DailyPercentualChange'] = pd.to_numeric(df['DailyPercentualChange'], errors='coerce')
            
            self.logger.info(f"Retrieved {len(df)} commodities")
            return df
        
        return None
    
    def get_market_summary(self) -> Dict[str, any]:
        """
        Get comprehensive market summary for Apple analysis context.
        
        Returns:
            Dict: Market summary with economic indicators, currencies, commodities
        """
        self.logger.info("Generating market summary for Apple analysis")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'economic_indicators': {},
            'currencies': {},
            'commodities': {},
            'market_sentiment': 'neutral'
        }
        
        # Get economic indicators (metadata only)
        econ_data = self.get_economic_indicators()
        if econ_data is not None and not econ_data.empty:
            for _, row in econ_data.iterrows():
                category = row['Category']
                date = row['LatestValueDate']
                unit = row.get('Unit', '')
                ticker = row.get('Ticker', '')

                summary['economic_indicators'][category] = {
                    'ticker': ticker,
                    'date': date.isoformat() if pd.notna(date) else None,
                    'unit': unit,
                    'source': row.get('Source', '')
                }
        
        # Get currency data
        currency_data = self.get_currency_data()
        if currency_data is not None and not currency_data.empty:
            for _, row in currency_data.iterrows():
                symbol = row['Symbol']
                rate = row['Last']
                change = row['DailyPercentualChange']
                
                summary['currencies'][symbol] = {
                    'rate': rate,
                    'daily_change_pct': change
                }
        
        # Get commodity data
        commodity_data = self.get_commodity_data()
        if commodity_data is not None and not commodity_data.empty:
            for _, row in commodity_data.iterrows():
                symbol = row.get('Symbol', 'Unknown')
                price = row['Last']
                change = row['DailyPercentualChange']
                
                summary['commodities'][symbol] = {
                    'price': price,
                    'daily_change_pct': change
                }
        
        # Simple market sentiment analysis
        currency_changes = [v['daily_change_pct'] for v in summary['currencies'].values() 
                          if pd.notna(v['daily_change_pct'])]
        
        if currency_changes:
            avg_currency_change = np.mean(currency_changes)
            if avg_currency_change > 0.5:
                summary['market_sentiment'] = 'risk_on'
            elif avg_currency_change < -0.5:
                summary['market_sentiment'] = 'risk_off'
        
        self.logger.info("Market summary generated successfully")
        return summary
    
    def save_market_data(self, filename: str = None) -> str:
        """
        Save current market data to JSON file.
        
        Args:
            filename (str): Output filename
            
        Returns:
            str: Saved filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/external/trading_economics_data_{timestamp}.json"
        
        summary = self.get_market_summary()
        
        try:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Market data saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return None
