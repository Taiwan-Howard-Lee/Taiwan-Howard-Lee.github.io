"""
InfluxDB Client - Time-series database integration for financial data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, List
import logging

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    print("InfluxDB client not available. Install with: pip install influxdb-client")


class InfluxDBManager:
    """
    Manager class for InfluxDB operations with financial time-series data.
    Handles writing and querying OHLCV data and other financial metrics.
    """
    
    def __init__(self, url: str = "http://localhost:8086", 
                 token: str = "", 
                 org: str = "apple_trading", 
                 bucket: str = "financial_data"):
        """
        Initialize InfluxDB connection.
        
        Args:
            url (str): InfluxDB server URL
            token (str): Authentication token
            org (str): Organization name
            bucket (str): Bucket name for data storage
        """
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB client not available. Install with: pip install influxdb-client")
        
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = None
        self.write_api = None
        self.query_api = None
        
        self.logger = self._setup_logger()
        
        # Initialize connection
        self._connect()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the InfluxDB manager."""
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
    
    def _connect(self) -> bool:
        """
        Establish connection to InfluxDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            # Test connection
            if self.test_connection():
                self.logger.info("Successfully connected to InfluxDB")
                return True
            else:
                self.logger.error("Failed to connect to InfluxDB")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to InfluxDB: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test the InfluxDB connection.
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            # Simple query to test connection
            query = f'buckets() |> filter(fn: (r) => r.name == "{self.bucket}") |> limit(n:1)'
            result = self.query_api.query(query)
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def write_price_data(self, data: pd.DataFrame, measurement: str = "stock_price", 
                        ticker: str = "AAPL", tags: Optional[Dict] = None) -> bool:
        """
        Write OHLCV price data to InfluxDB.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and datetime index
            measurement (str): Measurement name in InfluxDB
            ticker (str): Stock ticker symbol
            tags (Dict, optional): Additional tags to add to the data points
            
        Returns:
            bool: True if write successful, False otherwise
        """
        try:
            if data.empty:
                self.logger.warning("Empty DataFrame provided, nothing to write")
                return False
            
            points = []
            base_tags = {"ticker": ticker}
            if tags:
                base_tags.update(tags)
            
            for timestamp, row in data.iterrows():
                # Convert timestamp to UTC if it's timezone-naive
                if timestamp.tz is None:
                    timestamp = timestamp.tz_localize('UTC')
                elif timestamp.tz != timezone.utc:
                    timestamp = timestamp.tz_convert('UTC')
                
                point = Point(measurement) \
                    .time(timestamp, WritePrecision.S)
                
                # Add tags
                for tag_key, tag_value in base_tags.items():
                    point = point.tag(tag_key, str(tag_value))
                
                # Add OHLCV fields
                if 'Open' in row and not pd.isna(row['Open']):
                    point = point.field("open", float(row['Open']))
                if 'High' in row and not pd.isna(row['High']):
                    point = point.field("high", float(row['High']))
                if 'Low' in row and not pd.isna(row['Low']):
                    point = point.field("low", float(row['Low']))
                if 'Close' in row and not pd.isna(row['Close']):
                    point = point.field("close", float(row['Close']))
                if 'Volume' in row and not pd.isna(row['Volume']):
                    point = point.field("volume", int(row['Volume']))
                
                # Add any additional numeric fields
                for col in row.index:
                    if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and pd.api.types.is_numeric_dtype(type(row[col])):
                        if not pd.isna(row[col]):
                            point = point.field(col.lower().replace(' ', '_'), float(row[col]))
                
                points.append(point)
            
            # Write points to InfluxDB
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            self.logger.info(f"Successfully wrote {len(points)} data points for {ticker}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing price data: {str(e)}")
            return False
    
    def query_price_data(self, ticker: str, start_time: str, end_time: str, 
                        measurement: str = "stock_price") -> Optional[pd.DataFrame]:
        """
        Query price data from InfluxDB.
        
        Args:
            ticker (str): Stock ticker symbol
            start_time (str): Start time in RFC3339 format (e.g., "2023-01-01T00:00:00Z")
            end_time (str): End time in RFC3339 format
            measurement (str): Measurement name in InfluxDB
            
        Returns:
            pd.DataFrame: DataFrame with queried data, None if error
        """
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_time}, stop: {end_time})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> filter(fn: (r) => r.ticker == "{ticker}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''
            
            result = self.query_api.query_data_frame(query)
            
            if result.empty:
                self.logger.warning(f"No data found for {ticker} between {start_time} and {end_time}")
                return None
            
            # Clean up the DataFrame
            if '_time' in result.columns:
                result.set_index('_time', inplace=True)
                result.index = pd.to_datetime(result.index)
            
            # Remove InfluxDB metadata columns
            metadata_cols = ['result', 'table', '_start', '_stop', '_measurement', 'ticker']
            result = result.drop(columns=[col for col in metadata_cols if col in result.columns])
            
            # Rename columns to standard OHLCV format
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            result = result.rename(columns=column_mapping)
            
            self.logger.info(f"Successfully queried {len(result)} records for {ticker}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error querying price data: {str(e)}")
            return None
    
    def write_features(self, data: pd.DataFrame, measurement: str = "features", 
                      ticker: str = "AAPL", tags: Optional[Dict] = None) -> bool:
        """
        Write feature data (technical indicators, etc.) to InfluxDB.
        
        Args:
            data (pd.DataFrame): DataFrame with feature data and datetime index
            measurement (str): Measurement name in InfluxDB
            ticker (str): Stock ticker symbol
            tags (Dict, optional): Additional tags to add to the data points
            
        Returns:
            bool: True if write successful, False otherwise
        """
        try:
            if data.empty:
                self.logger.warning("Empty DataFrame provided, nothing to write")
                return False
            
            points = []
            base_tags = {"ticker": ticker}
            if tags:
                base_tags.update(tags)
            
            for timestamp, row in data.iterrows():
                # Convert timestamp to UTC if needed
                if timestamp.tz is None:
                    timestamp = timestamp.tz_localize('UTC')
                elif timestamp.tz != timezone.utc:
                    timestamp = timestamp.tz_convert('UTC')
                
                point = Point(measurement).time(timestamp, WritePrecision.S)
                
                # Add tags
                for tag_key, tag_value in base_tags.items():
                    point = point.tag(tag_key, str(tag_value))
                
                # Add all numeric fields as features
                for col in row.index:
                    if pd.api.types.is_numeric_dtype(type(row[col])) and not pd.isna(row[col]):
                        field_name = col.lower().replace(' ', '_').replace('/', '_')
                        point = point.field(field_name, float(row[col]))
                
                points.append(point)
            
            # Write points to InfluxDB
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            self.logger.info(f"Successfully wrote {len(points)} feature points for {ticker}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing feature data: {str(e)}")
            return False
    
    def close(self):
        """Close the InfluxDB connection."""
        if self.client:
            self.client.close()
            self.logger.info("InfluxDB connection closed")


if __name__ == "__main__":
    # Example usage (requires InfluxDB server running)
    try:
        # Initialize manager (adjust connection parameters as needed)
        influx_manager = InfluxDBManager(
            url="http://localhost:8086",
            token="your-token-here",  # Replace with actual token
            org="apple_trading",
            bucket="financial_data"
        )
        
        # Test connection
        if influx_manager.test_connection():
            print("InfluxDB connection successful!")
        else:
            print("InfluxDB connection failed!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure InfluxDB is installed and running, and update connection parameters")
