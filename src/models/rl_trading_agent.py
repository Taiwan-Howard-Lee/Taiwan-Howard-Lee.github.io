#!/usr/bin/env python3
"""
Reinforcement Learning Trading Agent
Decision tree-based RL agent for stock trading using historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import json

class TradingDecisionTree:
    """
    Decision tree for trading decisions based on technical indicators
    """
    
    def __init__(self, config: Dict = None):
        """Initialize decision tree with configuration"""
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
    
    def _get_default_config(self) -> Dict:
        """Default decision tree configuration"""
        return {
            "momentum_thresholds": {
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "macd_bullish": 0,  # MACD line > signal line
                "strong_momentum": 0.6
            },
            "trend_thresholds": {
                "price_above_sma20": 1.02,  # 2% above SMA20
                "price_below_sma20": 0.98,  # 2% below SMA20
                "sma_trend": 0.01  # 1% SMA slope
            },
            "volume_thresholds": {
                "high_volume": 1.5,  # 50% above average
                "low_volume": 0.7   # 30% below average
            },
            "volatility_thresholds": {
                "high_volatility": 0.3,  # 30% annualized
                "low_volatility": 0.15   # 15% annualized
            },
            "position_sizing": {
                "max_position": 0.1,  # 10% of portfolio
                "min_position": 0.02,  # 2% of portfolio
                "confidence_scaling": True
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        return logger
    
    def make_decision(self, market_data: pd.Series) -> Dict[str, Any]:
        """
        Make trading decision based on market data
        
        Args:
            market_data (pd.Series): Single row of market data with indicators
            
        Returns:
            Dict: Trading decision with action, confidence, and position size
        """
        decision = {
            'action': 'hold',  # buy, sell, hold
            'confidence': 0.0,  # 0-1 confidence score
            'position_size': 0.0,  # fraction of portfolio
            'reasoning': [],
            'signals': {}
        }
        
        # Extract key indicators
        rsi = market_data.get('rsi_14', 50)
        macd_line = market_data.get('macd_line', 0)
        macd_signal = market_data.get('macd_signal', 0)
        price_vs_sma20 = market_data.get('price_vs_sma20', 1.0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        atr = market_data.get('atr_14', 0)
        historical_vol = market_data.get('historical_vol_20', 0.2)
        
        # Calculate individual signals
        momentum_signal = self._analyze_momentum(rsi, macd_line, macd_signal)
        trend_signal = self._analyze_trend(price_vs_sma20)
        volume_signal = self._analyze_volume(volume_ratio)
        volatility_signal = self._analyze_volatility(historical_vol)
        
        decision['signals'] = {
            'momentum': momentum_signal,
            'trend': trend_signal,
            'volume': volume_signal,
            'volatility': volatility_signal
        }
        
        # Decision tree logic
        buy_signals = 0
        sell_signals = 0
        confidence_factors = []
        
        # Momentum analysis
        if momentum_signal['direction'] == 'bullish':
            buy_signals += momentum_signal['strength']
            confidence_factors.append(momentum_signal['strength'])
            decision['reasoning'].append(f"Bullish momentum (RSI: {rsi:.1f}, MACD: {macd_line:.3f})")
        elif momentum_signal['direction'] == 'bearish':
            sell_signals += momentum_signal['strength']
            confidence_factors.append(momentum_signal['strength'])
            decision['reasoning'].append(f"Bearish momentum (RSI: {rsi:.1f}, MACD: {macd_line:.3f})")
        
        # Trend analysis
        if trend_signal['direction'] == 'bullish':
            buy_signals += trend_signal['strength']
            confidence_factors.append(trend_signal['strength'])
            decision['reasoning'].append(f"Bullish trend (Price vs SMA20: {price_vs_sma20:.3f})")
        elif trend_signal['direction'] == 'bearish':
            sell_signals += trend_signal['strength']
            confidence_factors.append(trend_signal['strength'])
            decision['reasoning'].append(f"Bearish trend (Price vs SMA20: {price_vs_sma20:.3f})")
        
        # Volume confirmation
        if volume_signal['strength'] > 0.5:
            if buy_signals > sell_signals:
                buy_signals += 0.5
                decision['reasoning'].append("High volume confirms bullish signal")
            elif sell_signals > buy_signals:
                sell_signals += 0.5
                decision['reasoning'].append("High volume confirms bearish signal")
        
        # Final decision
        net_signal = buy_signals - sell_signals
        
        if net_signal > 1.0:
            decision['action'] = 'buy'
            decision['confidence'] = min(net_signal / 3.0, 1.0)  # Scale to 0-1
        elif net_signal < -1.0:
            decision['action'] = 'sell'
            decision['confidence'] = min(abs(net_signal) / 3.0, 1.0)
        else:
            decision['action'] = 'hold'
            decision['confidence'] = 0.1  # Low confidence for hold
        
        # Position sizing based on confidence and volatility
        if decision['action'] in ['buy', 'sell']:
            base_size = self.config['position_sizing']['min_position']
            max_size = self.config['position_sizing']['max_position']
            
            # Scale by confidence
            confidence_multiplier = decision['confidence']
            
            # Reduce size for high volatility
            volatility_multiplier = max(0.5, 1.0 - (historical_vol - 0.2))
            
            decision['position_size'] = base_size + (max_size - base_size) * confidence_multiplier * volatility_multiplier
        
        return decision
    
    def _analyze_momentum(self, rsi: float, macd_line: float, macd_signal: float) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        signal = {'direction': 'neutral', 'strength': 0.0, 'components': {}}
        
        # RSI analysis
        if rsi < self.config['momentum_thresholds']['rsi_oversold']:
            signal['components']['rsi'] = 'oversold_bullish'
            signal['strength'] += 0.7
        elif rsi > self.config['momentum_thresholds']['rsi_overbought']:
            signal['components']['rsi'] = 'overbought_bearish'
            signal['strength'] -= 0.7
        
        # MACD analysis
        macd_diff = macd_line - macd_signal
        if macd_diff > 0:
            signal['components']['macd'] = 'bullish'
            signal['strength'] += 0.5
        elif macd_diff < 0:
            signal['components']['macd'] = 'bearish'
            signal['strength'] -= 0.5
        
        # Determine overall direction
        if signal['strength'] > 0.3:
            signal['direction'] = 'bullish'
        elif signal['strength'] < -0.3:
            signal['direction'] = 'bearish'
            signal['strength'] = abs(signal['strength'])
        else:
            signal['strength'] = 0.1
        
        return signal
    
    def _analyze_trend(self, price_vs_sma20: float) -> Dict[str, Any]:
        """Analyze trend indicators"""
        signal = {'direction': 'neutral', 'strength': 0.0}
        
        if price_vs_sma20 > self.config['trend_thresholds']['price_above_sma20']:
            signal['direction'] = 'bullish'
            signal['strength'] = min((price_vs_sma20 - 1.0) * 10, 1.0)  # Scale strength
        elif price_vs_sma20 < self.config['trend_thresholds']['price_below_sma20']:
            signal['direction'] = 'bearish'
            signal['strength'] = min((1.0 - price_vs_sma20) * 10, 1.0)
        else:
            signal['strength'] = 0.1
        
        return signal
    
    def _analyze_volume(self, volume_ratio: float) -> Dict[str, Any]:
        """Analyze volume indicators"""
        signal = {'direction': 'neutral', 'strength': 0.0}
        
        if volume_ratio > self.config['volume_thresholds']['high_volume']:
            signal['direction'] = 'confirming'
            signal['strength'] = min((volume_ratio - 1.0), 1.0)
        elif volume_ratio < self.config['volume_thresholds']['low_volume']:
            signal['direction'] = 'weak'
            signal['strength'] = max(0.1, volume_ratio)
        else:
            signal['strength'] = 0.5
        
        return signal
    
    def _analyze_volatility(self, historical_vol: float) -> Dict[str, Any]:
        """Analyze volatility indicators"""
        signal = {'direction': 'neutral', 'strength': 0.0}
        
        if historical_vol > self.config['volatility_thresholds']['high_volatility']:
            signal['direction'] = 'high_risk'
            signal['strength'] = min(historical_vol / 0.5, 1.0)
        elif historical_vol < self.config['volatility_thresholds']['low_volatility']:
            signal['direction'] = 'low_risk'
            signal['strength'] = max(0.2, 1.0 - historical_vol / 0.3)
        else:
            signal['strength'] = 0.5
        
        return signal

class RLTradingEnvironment:
    """
    Reinforcement Learning environment for trading
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        """
        Initialize trading environment
        
        Args:
            data (pd.DataFrame): Historical market data
            initial_capital (float): Starting capital
        """
        self.data = data.sort_index()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.current_day = 0
        self.positions = {}  # {symbol: shares}
        self.portfolio_history = []
        self.decision_tree = TradingDecisionTree()
        self.logger = self._setup_logger()
        
        # Get unique symbols and dates
        self.symbols = self.data['symbol'].unique()
        self.dates = self.data.index.unique()
        
        self.logger.info(f"🏗️ RL Environment initialized:")
        self.logger.info(f"   📊 Data: {len(self.data)} records")
        self.logger.info(f"   📅 Date range: {self.dates[0]} to {self.dates[-1]}")
        self.logger.info(f"   🏢 Symbols: {len(self.symbols)}")
        self.logger.info(f"   💰 Initial capital: ${initial_capital:,.2f}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        return logger
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state"""
        self.current_capital = self.initial_capital
        self.current_day = 0
        self.positions = {}
        self.portfolio_history = []
        
        return self._get_current_state()
    
    def step(self, actions: Dict[str, Dict]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            actions (Dict): Actions for each symbol {symbol: decision_dict}
            
        Returns:
            Tuple: (next_state, reward, done, info)
        """
        if self.current_day >= len(self.dates) - 1:
            return self._get_current_state(), 0, True, {'reason': 'end_of_data'}
        
        current_date = self.dates[self.current_day]
        next_date = self.dates[self.current_day + 1]
        
        # Execute trades
        trade_info = self._execute_trades(actions, current_date)
        
        # Move to next day
        self.current_day += 1
        
        # Calculate portfolio value and reward
        portfolio_value = self._calculate_portfolio_value(next_date)
        reward = self._calculate_reward(portfolio_value)
        
        # Record portfolio history
        self.portfolio_history.append({
            'date': next_date,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital,
            'positions': self.positions.copy(),
            'daily_return': (portfolio_value - self.initial_capital) / self.initial_capital if self.portfolio_history else 0
        })
        
        # Check if done
        done = self.current_day >= len(self.dates) - 1
        
        info = {
            'portfolio_value': portfolio_value,
            'trades_executed': trade_info,
            'current_date': next_date
        }
        
        return self._get_current_state(), reward, done, info
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current market state"""
        if self.current_day >= len(self.dates):
            return {}
        
        current_date = self.dates[self.current_day]
        current_data = self.data[self.data.index == current_date]
        
        state = {}
        for _, row in current_data.iterrows():
            symbol = row['symbol']
            state[symbol] = row.to_dict()
        
        return state
    
    def _execute_trades(self, actions: Dict[str, Dict], date: pd.Timestamp) -> List[Dict]:
        """Execute trading actions"""
        trades = []
        
        for symbol, decision in actions.items():
            if symbol not in self.data['symbol'].values:
                continue
            
            # Get current price
            symbol_data = self.data[(self.data.index == date) & (self.data['symbol'] == symbol)]
            if symbol_data.empty:
                continue
            
            current_price = symbol_data['Close'].iloc[0]
            action = decision.get('action', 'hold')
            position_size = decision.get('position_size', 0)
            
            if action == 'buy' and position_size > 0:
                # Calculate shares to buy
                capital_to_invest = self.current_capital * position_size
                shares_to_buy = int(capital_to_invest / current_price)
                
                if shares_to_buy > 0 and capital_to_invest <= self.current_capital:
                    cost = shares_to_buy * current_price
                    self.current_capital -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                    
                    trades.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost,
                        'date': date
                    })
            
            elif action == 'sell' and symbol in self.positions and self.positions[symbol] > 0:
                # Sell all shares for simplicity
                shares_to_sell = self.positions[symbol]
                proceeds = shares_to_sell * current_price
                self.current_capital += proceeds
                del self.positions[symbol]
                
                trades.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'proceeds': proceeds,
                    'date': date
                })
        
        return trades
    
    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """Calculate total portfolio value"""
        total_value = self.current_capital
        
        for symbol, shares in self.positions.items():
            symbol_data = self.data[(self.data.index == date) & (self.data['symbol'] == symbol)]
            if not symbol_data.empty:
                current_price = symbol_data['Close'].iloc[0]
                total_value += shares * current_price
        
        return total_value
    
    def _calculate_reward(self, portfolio_value: float) -> float:
        """Calculate reward for RL agent"""
        if not self.portfolio_history:
            return 0
        
        # Daily return
        previous_value = self.portfolio_history[-1]['portfolio_value'] if self.portfolio_history else self.initial_capital
        daily_return = (portfolio_value - previous_value) / previous_value
        
        # Base reward is the daily return
        reward = daily_return * 100  # Scale up
        
        # Penalty for large drawdowns
        if daily_return < -0.05:  # More than 5% loss
            reward -= abs(daily_return) * 50  # Extra penalty
        
        # Bonus for consistent gains
        if len(self.portfolio_history) >= 5:
            recent_returns = [h['daily_return'] for h in self.portfolio_history[-5:]]
            if all(r > 0 for r in recent_returns):
                reward += 5  # Consistency bonus
        
        return reward

def main():
    """Test the RL trading agent"""
    print("🤖 Testing RL Trading Agent")
    print("=" * 30)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    sample_data = []
    for date in dates:
        for symbol in ['AAPL', 'MSFT']:
            # Generate realistic market data
            price = 150 + np.random.randn() * 5
            sample_data.append({
                'symbol': symbol,
                'Close': price,
                'rsi_14': 50 + np.random.randn() * 15,
                'macd_line': np.random.randn() * 2,
                'macd_signal': np.random.randn() * 2,
                'price_vs_sma20': 1.0 + np.random.randn() * 0.05,
                'volume_ratio': 1.0 + np.random.randn() * 0.3,
                'historical_vol_20': 0.2 + np.random.randn() * 0.05
            })
    
    df = pd.DataFrame(sample_data)
    df.index = pd.to_datetime([dates[i//2] for i in range(len(sample_data))])
    
    # Test decision tree
    decision_tree = TradingDecisionTree()
    sample_row = df.iloc[0]
    decision = decision_tree.make_decision(sample_row)
    
    print(f"📊 Sample Decision:")
    print(f"   Action: {decision['action']}")
    print(f"   Confidence: {decision['confidence']:.3f}")
    print(f"   Position Size: {decision['position_size']:.3f}")
    print(f"   Reasoning: {decision['reasoning']}")
    
    # Test RL environment
    env = RLTradingEnvironment(df, initial_capital=100000)
    state = env.reset()
    
    print(f"\n🏗️ RL Environment:")
    print(f"   Symbols: {len(state)}")
    print(f"   Initial capital: ${env.initial_capital:,.2f}")

if __name__ == "__main__":
    main()
