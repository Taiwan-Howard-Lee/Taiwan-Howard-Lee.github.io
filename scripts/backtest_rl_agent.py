#!/usr/bin/env python3
"""
RL Trading Agent Backtesting Script
Tests the reinforcement learning trading agent on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import logging
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.rl_trading_agent import TradingDecisionTree, RLTradingEnvironment

class RLBacktester:
    """
    Comprehensive backtesting framework for RL trading agent
    """
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize the backtester"""
        self.initial_capital = initial_capital
        self.logger = self._setup_logger()
        
        # Performance tracking
        self.results = {
            'trades': [],
            'daily_portfolio': [],
            'performance_metrics': {},
            'decisions_log': []
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - BACKTESTER - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load historical data for backtesting
        
        Args:
            data_path (str): Path to data file
            
        Returns:
            pd.DataFrame: Historical market data
        """
        if data_path is None:
            # Try to find the most recent Yahoo Finance dataset
            yahoo_data_dir = project_root / 'data' / 'yahoo_mvp'
            if yahoo_data_dir.exists():
                dataset_files = list(yahoo_data_dir.glob('yahoo_rl_training_dataset_*.csv'))
                if dataset_files:
                    data_path = max(dataset_files, key=lambda x: x.stat().st_mtime)
                    self.logger.info(f"📊 Using Yahoo Finance dataset: {data_path.name}")
        
        if data_path is None:
            self.logger.error("❌ No data file found")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            self.logger.info(f"✅ Loaded data: {len(data)} records, {data['symbol'].nunique()} symbols")
            self.logger.info(f"📅 Date range: {data.index.min().date()} to {data.index.max().date()}")
            
            return data
        except Exception as e:
            self.logger.error(f"❌ Failed to load data: {str(e)}")
            return pd.DataFrame()
    
    def run_backtest(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest
        
        Args:
            data (pd.DataFrame): Historical market data
            start_date (str): Start date for backtest
            end_date (str): End date for backtest
            
        Returns:
            Dict: Backtest results
        """
        self.logger.info("🚀 Starting RL Agent Backtest")
        
        # Filter data by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if data.empty:
            self.logger.error("❌ No data available for backtesting")
            return {}
        
        self.logger.info(f"📊 Backtesting period: {data.index.min().date()} to {data.index.max().date()}")
        
        # Initialize RL environment
        env = RLTradingEnvironment(data, initial_capital=self.initial_capital)
        decision_tree = TradingDecisionTree()
        
        # Reset environment
        state = env.reset()
        done = False
        step_count = 0
        
        self.logger.info(f"💰 Starting capital: ${self.initial_capital:,.2f}")
        
        # Run simulation
        while not done:
            step_count += 1
            
            # Make decisions for each symbol in current state
            actions = {}
            for symbol, symbol_data in state.items():
                decision = decision_tree.make_decision(pd.Series(symbol_data))
                actions[symbol] = decision
                
                # Log significant decisions
                if decision['action'] != 'hold':
                    self.results['decisions_log'].append({
                        'date': env.dates[env.current_day],
                        'symbol': symbol,
                        'action': decision['action'],
                        'confidence': decision['confidence'],
                        'position_size': decision['position_size'],
                        'reasoning': decision['reasoning'][:2]  # First 2 reasons
                    })
            
            # Execute step
            next_state, reward, done, info = env.step(actions)
            
            # Record portfolio performance
            portfolio_value = info.get('portfolio_value', self.initial_capital)
            self.results['daily_portfolio'].append({
                'date': info.get('current_date'),
                'portfolio_value': portfolio_value,
                'daily_return': (portfolio_value - self.initial_capital) / self.initial_capital,
                'cash': env.current_capital,
                'positions_count': len(env.positions)
            })
            
            # Record trades
            trades = info.get('trades_executed', [])
            self.results['trades'].extend(trades)
            
            # Progress logging
            if step_count % 10 == 0:
                self.logger.info(f"📈 Step {step_count}: Portfolio ${portfolio_value:,.2f} ({(portfolio_value/self.initial_capital-1)*100:+.2f}%)")
            
            state = next_state
        
        # Calculate final performance metrics
        final_value = env._calculate_portfolio_value(env.dates[-1])
        self.results['performance_metrics'] = self._calculate_performance_metrics(final_value)
        
        self.logger.info("🎉 Backtest completed!")
        self._print_results()
        
        return self.results
    
    def _calculate_performance_metrics(self, final_value: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        portfolio_df = pd.DataFrame(self.results['daily_portfolio'])
        
        if portfolio_df.empty:
            return {}
        
        # Basic metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Daily returns
        portfolio_df['daily_pct_change'] = portfolio_df['portfolio_value'].pct_change()
        daily_returns = portfolio_df['daily_pct_change'].dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Drawdown analysis
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Trade analysis
        trades_df = pd.DataFrame(self.results['trades'])
        total_trades = len(trades_df)
        
        # Win rate calculation
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            # Simple win rate based on profitable trades
            profitable_trades = 0
            total_trade_pairs = 0
            
            for symbol in trades_df['symbol'].unique():
                symbol_trades = trades_df[trades_df['symbol'] == symbol].sort_values('date')
                buy_price = None
                
                for _, trade in symbol_trades.iterrows():
                    if trade['action'] == 'buy' and buy_price is None:
                        buy_price = trade['price']
                    elif trade['action'] == 'sell' and buy_price is not None:
                        if trade['price'] > buy_price:
                            profitable_trades += 1
                        total_trade_pairs += 1
                        buy_price = None
            
            win_rate = profitable_trades / total_trade_pairs if total_trade_pairs > 0 else 0
        else:
            win_rate = 0
        
        # Days in market
        trading_days = len(portfolio_df)
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_value': final_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'trading_days': trading_days,
            'avg_daily_return': daily_returns.mean(),
            'best_day': daily_returns.max(),
            'worst_day': daily_returns.min()
        }
    
    def _print_results(self):
        """Print comprehensive backtest results"""
        metrics = self.results['performance_metrics']
        
        print("\n" + "="*60)
        print("🎯 RL TRADING AGENT - BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n💰 PERFORMANCE SUMMARY:")
        print(f"   Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"   Final Value:         ${metrics.get('final_value', 0):,.2f}")
        print(f"   Total Return:        {metrics.get('total_return_pct', 0):+.2f}%")
        print(f"   Trading Days:        {metrics.get('trading_days', 0)}")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   Volatility:          {metrics.get('volatility', 0)*100:.2f}%")
        print(f"   Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"   Best Day:            {metrics.get('best_day', 0)*100:+.2f}%")
        print(f"   Worst Day:           {metrics.get('worst_day', 0)*100:+.2f}%")
        
        print(f"\n🎯 TRADING ACTIVITY:")
        print(f"   Total Trades:        {metrics.get('total_trades', 0)}")
        print(f"   Win Rate:            {metrics.get('win_rate_pct', 0):.1f}%")
        print(f"   Avg Daily Return:    {metrics.get('avg_daily_return', 0)*100:+.4f}%")
        
        # Show recent decisions
        if self.results['decisions_log']:
            print(f"\n🧠 RECENT DECISIONS (Last 5):")
            for decision in self.results['decisions_log'][-5:]:
                print(f"   {decision['date'].date()} {decision['symbol']}: {decision['action'].upper()} "
                      f"(conf: {decision['confidence']:.2f}, size: {decision['position_size']:.2f})")
        
        # Benchmark comparison (assume 7% annual return for S&P 500)
        trading_days = metrics.get('trading_days', 252)
        benchmark_return = (1.07 ** (trading_days / 252)) - 1
        alpha = metrics.get('total_return', 0) - benchmark_return
        
        print(f"\n📈 BENCHMARK COMPARISON:")
        print(f"   S&P 500 (est):       {benchmark_return*100:+.2f}%")
        print(f"   Alpha:               {alpha*100:+.2f}%")
        
        print("="*60)
    
    def save_results(self, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rl_backtest_results_{timestamp}.json"
        
        results_dir = project_root / 'data' / 'backtest_results'
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to: {filepath}")

def main():
    """Run the RL agent backtest"""
    print("🤖 RL Trading Agent - Backtesting")
    print("=" * 35)
    
    # Initialize backtester
    backtester = RLBacktester(initial_capital=100000)
    
    # Load data
    data = backtester.load_data()
    
    if data.empty:
        print("❌ No data available for backtesting")
        return
    
    # Run backtest
    results = backtester.run_backtest(data)
    
    if results:
        # Save results
        backtester.save_results()
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Optimize parameters: python3 scripts/optimize_parameters.py")
        print(f"   2. Test different strategies: Modify decision tree rules")
        print(f"   3. Add more data sources: Run Alpha Vantage collector")
        print(f"   4. Deploy paper trading: python3 scripts/paper_trading.py")

if __name__ == "__main__":
    main()
