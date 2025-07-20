# 🔄 Sector Rotation - Market Leadership Analysis

## 🎯 **What is Sector Rotation?**

Sector rotation is the movement of investment money from one industry sector to another as investors and traders anticipate the next stage of the economic cycle.

### **Key Concept**
Different sectors perform better at different stages of the economic cycle:
- **Early Recovery**: Technology, Consumer Discretionary
- **Mid Recovery**: Industrials, Materials  
- **Late Recovery**: Energy, Financials
- **Recession**: Utilities, Consumer Staples, Healthcare

## 📊 **US Market Sectors (11 GICS Sectors)**

```
1. 🏭 Industrials (XLI)          - Manufacturing, transportation
2. 💻 Technology (XLK)           - Software, hardware, semiconductors  
3. 🏥 Healthcare (XLV)           - Pharmaceuticals, medical devices
4. 🏦 Financials (XLF)           - Banks, insurance, real estate
5. 🛒 Consumer Discretionary (XLY) - Retail, automotive, entertainment
6. 🥫 Consumer Staples (XLP)     - Food, beverages, household products
7. ⚡ Energy (XLE)               - Oil, gas, renewable energy
8. 🏗️ Materials (XLB)            - Chemicals, metals, mining
9. 📞 Communication (XLC)        - Telecom, media, internet
10. 🏠 Real Estate (XLRE)        - REITs, property companies
11. ⚡ Utilities (XLU)           - Electric, gas, water utilities
```

## 📈 **Sector Rotation Data Schema**

### **Daily Sector Performance Metrics**
```json
{
  "date": "2024-01-15",
  "sector_performance": {
    "XLK": {
      "daily_return": 0.0234,
      "relative_strength_vs_spy": 1.15,
      "volume_ratio": 1.23,
      "momentum_score": 0.78,
      "rank": 1
    },
    "XLF": {
      "daily_return": 0.0156,
      "relative_strength_vs_spy": 1.08,
      "volume_ratio": 1.45,
      "momentum_score": 0.65,
      "rank": 2
    }
  },
  "rotation_signals": {
    "leading_sectors": ["XLK", "XLF", "XLI"],
    "lagging_sectors": ["XLU", "XLP", "XLRE"],
    "rotation_strength": 0.67,
    "market_regime": "risk_on"
  }
}
```

## 🎯 **MVP Implementation Plan**

### **Phase 1: Company-Level Schema (Focus)**
```python
company_metrics = {
    "symbol": "AAPL",
    "date": "2024-01-15", 
    "sector": "XLK",  # Technology
    "categories": {
        "volume": {
            "volume_sma_20": 45000000,
            "volume_ratio": 1.23,
            "obv": 1250000000
        },
        "momentum": {
            "rsi_14": 67.5,
            "macd_line": 2.34,
            "stochastic_k": 78.2
        },
        "trend": {
            "sma_20": 185.50,
            "sma_50": 182.30,
            "price_vs_sma20": 1.02
        },
        "volatility": {
            "atr_14": 3.45,
            "historical_vol_20": 0.28
        },
        "support_resistance": {
            "pivot_point": 187.25,
            "support_1": 184.50,
            "resistance_1": 190.00
        }
    },
    "sector_context": {
        "sector_etf": "XLK",
        "relative_strength_vs_sector": 1.05,
        "sector_rank": 3  # Rank within XLK holdings
    }
}
```

## 🏗️ **Data Collection Strategy**

### **Data Sources for MVP**
1. **Individual Stocks**: Polygon.io (OHLCV)
2. **Sector ETFs**: Same source (XLK, XLF, XLI, etc.)
3. **Market Index**: SPY for market comparison

### **Collection Pipeline**
```python
# Daily collection targets
tickers_to_collect = [
    # Individual stocks (start with major ones)
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
    "JPM", "BAC", "WFC",  # Financials
    "XOM", "CVX",         # Energy
    
    # Sector ETFs for context
    "XLK", "XLF", "XLI", "XLV", "XLY", "XLP", "XLE", "XLB", "XLC", "XLRE", "XLU",
    
    # Market benchmark
    "SPY"
]
```

## 🤖 **Reinforcement Learning Framework**

### **Decision Tree Structure**
```python
class TradingDecisionTree:
    def __init__(self):
        self.decision_nodes = {
            "market_regime": {
                "bull_market": "aggressive_growth",
                "bear_market": "defensive", 
                "sideways": "sector_rotation"
            },
            "sector_strength": {
                "strong_sector": "buy_leaders",
                "weak_sector": "avoid_or_short",
                "rotating": "momentum_play"
            },
            "individual_signals": {
                "strong_momentum": "buy",
                "weak_momentum": "sell",
                "neutral": "hold"
            }
        }
    
    def make_decision(self, market_data, sector_data, stock_data):
        # Decision logic here
        return {"action": "buy", "confidence": 0.75, "position_size": 0.1}
```

### **RL Environment Setup**
```python
class TradingEnvironment:
    def __init__(self, historical_data, initial_capital=100000):
        self.data = historical_data
        self.capital = initial_capital
        self.current_day = 0
        self.positions = {}
        
    def step(self, action):
        # Execute trade
        # Calculate reward/penalty
        # Update portfolio
        reward = self.calculate_reward(action)
        return next_state, reward, done, info
    
    def calculate_reward(self, action):
        # Reward based on:
        # - Profit/Loss from trade
        # - Risk-adjusted returns
        # - Drawdown penalties
        return reward
```

## 📊 **MVP Database Schema**

### **Simplified Tables**
```sql
-- Daily stock data with calculated indicators
CREATE TABLE daily_stock_metrics (
    id SERIAL PRIMARY KEY,
    date DATE,
    symbol VARCHAR(10),
    sector VARCHAR(10),
    
    -- Price data
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2), 
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    
    -- Volume indicators
    volume_sma_20 BIGINT,
    volume_ratio DECIMAL(5,2),
    obv BIGINT,
    
    -- Momentum indicators  
    rsi_14 DECIMAL(5,2),
    macd_line DECIMAL(8,4),
    stochastic_k DECIMAL(5,2),
    
    -- Trend indicators
    sma_20 DECIMAL(10,2),
    sma_50 DECIMAL(10,2),
    price_vs_sma20 DECIMAL(5,4),
    
    -- Volatility indicators
    atr_14 DECIMAL(8,4),
    historical_vol_20 DECIMAL(6,4),
    
    -- Support/Resistance
    pivot_point DECIMAL(10,2),
    support_1 DECIMAL(10,2),
    resistance_1 DECIMAL(10,2),
    
    -- Sector context
    relative_strength_vs_sector DECIMAL(6,4),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- RL training results
CREATE TABLE rl_training_results (
    id SERIAL PRIMARY KEY,
    training_date DATE,
    episode INTEGER,
    total_reward DECIMAL(15,2),
    final_portfolio_value DECIMAL(15,2),
    max_drawdown DECIMAL(6,4),
    sharpe_ratio DECIMAL(6,4),
    decisions_made INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🚀 **Implementation Steps**

### **Week 1: Data Collection & Schema**
1. ✅ Set up enhanced data collection for 20-30 key stocks + sector ETFs
2. ✅ Implement company-level technical indicators
3. ✅ Create normalized data storage schema
4. ✅ Build data validation and cleaning pipeline

### **Week 2: Decision Tree & RL Framework**  
1. 📋 Design trading decision tree logic
2. 📋 Implement RL environment with historical data
3. 📋 Create reward/penalty system
4. 📋 Build backtesting framework

### **Week 3: Training & Optimization**
1. 📋 Train RL agent on 2+ years of historical data
2. 📋 Optimize decision parameters
3. 📋 Validate on out-of-sample data
4. 📋 Deploy live paper trading

## 🎯 **Key Questions for Next Steps**

1. **Stock Universe**: Start with S&P 100 or focus on specific sectors?
2. **Decision Frequency**: Daily decisions or intraday?
3. **Position Sizing**: Fixed size or dynamic based on confidence?
4. **Risk Management**: Stop-loss rules in the decision tree?
5. **Benchmark**: Compare against buy-and-hold SPY?

**Ready to start with the data collection pipeline for this MVP approach?**
