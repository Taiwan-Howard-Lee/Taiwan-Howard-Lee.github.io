# 📊 Technical Analysis Data Schema

## 🎯 **Comprehensive Framework Overview**

### **Primary Classification Dimensions**

1. **📈 Indicator Categories** (6 main types)
2. **🎯 Scope Level** (Company vs Market)
3. **⏱️ Time Horizon** (Short, Medium, Long-term)
4. **📊 Data Requirements** (Price, Volume, Market breadth)

## 🏗️ **Schema Structure**

### **Level 1: Scope Classification**

```json
{
  "company_level": {
    "description": "Metrics calculated for individual securities",
    "data_source": "OHLCV data for specific ticker",
    "examples": ["AAPL RSI", "MSFT MACD", "TSLA Bollinger Bands"]
  },
  "market_level": {
    "description": "Metrics calculated for entire market/sector",
    "data_source": "Multiple tickers, indices, breadth data",
    "examples": ["S&P 500 VIX", "Advance/Decline Ratio", "Market Breadth"]
  }
}
```

### **Level 2: Technical Categories (6 Primary)**

#### **1. 📊 VOLUME**
```json
{
  "volume": {
    "description": "Trading activity and liquidity metrics",
    "company_level": {
      "basic": ["Volume", "Volume SMA", "Volume Ratio"],
      "advanced": ["OBV", "VWAP", "Volume Profile", "Accumulation/Distribution"],
      "sophisticated": ["Chaikin Money Flow", "Ease of Movement", "Force Index"]
    },
    "market_level": {
      "basic": ["Total Market Volume", "Up/Down Volume Ratio"],
      "advanced": ["Arms Index (TRIN)", "Volume Oscillator"],
      "sophisticated": ["Klinger Oscillator", "Volume-Price Trend"]
    }
  }
}
```

#### **2. 🚀 MOMENTUM**
```json
{
  "momentum": {
    "description": "Rate of price change and momentum indicators",
    "company_level": {
      "basic": ["RSI", "Stochastic", "Williams %R"],
      "advanced": ["MACD", "ROC", "Momentum Oscillator"],
      "sophisticated": ["Commodity Channel Index", "Ultimate Oscillator", "Aroon"]
    },
    "market_level": {
      "basic": ["Market RSI", "Sector Rotation Momentum"],
      "advanced": ["McClellan Oscillator", "Summation Index"],
      "sophisticated": ["Market Momentum Model", "Cross-Asset Momentum"]
    }
  }
}
```

#### **3. 📈 TREND**
```json
{
  "trend": {
    "description": "Direction and strength of price trends",
    "company_level": {
      "basic": ["SMA", "EMA", "Price vs MA"],
      "advanced": ["Bollinger Bands", "Keltner Channels", "PSAR"],
      "sophisticated": ["Ichimoku Cloud", "SuperTrend", "Adaptive Moving Averages"]
    },
    "market_level": {
      "basic": ["Market Trend Direction", "Sector Trends"],
      "advanced": ["Advance/Decline Line", "New Highs/Lows"],
      "sophisticated": ["Market Regime Detection", "Trend Strength Index"]
    }
  }
}
```

#### **4. 🌊 VOLATILITY**
```json
{
  "volatility": {
    "description": "Price variability and risk measures",
    "company_level": {
      "basic": ["Historical Volatility", "True Range", "ATR"],
      "advanced": ["Bollinger Band Width", "Volatility Ratio"],
      "sophisticated": ["GARCH Models", "Realized Volatility", "Volatility Surface"]
    },
    "market_level": {
      "basic": ["VIX", "Market Volatility"],
      "advanced": ["Term Structure of Volatility", "Volatility Skew"],
      "sophisticated": ["Cross-Asset Volatility", "Volatility Regime Models"]
    }
  }
}
```

#### **5. 🌐 BREADTH**
```json
{
  "breadth": {
    "description": "Market participation and internal strength",
    "company_level": {
      "basic": ["Relative Strength vs Market"],
      "advanced": ["Beta", "Correlation with Market"],
      "sophisticated": ["Sector Relative Performance", "Factor Exposure"]
    },
    "market_level": {
      "basic": ["Advance/Decline Ratio", "Up/Down Volume"],
      "advanced": ["McClellan Summation", "High-Low Index"],
      "sophisticated": ["Breadth Thrust", "Participation Rate", "Sector Rotation"]
    }
  }
}
```

#### **6. 🎯 SUPPORT & RESISTANCE**
```json
{
  "support_resistance": {
    "description": "Key price levels and psychological barriers",
    "company_level": {
      "basic": ["Pivot Points", "Previous High/Low", "Round Numbers"],
      "advanced": ["Fibonacci Retracements", "Support/Resistance Zones"],
      "sophisticated": ["Dynamic S/R", "Volume-Weighted Levels", "Order Flow Levels"]
    },
    "market_level": {
      "basic": ["Index Support/Resistance", "Sector Level Analysis"],
      "advanced": ["Market Structure Levels", "Institutional Levels"],
      "sophisticated": ["Cross-Market S/R", "Intermarket Analysis"]
    }
  }
}
```

## 🏗️ **Implementation Schema**

### **Database Structure**
```sql
-- Main metrics table
CREATE TABLE technical_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    symbol VARCHAR(10),           -- NULL for market-level metrics
    scope_level VARCHAR(20),      -- 'company' or 'market'
    category VARCHAR(30),         -- volume, momentum, trend, volatility, breadth, support_resistance
    subcategory VARCHAR(50),      -- basic, advanced, sophisticated
    metric_name VARCHAR(100),     -- RSI, MACD, etc.
    metric_value DECIMAL(15,6),
    time_horizon VARCHAR(20),     -- short, medium, long
    calculation_params JSON,      -- Parameters used for calculation
    data_quality_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Market context table
CREATE TABLE market_context (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    market_regime VARCHAR(50),    -- bull, bear, sideways, volatile
    sector_rotation JSON,         -- Current sector leadership
    volatility_regime VARCHAR(30), -- low, normal, high, extreme
    breadth_condition VARCHAR(30), -- strong, weak, diverging
    created_at TIMESTAMP DEFAULT NOW()
);
```

### **JSON Configuration Schema**
```json
{
  "technical_analysis_config": {
    "company_level": {
      "volume": {
        "basic": {
          "volume_sma": {"period": 20},
          "volume_ratio": {"period": 10}
        },
        "advanced": {
          "obv": {"enabled": true},
          "vwap": {"enabled": true},
          "ad_line": {"enabled": true}
        }
      },
      "momentum": {
        "basic": {
          "rsi": {"period": 14},
          "stochastic": {"k_period": 14, "d_period": 3},
          "williams_r": {"period": 14}
        },
        "advanced": {
          "macd": {"fast": 12, "slow": 26, "signal": 9},
          "roc": {"period": 12}
        }
      }
    },
    "market_level": {
      "breadth": {
        "basic": {
          "advance_decline_ratio": {"enabled": true},
          "up_down_volume": {"enabled": true}
        },
        "advanced": {
          "mcclellan_oscillator": {"enabled": true},
          "high_low_index": {"period": 10}
        }
      }
    }
  }
}
```

## 🎯 **Discussion Points**

### **1. Data Requirements**
- **Company Level**: Need OHLCV for individual stocks
- **Market Level**: Need broader market data (indices, sector data, breadth metrics)
- **Cross-Reference**: Some metrics need both (e.g., relative strength)

### **2. Calculation Complexity**
- **Basic**: Simple calculations, fast computation
- **Advanced**: More complex, moderate computation
- **Sophisticated**: Complex models, intensive computation

### **3. Time Horizons**
- **Short-term**: 1-20 periods (intraday to weeks)
- **Medium-term**: 20-100 periods (weeks to months)
- **Long-term**: 100+ periods (months to years)

### **4. Implementation Priorities**
1. **Phase 1**: Company-level basic indicators (RSI, MACD, SMA)
2. **Phase 2**: Company-level advanced indicators (Bollinger Bands, Stochastic)
3. **Phase 3**: Market-level breadth indicators
4. **Phase 4**: Sophisticated models and cross-asset analysis

## 🚀 **Questions for Discussion**

1. **Data Sources**: Do we have access to market breadth data beyond individual stocks?
2. **Computation Resources**: Should we prioritize basic indicators first?
3. **Storage Strategy**: Real-time calculation vs pre-computed storage?
4. **Market Coverage**: Focus on US markets or include global markets?
5. **Update Frequency**: Real-time, hourly, daily updates?

## 📊 **Proposed Next Steps**

1. **Validate Schema**: Review and refine the 6-category structure
2. **Data Audit**: Identify what market-level data we can access
3. **Implementation Plan**: Prioritize indicators by complexity and value
4. **Testing Framework**: Create validation tests for each category
5. **Performance Optimization**: Plan for efficient calculation and storage

**What aspects would you like to dive deeper into first?**
