# 🍎 Apple ML Trading System

A comprehensive machine learning trading system for Apple (AAPL) stock with advanced features, dual-API data collection, technical analysis, and a professional web dashboard.

## 📋 Project Overview

This project implements a complete ML-driven trading system that:
- **Dual-API Data Collection**: Polygon.io (US market data) + Trading Economics (global economic data)
- **Continuous Data Harvesting**: Maximizes API rate limits with intelligent 24/7 collection
- **Advanced Technical Analysis**: RSI, MACD, Bollinger Bands, Stochastic Oscillator
- **Real-time Dashboard**: Professional GitHub Pages interface with live data
- **Cross-Data Validation**: Ensures data accuracy across multiple sources
- **Comprehensive Storage**: JSON, CSV, and time-series database support

## 🏗️ Project Structure

```
apple_ml_trading/
├── data/
│   ├── raw/                 # Raw data files
│   ├── processed/           # Clean, feature-engineered data
│   ├── external/            # Economic, sentiment data
│   └── influxdb/            # InfluxDB data files
├── src/
│   ├── data_collection/     # Data gathering modules
│   ├── feature_engineering/ # Feature creation
│   │   ├── regime_detection.py    # HMM regime models
│   │   ├── volatility_models.py   # GARCH implementations
│   │   └── transformers.py        # Transformer features
│   ├── models/              # ML model implementations
│   │   ├── traditional/     # RF, XGB, LGBM
│   │   ├── deep_learning/   # LSTM, Transformers
│   │   └── ensemble/        # Model combinations
│   ├── backtesting/         # Trading simulation
│   ├── risk_metrics/        # Advanced risk calculations
│   └── utils/               # Helper functions
├── dashboard/               # Web frontend
│   ├── app.py              # Main Streamlit app
│   ├── pages/              # Dashboard pages
│   ├── components/         # Reusable UI components
│   └── assets/             # CSS, images, etc.
├── notebooks/               # Jupyter notebooks for research
├── config/                  # Configuration files
├── tests/                   # Unit tests
├── logs/                    # Application logs
└── scripts/                 # Utility scripts
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd apple_ml_trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Setup

```bash
# Run the setup test to verify everything works
python scripts/test_setup.py
```

### 3. Launch Dashboard

```bash
# Start the web dashboard
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

## 📦 Dependencies

### Core Requirements
- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **yfinance** - Stock data collection
- **streamlit** - Web dashboard
- **plotly** - Interactive visualizations

### ML & Analysis
- **scikit-learn** - Traditional ML models
- **xgboost** - Gradient boosting
- **lightgbm** - Light gradient boosting
- **tensorflow** - Deep learning (LSTM)
- **torch** - PyTorch (Transformers)

### Advanced Features
- **ta-lib** - Technical analysis indicators
- **hmmlearn** - Hidden Markov Models (regime detection)
- **arch** - GARCH volatility models
- **influxdb-client** - Time-series database
- **textblob** - Sentiment analysis

### Optional
- **pandas-ta** - Additional technical indicators
- **pyfolio** - Portfolio analysis
- **empyrical** - Risk metrics

## 🎯 Key Features

### 📊 Data Collection
- **Multi-source data**: Stock prices, market indices, economic indicators
- **Real-time updates**: Automated daily data collection
- **Data validation**: Quality checks and cleaning
- **Time-series storage**: InfluxDB integration for performance

### 🔬 Advanced Features
- **Regime Detection**: Hidden Markov Models for market states
- **Volatility Modeling**: GARCH models for volatility clustering
- **Sentiment Analysis**: News and social media sentiment
- **Technical Indicators**: 50+ technical analysis indicators

### 🤖 Machine Learning
- **Ensemble Models**: Random Forest, XGBoost, LightGBM
- **Deep Learning**: LSTM and Transformer models
- **Time-series CV**: Walk-forward validation
- **Feature Engineering**: 200+ engineered features

### ⚠️ Risk Management
- **Advanced Metrics**: VaR, CVaR, tail risk analysis
- **Position Sizing**: Kelly criterion and risk parity
- **Drawdown Control**: Maximum drawdown limits
- **Stress Testing**: Performance under extreme conditions

### 🌐 Web Dashboard
- **Real-time Monitoring**: Live price and prediction tracking
- **Interactive Charts**: Plotly-based visualizations
- **Risk Analysis**: Comprehensive risk dashboards
- **Model Insights**: Feature importance and prediction confidence

## 📈 Usage Examples

### 🌐 Live Dashboard
Visit: **https://taiwan-howard-lee.github.io**
- Real-time Apple stock data from Polygon.io
- Technical indicators (RSI, MACD, Bollinger Bands)
- Global economic context from Trading Economics
- Interactive charts and professional analysis

### 📊 Continuous Data Collection (NEW!)
```bash
# Run continuous collection for 8 hours (maximizes API usage)
python3 run_continuous_collection.py --hours 8

# Test mode (2 minutes)
python3 test_continuous_collector.py
```

### 🔌 Polygon.io API Integration
```python
from src.data_collection.polygon_collector import PolygonCollector

polygon = PolygonCollector()
# Get real-time Apple data
aapl_data = polygon.get_market_summary('AAPL')
print(f"Current AAPL price: ${aapl_data['current_data']['close']}")

# Get historical data
historical = polygon.get_aggregates('AAPL', days=30)
print(f"Retrieved {len(historical)} days of OHLCV data")
```

### 🌍 Trading Economics API Integration
```python
from src.data_collection.trading_economics_collector import TradingEconomicsCollector

te = TradingEconomicsCollector()
# Get global economic indicators
economic_data = te.get_market_summary()
print(f"Economic indicators: {len(economic_data['economic_indicators'])}")

# Get currency data
currency_data = te.get_currency_data()
print(f"Currency pairs: {len(currency_data)}")
```

### 📈 Basic Data Collection (Legacy)
```python
from src.data_collection.apple_collector import AppleDataCollector

collector = AppleDataCollector()
data = collector.fetch_daily_data(period="1y")
print(f"Collected {len(data)} days of data")
```

### 🔧 Technical Indicators
```python
from src.feature_engineering.technical_indicators import TechnicalIndicators

indicators = TechnicalIndicators(data)
all_indicators = indicators.calculate_all_indicators()
print(f"Generated {len(all_indicators.columns)} indicators")
```

## 🛠️ Development Workflow

Follow the detailed task-based development guide in `TEST_TIMELINE.md`:

1. **Phase 1**: Foundation & Infrastructure (5 tasks)
2. **Phase 2**: Advanced Features & Alternative Data (5 tasks)
3. **Phase 3**: Machine Learning Development (5 tasks)
4. **Phase 4**: Risk Management & Backtesting (2 tasks)
5. **Phase 5**: Web Dashboard Development (3 tasks)
6. **Phase 6**: Integration & Final Testing (2 tasks)

## 📊 Performance Targets

### Technical Validation
- ✅ **Data Coverage**: 95%+ successful daily data collection
- ✅ **Feature Count**: 200+ engineered features
- ✅ **Model Accuracy**: >55% classification accuracy
- ✅ **Processing Speed**: <30 seconds for daily predictions

### Financial Performance
- ✅ **Sharpe Ratio**: >1.0 in backtesting
- ✅ **Maximum Drawdown**: <20% portfolio drawdown
- ✅ **Win Rate**: >50% for buy/sell signals
- ✅ **Risk-Adjusted Returns**: Beat SPY benchmark

## 🔧 Configuration

Key configuration files:
- `config/data_config.py` - Data sources and processing settings
- `requirements.txt` - Python dependencies
- `dashboard/app.py` - Dashboard configuration

Environment variables:
```bash
export INFLUXDB_TOKEN="your-influxdb-token"
export ALPHA_VANTAGE_API_KEY="your-api-key"
export NEWS_API_KEY="your-news-api-key"
```

## 🧪 Testing

```bash
# Run setup tests
python scripts/test_setup.py

# Run unit tests (when implemented)
python -m pytest tests/

# Test dashboard
streamlit run dashboard/app.py
```

## 📝 Documentation

- `TEST_PROJECT_PLAN.md` - Comprehensive project plan
- `TEST_TIMELINE.md` - Task-based development checklist
- `notebooks/` - Jupyter notebooks with analysis examples

## 🤝 Contributing

1. Follow the task-based development approach in `TEST_TIMELINE.md`
2. Write tests for new functionality
3. Update documentation
4. Follow Python best practices (PEP 8)

## ⚠️ Disclaimer

This is an educational project for learning ML and quantitative finance. 
**Not intended for actual trading without proper risk management and testing.**

## 📄 License

This project is for educational purposes. Please ensure compliance with data provider terms of service.

---

**🍎 Apple ML Trading System** - Built with Python, Streamlit, and modern ML techniques.
