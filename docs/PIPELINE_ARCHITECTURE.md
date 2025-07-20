# 🏗️ Apple ML Trading - Pipeline Architecture

## 📊 **Complete Data Pipeline Overview**

The Apple ML Trading system now features a comprehensive, professional-grade data pipeline that orchestrates the entire data flow from collection to analysis-ready features.

## 🎯 **Pipeline Architecture**

### **Data Flow Diagram**
```
APIs → Collection → Validation → Processing → Features → Export → Models
  ↓        ↓           ↓           ↓          ↓        ↓       ↓
Logs → Monitoring → Quality → Cleaning → Indicators → Storage → Dashboard
```

### **Pipeline Stages**
1. **📊 Data Collection** - Multi-source API data gathering
2. **🔍 Data Validation** - Quality checks and integrity verification  
3. **⚙️ Data Processing** - Cleaning, transformation, and preparation
4. **🔧 Feature Engineering** - Technical indicators and derived features
5. **📤 Data Export** - Final dataset preparation and storage

## 🏗️ **Directory Structure**

### **Organized Codebase**
```
apple_ml_trading/
├── 📁 src/                          # Core application code
│   ├── 📁 data_pipeline/            # Complete data pipeline
│   │   ├── 📁 collectors/           # Data collection modules
│   │   │   ├── polygon_data_collector.py
│   │   │   ├── polygon_collector.py
│   │   │   ├── trading_economics_collector.py
│   │   │   └── enhanced_collector.py
│   │   ├── 📁 processors/           # Data processing & ETL
│   │   │   └── data_processor.py
│   │   ├── 📁 validators/           # Data quality validation
│   │   │   └── data_validator.py
│   │   └── 📁 orchestrators/        # Pipeline orchestration
│   │       └── pipeline_orchestrator.py
│   ├── 📁 feature_engineering/      # Technical indicators & features
│   ├── 📁 models/                   # ML models & training
│   └── 📁 utils/                    # Shared utilities
├── 📁 data/                         # Organized data storage
│   ├── 📁 raw/                      # Raw API data (immutable)
│   │   ├── 📁 polygon/              # Polygon.io data
│   │   ├── 📁 trading_economics/    # Trading Economics data
│   │   └── 📁 sessions/             # Collection sessions
│   ├── 📁 processed/                # Cleaned & processed data
│   ├── 📁 features/                 # Feature engineered data
│   ├── 📁 models/                   # Trained model artifacts
│   ├── 📁 exports/                  # Final datasets
│   ├── 📁 validation_reports/       # Data quality reports
│   └── 📁 processing_reports/       # Processing summaries
├── 📁 config/                       # Configuration management
│   ├── 📁 pipelines/                # Pipeline configurations
│   │   └── default_pipeline.json
│   ├── 📁 environments/             # Environment-specific configs
│   └── 📁 models/                   # Model configurations
├── 📁 scripts/                      # Automation & utility scripts
│   ├── 📁 data_collection/          # Collection automation
│   │   ├── run_continuous_collection.py
│   │   ├── run_enhanced_collection.py
│   │   ├── compare_collection_strategies.py
│   │   └── test_*.py
│   ├── 📁 pipeline/                 # Pipeline management
│   │   └── run_pipeline.py
│   └── 📁 deployment/               # Deployment scripts
├── 📁 tests/                        # Comprehensive test suite
│   ├── 📁 unit/                     # Unit tests
│   ├── 📁 integration/              # Integration tests
│   └── 📁 data/                     # Data quality tests
├── 📁 docs/                         # Documentation
│   ├── 📁 user_guides/              # User guides
│   │   ├── README_backup.md
│   │   ├── QUICK_START.md
│   │   ├── DATA_DIVERSITY_STRATEGY.md
│   │   └── CODEBASE_REORGANIZATION_PLAN.md
│   ├── 📁 api/                      # API documentation
│   └── 📁 pipeline/                 # Pipeline documentation
├── 📁 logs/                         # Application logs
└── 📁 notebooks/                    # Jupyter notebooks
```

## 🚀 **Pipeline Components**

### **1. Data Collection Layer**
- **PolygonDataCollector**: Professional US market data collection
- **TradingEconomicsCollector**: Global economic indicators
- **EnhancedCollector**: Maximum diversity collection strategy
- **Rate limiting**: Intelligent API usage optimization
- **Error handling**: Robust retry mechanisms

### **2. Data Validation Layer**
- **Comprehensive validation**: Price, volume, date consistency
- **Quality metrics**: OHLC logic, missing data detection
- **Automated reporting**: JSON validation reports
- **Configurable thresholds**: Customizable quality standards

### **3. Data Processing Layer**
- **Data cleaning**: Duplicate removal, outlier detection
- **Transformation**: Returns calculation, normalization
- **Missing data handling**: Interpolation, forward fill
- **Quality assurance**: Automated data quality checks

### **4. Feature Engineering Layer**
- **Technical indicators**: RSI, MACD, Bollinger Bands, Stochastic
- **Economic features**: Integration with Trading Economics data
- **Derived features**: Returns, volatility, momentum indicators
- **Configurable parameters**: Customizable indicator settings

### **5. Orchestration Layer**
- **Pipeline orchestrator**: Complete workflow management
- **Stage coordination**: Sequential and parallel execution
- **Error recovery**: Intelligent failure handling
- **Progress monitoring**: Real-time pipeline status

## ⚙️ **Configuration Management**

### **Pipeline Configuration (default_pipeline.json)**
```json
{
  "pipeline": {
    "name": "apple_ml_trading_pipeline",
    "version": "2.0.0",
    "stages": [
      "data_collection",
      "data_validation", 
      "data_processing",
      "feature_engineering",
      "data_export"
    ]
  },
  "data_collection": {
    "sources": ["polygon", "trading_economics"],
    "rate_limiting": {
      "polygon": {"requests_per_minute": 5},
      "trading_economics": {"requests_per_minute": 60}
    }
  },
  "data_validation": {
    "price_validation": {"min_price": 1.0, "max_price": 1000.0},
    "quality_thresholds": {"min_pass_rate": 0.90}
  },
  "feature_engineering": {
    "technical_indicators": {
      "momentum": {"rsi": {"period": 14}, "macd": {"fast": 12, "slow": 26}},
      "trend": {"sma": {"periods": [5, 10, 20, 50, 200]}}
    }
  }
}
```

## 🚀 **Usage Commands**

### **Complete Pipeline Execution**
```bash
# Run full pipeline
python3 scripts/pipeline/run_pipeline.py

# Dry run (show what would be executed)
python3 scripts/pipeline/run_pipeline.py --dry-run

# Run specific stages
python3 scripts/pipeline/run_pipeline.py --stages data_collection data_validation

# Check pipeline status
python3 scripts/pipeline/run_pipeline.py status
```

### **Individual Component Testing**
```bash
# Test data collection
python3 src/data_pipeline/collectors/polygon_data_collector.py

# Test data validation
python3 src/data_pipeline/validators/data_validator.py

# Test data processing
python3 src/data_pipeline/processors/data_processor.py

# Test pipeline orchestrator
python3 src/data_pipeline/orchestrators/pipeline_orchestrator.py
```

### **Enhanced Data Collection**
```bash
# Maximum diversity collection
python3 scripts/data_collection/run_enhanced_collection.py --hours 8

# Compare collection strategies
python3 scripts/data_collection/compare_collection_strategies.py
```

## 📊 **Data Quality & Monitoring**

### **Validation Reports**
- **Location**: `data/validation_reports/`
- **Format**: JSON with detailed quality metrics
- **Frequency**: Generated after each validation stage
- **Content**: File-level and aggregate quality scores

### **Processing Reports**
- **Location**: `data/processing_reports/`
- **Format**: JSON with processing statistics
- **Metrics**: Records processed, cleaning operations, duration
- **Quality tracking**: Before/after data quality comparison

### **Pipeline Logs**
- **Location**: `logs/pipeline.log`
- **Level**: Configurable (DEBUG, INFO, WARNING, ERROR)
- **Rotation**: Automatic log rotation with size limits
- **Monitoring**: Real-time pipeline status tracking

## 🎯 **Benefits Achieved**

### **Code Organization**
- ✅ **Professional structure** - Clear separation of concerns
- ✅ **Maintainable codebase** - Easy navigation and updates
- ✅ **Scalable architecture** - Ready for additional features
- ✅ **Comprehensive testing** - Unit and integration test support

### **Data Pipeline**
- ✅ **Automated workflow** - End-to-end data processing
- ✅ **Quality assurance** - Comprehensive validation and monitoring
- ✅ **Error resilience** - Robust error handling and recovery
- ✅ **Performance optimization** - Efficient data processing

### **Developer Experience**
- ✅ **Easy deployment** - Simple command-line interface
- ✅ **Clear documentation** - Comprehensive guides and examples
- ✅ **Flexible configuration** - Customizable pipeline behavior
- ✅ **Monitoring tools** - Real-time status and quality tracking

## 🚀 **Next Steps**

1. **Run the complete pipeline**: `python3 scripts/pipeline/run_pipeline.py`
2. **Monitor data quality**: Check validation reports in `data/validation_reports/`
3. **Customize configuration**: Edit `config/pipelines/default_pipeline.json`
4. **Add new features**: Extend collectors, processors, or validators
5. **Scale the system**: Add parallel processing and cloud deployment

**🎉 Your Apple ML Trading system now has a professional-grade, production-ready data pipeline!**
