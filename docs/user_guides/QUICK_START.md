# 🚀 Apple ML Trading - Quick Start Guide

## ⚡ **Immediate Commands (Use Python 3)**

### **🌐 View Live Dashboard**
```
https://taiwan-howard-lee.github.io
```

### **📊 Start Enhanced Data Collection (RECOMMENDED)**
```bash
# 8-hour ENHANCED collection (maximum diversity)
python3 run_enhanced_collection.py --hours 8

# Test enhanced strategy (5 minutes)
python3 run_enhanced_collection.py --test

# Compare basic vs enhanced strategies
python3 compare_collection_strategies.py
```

### **📊 Basic Data Collection (Legacy)**
```bash
# 8-hour basic collection session
python3 run_continuous_collection.py --hours 8

# Quick test (2 minutes)
python3 test_continuous_collector.py
```

### **🧪 Test Individual APIs**
```bash
# Test Polygon.io API
python3 test_polygon_api.py

# Test all collectors
python3 test_polygon_collector.py
```

## 📈 **Expected Results**

### **8-Hour Enhanced Collection Session:**
- **2,400 API calls** to Polygon.io (5 per minute × 480 minutes)
- **31 unique tickers** (vs 7 basic) - 343% more coverage
- **16 time periods** (1d to 365d) - 220% more diversity
- **90%+ fresh data** (vs 60% basic) - smart deduplication
- **~800 AAPL data points** (comprehensive historical coverage)
- **~400 news articles** (varied limits: 3-25 articles)
- **~100 dividend records** across all tickers
- **Tech sector coverage**: MSFT, GOOGL, AMZN, META, TSLA, NVDA, NFLX, ADBE
- **Market ETFs**: SPY, QQQ, IWM, VTI, VOO, GLD, TLT
- **Sector ETFs**: XLK, XLF, XLE, XLV, XLI, XLY, XLP
- **135 economic indicators** from Trading Economics
- **4 currency pairs** (USD/MXN, USD/SEK, etc.)

### **Data Storage:**
```
data/continuous_collection/
├── session_YYYYMMDD_HHMMSS.json    # Complete session data
└── continuous_collection.log        # Detailed logs
```

## 🔧 **System Requirements**

### **Python Version:**
- **Python 3.8+** (Use `python3` command)
- Virtual environment recommended

### **API Keys (Already Configured):**
- ✅ **Polygon.io**: `YpK43xQz3xo0hRVS2l6u8lqwJPSn_Tgf`
- ✅ **Trading Economics**: `50439e96184c4b1:7008dwvh5w03yxa`

### **Rate Limits:**
- **Polygon.io**: 5 requests/minute (free tier)
- **Trading Economics**: No limit for available countries

## 📊 **Real Data Examples**

### **Current AAPL Data (Polygon.io):**
- **Price**: $211.18 (real-time)
- **Volume**: 48,974,591 shares
- **Date**: 2025-07-19
- **52W Range**: $164.08 - $237.23

### **Economic Context (Trading Economics):**
- **USD/MXN**: 18.7369 (-0.09%)
- **USD/SEK**: 9.6498 (-0.77%)
- **Mexico GDP**: Available
- **Inflation Data**: Available

## 🎯 **Monitoring Collection**

### **Watch Progress:**
```bash
# Monitor log file in real-time
tail -f data/continuous_collection/continuous_collection.log

# Check latest session file
ls -la data/continuous_collection/session_*.json
```

### **Collection Status:**
- ✅ **Success Rate**: ~90%+ expected
- ⏱️ **Request Interval**: 12 seconds (rate limit compliance)
- 💾 **Auto-Save**: Every 10 requests
- 📝 **Full Logging**: All API calls tracked

## 🚨 **Important Notes**

### **Use Python 3:**
- Always use `python3` command (not `python`)
- Ensures compatibility and avoids Python 2/3 conflicts

### **Background Running:**
- Collection can run for hours unattended
- Use `Ctrl+C` to stop gracefully
- Data is saved automatically during collection

### **VPN Consideration:**
- Trading Economics works better with VPN
- Polygon.io works globally

## 🎉 **Ready to Start!**

**Recommended First Steps:**
1. **Test**: `python3 test_continuous_collector.py`
2. **Start Collection**: `python3 run_continuous_collection.py --hours 8`
3. **Monitor**: `tail -f data/continuous_collection/continuous_collection.log`
4. **View Dashboard**: https://taiwan-howard-lee.github.io

**Let the system run for hours to maximize data collection!** 🚀
