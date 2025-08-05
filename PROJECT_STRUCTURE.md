# 📁 Stock Prediction Agent SDK - Project Structure

## 🎯 **Core Application Files**

### **Main Application**
- **`app_fresh.py`** - Main dashboard application with all analysis types
  - Price Prediction
  - Technical Analysis  
  - Backtesting
  - Prediction Accuracy Analysis

### **Prediction Accuracy & Learning System**
- **`prediction_tracker.py`** - Core prediction tracking and accuracy calculation
- **`prediction_accuracy_dashboard.py`** - Accuracy analysis dashboard components
- **`accuracy_learning_engine.py`** - Advanced learning and improvement engine
- **`update_actual_prices.py`** - Automated price update script

### **Core System Components**
- **`agents/data_agent.py`** - Data fetching and preprocessing
- **`agents/prediction_agent.py`** - Machine learning predictions
- **`backtesting/engine.py`** - Strategy testing framework

## 📊 **Data Storage**

### **`data/` Directory**
- **`data/predictions/`** - Prediction tracking data (JSON)
- **`data/learning/`** - Learning insights and analysis (JSON)

### **`models/` Directory**
- Trained ML models (.joblib files)
- Model scalers and preprocessing components

## 🗂️ **Supporting Directories**

- **`logs/`** - Application logs
- **`reports/`** - Generated analysis reports
- **`Scripts/`** - Virtual environment scripts
- **`Lib/`** - Python libraries
- **`Include/`** - C headers
- **`etc/`** - Configuration files
- **`share/`** - Shared resources

## 🚀 **How to Use**

### **1. Start the Dashboard**
```bash
python app_fresh.py
```

### **2. Access Analysis Types**
- **Price Prediction**: Make ML-based price forecasts
- **Technical Analysis**: View technical indicators and charts
- **Backtesting**: Test trading strategies on historical data
- **Prediction Accuracy**: View accuracy analysis and improvement recommendations

### **3. Automated Learning**
- All predictions are automatically tracked
- Accuracy is calculated and analyzed
- Improvement recommendations are generated
- System learns and improves over time

## 📈 **Key Features**

### **Prediction Tracking**
- ✅ Automatic prediction storage
- ✅ Accuracy calculation
- ✅ Error analysis
- ✅ Confidence correlation

### **Learning System**
- ✅ Pattern recognition
- ✅ Factor analysis
- ✅ Improvement recommendations
- ✅ Accuracy progression tracking

### **Dashboard Interface**
- ✅ Interactive charts
- ✅ Real-time updates
- ✅ Comprehensive metrics
- ✅ User-friendly design

## 🔧 **Maintenance**

### **Update Actual Prices**
```bash
python update_actual_prices.py --update
```

### **View Accuracy Summary**
```bash
python update_actual_prices.py --summary
```

### **Get Recommendations**
```bash
python update_actual_prices.py --recommendations
```

## 📋 **Dependencies**

See `requirements.txt` for all required packages:
- Dash, Plotly, Pandas, NumPy
- Scikit-learn, LightGBM
- YFinance, Backtrader
- And more...

## 🎯 **Expected Outcomes**

### **Accuracy Improvement Timeline**
- **Week 1-2**: Baseline performance tracking
- **Week 3-4**: Pattern recognition and analysis
- **Week 5-6**: Improvement recommendations
- **Week 7-8**: Measurable accuracy gains
- **Month 2-3**: 10-15% accuracy improvement
- **Month 4-6**: 15-25% accuracy improvement

### **Continuous Learning**
- System adapts to market changes
- Learns from prediction successes/failures
- Provides increasingly accurate recommendations
- Shows measurable improvements over time 