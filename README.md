# Stock AI - Autonomous Trading Bot

A comprehensive stock prediction and analysis system with continuous learning capabilities that automatically improves prediction accuracy over time. The system is designed to evolve into a fully autonomous trading bot capable of achieving 60-70% win rate through advanced machine learning and multi-factor analysis.

## Key Features

### Prediction & Analysis
- **Price Prediction**: ML-based price forecasting with confidence scores using multiple models (Random Forest, XGBoost, LightGBM, Gradient Boosting, Linear Regression)
- **Technical Analysis**: 15+ technical indicators including MACD, RSI, Bollinger Bands, Moving Averages, Stochastic Oscillator
- **Backtesting**: Strategy testing with detailed performance metrics and multiple trading strategies
- **Prediction Accuracy**: Real-time accuracy tracking and improvement recommendations

### Learning System
- **Automatic Tracking**: Every prediction is tracked and analyzed for continuous improvement
- **Pattern Recognition**: Identifies factors affecting prediction accuracy using clustering and statistical analysis
- **Continuous Improvement**: Provides specific recommendations for model enhancement
- **Accuracy Progression**: Shows measurable improvements over time with detailed metrics

### Dashboard Interface
- **Interactive Charts**: Real-time visualization of predictions and accuracy using Plotly
- **Comprehensive Metrics**: Detailed performance analysis with Sharpe ratio, win rate, drawdown metrics
- **Multi-Analysis Support**: Price prediction, technical analysis, backtesting, accuracy analysis
- **Responsive Design**: Modern web interface built with Dash and Bootstrap

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Vinayakp2001/Stock-AI.git
cd Stock-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the dashboard
python app_fresh.py

# Access the system at http://localhost:8050
```

### Basic Usage
1. **Price Prediction**: Select stocks and timeframes for ML-based forecasts
2. **Technical Analysis**: View comprehensive technical indicators and charts
3. **Backtesting**: Test trading strategies on historical data
4. **Prediction Accuracy**: Monitor and analyze prediction performance

## Project Structure

```
Stock-AI/
├── agents/                         # Core system components
│   ├── data_agent.py              # Data fetching and preprocessing
│   └── prediction_agent.py        # Machine learning predictions
├── backtesting/                    # Strategy testing framework
│   └── engine.py                  # Backtesting engine with multiple strategies
├── data/                          # Data storage
│   ├── predictions/               # Prediction tracking data
│   └── learning/                  # Learning insights and analysis
├── models/                        # Trained ML models storage
├── app_fresh.py                   # Main dashboard application
├── prediction_tracker.py          # Prediction tracking system
├── prediction_accuracy_dashboard.py # Accuracy analysis dashboard
├── accuracy_learning_engine.py    # Learning and improvement engine
├── update_actual_prices.py        # Automated price updates
├── main.py                        # Command-line interface
└── requirements.txt               # Project dependencies
```

## Architecture Overview

### Data Collection & Processing
- **Real-time Data**: Yahoo Finance integration for market data
- **Technical Indicators**: 15+ indicators using TA-Lib
- **Feature Engineering**: 40+ derived features for ML models
- **Data Caching**: Intelligent caching to reduce API calls

### Machine Learning Pipeline
- **Multiple Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, Linear Regression
- **Ensemble Methods**: Weighted predictions from multiple models
- **Multi-timeframe Analysis**: 1-day, 1-week, 1-month predictions
- **Confidence Scoring**: Model performance-based confidence intervals

### Backtesting Framework
- **Strategy Testing**: Moving Average, RSI, MACD strategies
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rate, profit factor
- **Risk Analysis**: Comprehensive risk-adjusted return calculations
- **Historical Validation**: Multi-year backtesting capabilities

### Learning & Improvement
- **Prediction Tracking**: Automatic tracking of all predictions vs actual outcomes
- **Error Analysis**: Pattern recognition in prediction errors using clustering
- **Performance Attribution**: Detailed analysis of what drives accuracy
- **Continuous Learning**: Automatic model retraining and parameter optimization

## Usage Examples

### Command Line Interface
```bash
# Analyze a specific stock
python main.py --symbol AAPL --period 6mo --indicators

# Run backtesting
python main.py --symbol RELIANCE.NS --backtest

# Compare multiple stocks
python main.py --compare AAPL GOOGL MSFT --period 1y

# Run comprehensive demo
python main.py --demo
```

### Dashboard Interface
1. **Stock Selection**: Choose from US stocks, Indian stocks, or market indices
2. **Analysis Type**: Select prediction, technical analysis, backtesting, or accuracy analysis
3. **Time Period**: Configure analysis period from 1 month to 2 years
4. **Real-time Results**: View interactive charts and detailed metrics

### Prediction Tracking
```bash
# Update actual prices for completed predictions
python update_actual_prices.py --update

# View accuracy summary
python update_actual_prices.py --summary

# Get improvement recommendations
python update_actual_prices.py --recommendations
```

## Technical Specifications

### Supported Markets
- **US Markets**: NYSE, NASDAQ (AAPL, GOOGL, MSFT, TSLA, etc.)
- **Indian Markets**: NSE, BSE (RELIANCE.NS, TCS.NS, INFY.NS, etc.)
- **Market Indices**: S&P 500, NASDAQ, Dow Jones, NIFTY, SENSEX

### Machine Learning Models
- **Random Forest**: Ensemble learning with 200 estimators
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting framework
- **Gradient Boosting**: Sequential ensemble method
- **Linear Regression**: Baseline model with feature scaling

### Technical Indicators
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, CCI
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA
- **Custom**: Price patterns, volatility measures, momentum features

### Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Confidence Correlation**: Relationship between confidence and accuracy

## Expected Outcomes

### Accuracy Improvement Timeline
- **Month 1**: 5-10% accuracy improvement through basic learning
- **Month 2**: 10-15% accuracy improvement with pattern recognition
- **Month 3**: 15-20% accuracy improvement with advanced optimization
- **Month 4+**: 20-25% accuracy improvement with continuous learning

### Performance Targets
- **Win Rate**: Target 60-70% for autonomous trading
- **Sharpe Ratio**: Target >1.5 for risk-adjusted returns
- **Maximum Drawdown**: Keep below 15% for capital protection
- **Prediction Accuracy**: Achieve >75% directional accuracy

## Development Roadmap

### Phase 1: Foundation Enhancement
- Add fundamental analysis (P/E ratios, financial metrics)
- Implement sentiment analysis (news, social media)
- Build multi-factor scoring system
- Enhance risk management framework

### Phase 2: Autonomous Trading
- Integrate broker APIs (Zerodha, Alpaca)
- Implement order execution system
- Add position management
- Build safety controls and circuit breakers

### Phase 3: Advanced Features
- Portfolio optimization
- Market regime detection
- Advanced learning algorithms
- Mobile application

## Contributing

We welcome contributions from the community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up the development environment
- Code style and testing requirements
- Issue reporting and feature requests
- Pull request process
- Community guidelines

### Quick Start for Contributors
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

See our [GitHub Issues](https://github.com/Vinayakp2001/Stock-AI/issues) for current development priorities.

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting framework
- **lightgbm**: Fast gradient boosting
- **yfinance**: Yahoo Finance data
- **ta**: Technical analysis indicators

### Visualization & Dashboard
- **dash**: Web application framework
- **plotly**: Interactive plotting
- **dash-bootstrap-components**: UI components

### Additional Libraries
- **tensorflow**: Deep learning (optional)
- **backtrader**: Backtesting framework
- **scipy**: Scientific computing
- **requests**: HTTP library

See [requirements.txt](requirements.txt) for complete dependency list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yahoo Finance for providing free market data
- The open-source community for excellent libraries
- Contributors who help improve the system

## Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. Trading stocks involves risk, and you should carefully consider your investment objectives and risk tolerance before making any investment decisions. Past performance does not guarantee future results.

## Contact

- **Repository**: https://github.com/Vinayakp2001/Stock-AI
- **Issues**: https://github.com/Vinayakp2001/Stock-AI/issues
- **Discussions**: https://github.com/Vinayakp2001/Stock-AI/discussions

---

**Built with Python and Machine Learning for the Future of Algorithmic Trading**
