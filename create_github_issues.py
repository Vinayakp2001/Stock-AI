#!/usr/bin/env python3
"""
GitHub Issues Creator - All 25 Issues for Stock AI Project
"""

import subprocess
import sys

REPO = "Vinayakp2001/Stock-AI"

ISSUES = [
    {
        "title": "Implement Fundamental Analysis Engine for Stock Evaluation",
        "body": """## Objective
Add fundamental analysis capabilities to complement existing technical analysis. Key component for achieving 60-70% win rate.

## Acceptance Criteria
- [ ] Create `FundamentalAnalyzer` class in `agents/fundamental_agent.py`
- [ ] Implement financial ratios (P/E, P/B, ROE, ROA, Debt-to-Equity)
- [ ] Add earnings data analysis (EPS, revenue growth, profit margins)
- [ ] Implement valuation metrics (PEG ratio, dividend yield)
- [ ] Add company health indicators (cash flow, debt levels)
- [ ] Create sector comparison functionality
- [ ] Generate fundamental score (0-100)
- [ ] Add unit tests with 80%+ coverage
- [ ] Update documentation

## Technical Requirements
- Use `yfinance` for financial data extraction
- Return standardized scores (0-100)
- Handle missing data gracefully with fallback values
- Cache results for 24 hours
- Support both US stocks and Indian stocks (.NS, .BO)

## Files to Create/Modify
- `agents/fundamental_agent.py` (new)
- `tests/test_fundamental_agent.py` (new)
- `requirements.txt` (update)
- `README.md` (update)

## Success Metrics
- Processing time < 2 seconds per stock
- 80%+ test coverage
- Fundamental scores correlate with actual stock performance

## Definition of Done
- [ ] Code implemented and reviewed
- [ ] All tests passing with 80%+ coverage
- [ ] Documentation updated
- [ ] Integration with existing prediction system verified"""
    },
    {
        "title": "Add News and Social Media Sentiment Analysis",
        "body": """## Objective
Implement sentiment analysis from news articles and social media to gauge market sentiment for better trading decisions.

## Acceptance Criteria
- [ ] Create `SentimentAnalyzer` class in `agents/sentiment_agent.py`
- [ ] Integrate NewsAPI for news sentiment analysis
- [ ] Implement BERT-based sentiment scoring
- [ ] Create sentiment aggregation logic (multiple sources)
- [ ] Generate sentiment score (0-100) with confidence levels
- [ ] Add caching mechanism (1 hour)
- [ ] Handle API rate limits gracefully
- [ ] Add comprehensive tests with mock data

## Technical Requirements
- Use `newsapi-python` for news data
- Use `transformers` library for BERT sentiment analysis
- Use `textblob` as fallback
- Return sentiment: positive/negative/neutral with confidence score
- Support multiple symbols simultaneously

## Files to Create/Modify
- `agents/sentiment_agent.py` (new)
- `tests/test_sentiment_agent.py` (new)
- `requirements.txt` (update)
- `config/sentiment_config.py` (new)

## Success Metrics
- Processing time < 5 seconds per stock
- Sentiment classification accuracy > 75%
- 85%+ test coverage

## Dependencies
- Requires Issue #1 (fundamental analysis)

## Definition of Done
- [ ] Code implemented with proper error handling
- [ ] All tests passing with 85%+ coverage
- [ ] API integration working reliably
- [ ] Documentation complete with examples"""
    },
    {
        "title": "Build Comprehensive Multi-Factor Stock Scoring System",
        "body": """## Objective
Create a unified scoring system combining technical, fundamental, sentiment, and ML predictions. This is the CORE component for achieving 60-70% win rate.

## Acceptance Criteria
- [ ] Create `TradingDecisionEngine` class in `trading/decision_engine.py`
- [ ] Implement weighted scoring algorithm:
  - Technical Analysis: 30%
  - Fundamental Analysis: 25%
  - Sentiment Analysis: 15%
  - ML Predictions: 20%
  - Risk Metrics: 10%
- [ ] Generate final score (0-100) with confidence intervals
- [ ] Create decision logic (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)
- [ ] Implement score breakdown tracking
- [ ] Add backtesting for scoring system
- [ ] Create visualization for score components

## Files to Create/Modify
- `trading/decision_engine.py` (new)
- `trading/__init__.py` (new)
- `tests/test_decision_engine.py` (new)
- `config/scoring_config.py` (new)

## Success Metrics
- Backtesting shows >60% win rate over 2+ years
- Processing time < 3 seconds per stock
- Decision consistency > 85%

## Dependencies
- Requires Issue #1 (fundamental analysis)
- Requires Issue #2 (sentiment analysis)

## Definition of Done
- [ ] All scoring components integrated
- [ ] Backtesting shows target win rate (>60%)
- [ ] Real-time scoring operational
- [ ] Comprehensive test suite passing
- [ ] Code review completed"""
    },
    {
        "title": "Build Comprehensive Risk Management Module",
        "body": """## Objective
Implement a robust risk management system to protect capital and optimize position sizing. CRITICAL for consistent 60-70% win rate.

## Acceptance Criteria
- [ ] Create `RiskManager` class in `trading/risk_manager.py`
- [ ] Implement Kelly Criterion for position sizing
- [ ] Add portfolio risk controls:
  - Max position size: 10% per trade (configurable)
  - Max portfolio risk: 20% (configurable)
  - Max correlation exposure: 30% (configurable)
  - Max sector exposure: 40% (configurable)
- [ ] Implement stop-loss and take-profit logic
- [ ] Add drawdown protection mechanisms
- [ ] Create risk metrics (VaR, Sharpe, etc.)
- [ ] Add position correlation analysis
- [ ] Implement emergency exit mechanisms

## Files to Create/Modify
- `trading/risk_manager.py` (new)
- `tests/test_risk_manager.py` (new)
- `config/risk_config.py` (new)
- `utils/risk_metrics.py` (new)

## Success Metrics
- Maximum drawdown < 15% in backtesting
- Sharpe ratio > 1.5
- Position sizing prevents any single loss > 5%
- Risk monitoring latency < 1 second

## Dependencies
- Integrates with Issue #3 (decision engine)

## Definition of Done
- [ ] All risk management rules implemented
- [ ] Stress testing passed
- [ ] Integration with decision engine complete
- [ ] Emergency procedures tested
- [ ] Documentation complete"""
    },
    {
        "title": "Implement Flexible Trading Strategy Framework",
        "body": """## Objective
Create a flexible framework for defining and executing different trading strategies with the multi-factor scoring system.

## Acceptance Criteria
- [ ] Create abstract `TradingStrategy` base class
- [ ] Implement `MultiFactorStrategy` as the main strategy
- [ ] Add strategy configuration system (JSON/YAML based)
- [ ] Create strategy backtesting framework
- [ ] Implement strategy performance tracking
- [ ] Add strategy comparison tools
- [ ] Create strategy optimization tools

## Files to Create/Modify
- `trading/strategies/base_strategy.py` (new)
- `trading/strategies/multi_factor_strategy.py` (new)
- `trading/strategies/__init__.py` (new)
- `tests/test_strategies.py` (new)
- `config/strategies/` (new directory)

## Success Metrics
- Easy to add new strategies (plugin architecture)
- Strategy performance tracking accurate
- Backtesting framework reliable

## Dependencies
- Requires Issue #3 (decision engine)
- Requires Issue #4 (risk management)

## Definition of Done
- [ ] Strategy framework implemented
- [ ] Multi-factor strategy operational
- [ ] Configuration system working
- [ ] Backtesting framework complete
- [ ] Documentation and examples ready"""
    },
    {
        "title": "Create Universal Broker API Interface",
        "body": """## Objective
Build an abstraction layer that works with multiple brokers (Zerodha, Alpaca, etc.) for order execution.

## Acceptance Criteria
- [ ] Create abstract `BrokerConnector` base class in `brokers/base_broker.py`
- [ ] Define standard interface for:
  - Place orders (market/limit/stop)
  - Cancel orders
  - Get current positions
  - Get account balance
  - Get order status
  - Get real-time quotes
- [ ] Implement error handling and retries
- [ ] Add order validation
- [ ] Create mock broker for testing
- [ ] Add comprehensive logging
- [ ] Implement rate limiting

## Files to Create/Modify
- `brokers/base_broker.py` (new)
- `brokers/mock_broker.py` (new)
- `brokers/__init__.py` (new)
- `tests/test_broker_interface.py` (new)
- `config/broker_config.py` (new)

## Success Metrics
- Order execution latency < 2 seconds
- 99.9% uptime with proper error handling
- Consistent interface across all brokers

## Definition of Done
- [ ] Abstract interface implemented
- [ ] Mock broker working for testing
- [ ] All error scenarios handled
- [ ] Comprehensive test suite passing
- [ ] Documentation complete"""
    },
    {
        "title": "Implement Zerodha Kite Connect Integration",
        "body": """## Objective
Integrate with Zerodha's Kite Connect API for Indian stock market trading.

## Acceptance Criteria
- [ ] Create `ZerodhaConnector` class in `brokers/zerodha_broker.py`
- [ ] Implement authentication flow
- [ ] Add order placement functionality
- [ ] Implement position monitoring
- [ ] Add margin calculation
- [ ] Handle API rate limits
- [ ] Add error handling for common scenarios
- [ ] Create configuration management
- [ ] Add comprehensive tests (with mocking)

## Files to Create/Modify
- `brokers/zerodha_broker.py` (new)
- `tests/test_zerodha_broker.py` (new)
- `config/zerodha_config.py` (new)

## Success Metrics
- Successful order placement in paper trading
- Proper error handling for all API failures
- Authentication flow working reliably

## Dependencies
- Requires Issue #6 (broker abstraction)

## Definition of Done
- [ ] Authentication working
- [ ] Order placement tested
- [ ] Position monitoring operational
- [ ] All tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Implement Alpaca API Integration for US Stocks",
        "body": """## Objective
Integrate with Alpaca API for commission-free US stock trading.

## Acceptance Criteria
- [ ] Create `AlpacaConnector` class in `brokers/alpaca_broker.py`
- [ ] Implement authentication
- [ ] Add order management
- [ ] Implement position tracking
- [ ] Add account information retrieval
- [ ] Handle market hours
- [ ] Add paper trading support
- [ ] Create comprehensive tests

## Files to Create/Modify
- `brokers/alpaca_broker.py` (new)
- `tests/test_alpaca_broker.py` (new)
- `config/alpaca_config.py` (new)

## Success Metrics
- Paper trading working end-to-end
- Market hours handling correct
- All order types supported

## Dependencies
- Requires Issue #6 (broker abstraction)

## Definition of Done
- [ ] Authentication working
- [ ] Paper trading tested
- [ ] Order management operational
- [ ] All tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Create Intelligent Order Execution System",
        "body": """## Objective
Build a sophisticated order execution engine that handles order placement, monitoring, and management.

## Acceptance Criteria
- [ ] Create `OrderExecutor` class in `trading/order_executor.py`
- [ ] Implement smart order routing
- [ ] Add order validation logic
- [ ] Implement partial fill handling
- [ ] Add order modification capabilities
- [ ] Create order status monitoring
- [ ] Implement execution algorithms (TWAP, VWAP)
- [ ] Add slippage tracking
- [ ] Create execution reports

## Files to Create/Modify
- `trading/order_executor.py` (new)
- `tests/test_order_executor.py` (new)

## Success Metrics
- Order execution latency < 2 seconds
- Slippage tracking accurate
- Zero order loss during execution

## Dependencies
- Requires Issue #6 (broker abstraction)
- Requires Issue #4 (risk management)

## Definition of Done
- [ ] Order routing implemented
- [ ] All order types supported
- [ ] Slippage tracking working
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Build Automated Position Management and Monitoring",
        "body": """## Objective
Create a system to monitor open positions and execute exit strategies automatically.

## Acceptance Criteria
- [ ] Create `PositionManager` class in `trading/position_manager.py`
- [ ] Implement real-time position monitoring
- [ ] Add stop-loss execution
- [ ] Add take-profit execution
- [ ] Implement trailing stops
- [ ] Add time-based exits
- [ ] Create position sizing adjustments
- [ ] Add P&L tracking
- [ ] Implement position alerts

## Files to Create/Modify
- `trading/position_manager.py` (new)
- `tests/test_position_manager.py` (new)

## Success Metrics
- Stop-loss execution within 1 second
- P&L tracking accurate to 2 decimal places
- Zero missed exits

## Dependencies
- Requires Issue #9 (order execution)

## Definition of Done
- [ ] Position monitoring operational
- [ ] All exit strategies working
- [ ] P&L tracking accurate
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Build Market Regime Detection System",
        "body": """## Objective
Implement a system to detect different market conditions (bull/bear, trending/ranging, high/low volatility) and adjust strategies accordingly.

## Acceptance Criteria
- [ ] Create `MarketRegimeDetector` class in `analysis/market_regime.py`
- [ ] Implement trend detection algorithms
- [ ] Add volatility regime detection
- [ ] Create market sentiment indicators
- [ ] Implement regime-based strategy adjustment
- [ ] Add regime change alerts
- [ ] Create historical regime analysis
- [ ] Add visualization tools

## Files to Create/Modify
- `analysis/market_regime.py` (new)
- `tests/test_market_regime.py` (new)

## Success Metrics
- Regime detection accuracy > 75%
- Regime change alerts within 5 minutes
- Strategy adjustment improves win rate by 5%+

## Definition of Done
- [ ] Regime detection implemented
- [ ] Strategy adjustment working
- [ ] Alerts operational
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Enhance Continuous Learning System for Strategy Optimization",
        "body": """## Objective
Upgrade the existing learning engine to automatically optimize trading strategies based on performance data.

## Acceptance Criteria
- [ ] Enhance existing `AccuracyLearningEngine`
- [ ] Add strategy parameter optimization
- [ ] Implement A/B testing framework
- [ ] Add performance attribution analysis
- [ ] Create automatic model retraining
- [ ] Implement feature importance tracking
- [ ] Add strategy ensemble methods
- [ ] Create performance prediction models

## Files to Create/Modify
- `accuracy_learning_engine.py` (enhance existing)
- `learning/strategy_optimizer.py` (new)
- `learning/ab_testing.py` (new)
- `tests/test_learning_engine.py` (new)

## Success Metrics
- Model retraining improves accuracy by 3%+ per cycle
- A/B testing framework reliable
- Feature importance tracking accurate

## Dependencies
- Builds upon existing learning engine
- Requires trading history data

## Definition of Done
- [ ] Learning engine enhanced
- [ ] A/B testing working
- [ ] Auto-retraining operational
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Build Modern Portfolio Theory Based Optimization",
        "body": """## Objective
Implement portfolio optimization techniques to maximize risk-adjusted returns across multiple positions.

## Acceptance Criteria
- [ ] Create `PortfolioOptimizer` class in `trading/portfolio_optimizer.py`
- [ ] Implement Markowitz optimization
- [ ] Add Black-Litterman model
- [ ] Implement risk parity allocation
- [ ] Add correlation analysis
- [ ] Create efficient frontier calculation
- [ ] Implement dynamic rebalancing
- [ ] Add portfolio performance metrics

## Files to Create/Modify
- `trading/portfolio_optimizer.py` (new)
- `tests/test_portfolio_optimizer.py` (new)

## Success Metrics
- Portfolio Sharpe ratio > 1.5
- Efficient frontier calculation accurate
- Rebalancing reduces drawdown by 5%+

## Definition of Done
- [ ] Markowitz optimization working
- [ ] Dynamic rebalancing operational
- [ ] Performance metrics accurate
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Create Comprehensive Trading Bot Dashboard",
        "body": """## Objective
Build a real-time dashboard to monitor bot performance, positions, and key metrics.

## Acceptance Criteria
- [ ] Enhance existing Dash dashboard
- [ ] Add real-time P&L tracking
- [ ] Create position monitoring panel
- [ ] Add performance metrics display
- [ ] Implement alert system
- [ ] Create trade history visualization
- [ ] Add risk metrics dashboard
- [ ] Implement mobile-responsive design

## Files to Create/Modify
- `app_fresh.py` (enhance existing)
- `dashboard/trading_dashboard.py` (new)
- `dashboard/components/` (new directory)

## Success Metrics
- Dashboard loads in < 3 seconds
- Real-time updates every 5 seconds
- Mobile-responsive on all screen sizes

## Dependencies
- Builds upon existing dashboard
- Requires trading system components

## Definition of Done
- [ ] Real-time P&L working
- [ ] Position monitoring operational
- [ ] Mobile-responsive design complete
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Build Comprehensive Alert and Notification System",
        "body": """## Objective
Create a system to send alerts for important trading events via multiple channels.

## Acceptance Criteria
- [ ] Create `AlertManager` class in `notifications/alert_manager.py`
- [ ] Implement email notifications
- [ ] Add Telegram bot integration
- [ ] Create webhook support
- [ ] Implement alert prioritization
- [ ] Create alert templates
- [ ] Add alert history tracking

## Files to Create/Modify
- `notifications/alert_manager.py` (new)
- `notifications/telegram_bot.py` (new)
- `notifications/email_notifier.py` (new)
- `tests/test_alert_manager.py` (new)

## Success Metrics
- Alert delivery within 30 seconds
- 99%+ delivery rate
- Zero false positives for critical alerts

## Definition of Done
- [ ] Email notifications working
- [ ] Telegram integration complete
- [ ] Alert history tracking operational
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Build Comprehensive Safety Control System",
        "body": """## Objective
Implement multiple layers of safety controls to protect against system failures and market anomalies.

## Acceptance Criteria
- [ ] Create `SafetyController` class in `trading/safety_controller.py`
- [ ] Implement daily loss limits (5% default)
- [ ] Add consecutive loss protection (halt after 5 losses)
- [ ] Create volatility-based trading halts
- [ ] Implement emergency stop mechanisms
- [ ] Add system health monitoring
- [ ] Create automatic position liquidation
- [ ] Add manual override capabilities

## Files to Create/Modify
- `trading/safety_controller.py` (new)
- `tests/test_safety_controller.py` (new)
- `config/safety_config.py` (new)

## Success Metrics
- Emergency stop executes within 1 second
- Daily loss limit never exceeded
- Zero system failures due to missing safety checks

## Definition of Done
- [ ] All safety rules implemented
- [ ] Emergency stop tested
- [ ] Circuit breakers working
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Implement Detailed Logging and Audit System",
        "body": """## Objective
Create comprehensive logging for all trading activities to ensure transparency and debugging capabilities.

## Acceptance Criteria
- [ ] Implement structured logging across all modules
- [ ] Add trade execution logging
- [ ] Create decision audit trail
- [ ] Add performance logging
- [ ] Implement log rotation
- [ ] Create log analysis tools
- [ ] Add compliance reporting

## Files to Create/Modify
- `utils/logger.py` (new)
- `utils/audit_trail.py` (new)
- `config/logging_config.py` (new)

## Success Metrics
- All trades logged with full details
- Log analysis tools working
- Compliance reports generated correctly

## Definition of Done
- [ ] Structured logging implemented
- [ ] Audit trail complete
- [ ] Log rotation working
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Enhanced Backtesting and Paper Trading Platform",
        "body": """## Objective
Enhance the existing backtesting system and add comprehensive paper trading capabilities.

## Acceptance Criteria
- [ ] Enhance existing backtesting engine
- [ ] Add paper trading mode
- [ ] Implement realistic slippage modeling
- [ ] Add commission and fee calculations
- [ ] Create walk-forward analysis
- [ ] Add Monte Carlo simulations
- [ ] Implement stress testing
- [ ] Create performance attribution

## Files to Create/Modify
- `backtesting/engine.py` (enhance existing)
- `backtesting/paper_trading.py` (new)
- `backtesting/monte_carlo.py` (new)
- `tests/test_backtesting.py` (new)

## Success Metrics
- Paper trading results within 5% of live trading
- Monte Carlo simulations statistically valid
- Walk-forward analysis accurate

## Dependencies
- Builds upon existing backtesting engine

## Definition of Done
- [ ] Paper trading mode working
- [ ] Slippage modeling accurate
- [ ] Monte Carlo simulations complete
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Build Flexible Configuration Management System",
        "body": """## Objective
Create a comprehensive configuration system for managing bot parameters and settings.

## Acceptance Criteria
- [ ] Create configuration schema using Pydantic
- [ ] Implement environment-based configs (.env support)
- [ ] Add parameter validation
- [ ] Create configuration UI in dashboard
- [ ] Implement hot-reloading of configs
- [ ] Add configuration versioning
- [ ] Create backup and restore
- [ ] Add configuration templates for different strategies

## Files to Create/Modify
- `config/config_manager.py` (new)
- `config/schemas.py` (new)
- `config/templates/` (new directory)
- `tests/test_config_manager.py` (new)

## Success Metrics
- Configuration validation catches all invalid inputs
- Hot-reloading works without restart
- All parameters documented

## Definition of Done
- [ ] Configuration schema complete
- [ ] Validation working
- [ ] Hot-reloading operational
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Build Complete Testing Framework",
        "body": """## Objective
Create a comprehensive testing suite covering unit tests, integration tests, and end-to-end tests.

## Acceptance Criteria
- [ ] Achieve 90%+ code coverage
- [ ] Add unit tests for all modules
- [ ] Create integration tests
- [ ] Add end-to-end trading tests
- [ ] Implement performance tests
- [ ] Add stress tests
- [ ] Create mock data generators
- [ ] Add CI/CD pipeline configuration

## Files to Create/Modify
- `tests/unit/` (new directory structure)
- `tests/integration/` (new directory structure)
- `tests/fixtures/` (new directory)
- `.github/workflows/tests.yml` (new)

## Success Metrics
- 90%+ code coverage
- All tests run in < 5 minutes
- CI/CD pipeline passing on all PRs

## Definition of Done
- [ ] 90%+ coverage achieved
- [ ] All test categories implemented
- [ ] CI/CD pipeline working
- [ ] Documentation complete"""
    },
    {
        "title": "Build Complete Project Documentation",
        "body": """## Objective
Create comprehensive documentation for users and contributors.

## Acceptance Criteria
- [ ] Update README with all new features
- [ ] Create detailed installation guide
- [ ] Add configuration documentation
- [ ] Create API documentation using Sphinx
- [ ] Add trading strategy guides
- [ ] Create troubleshooting guide
- [ ] Add contribution guidelines
- [ ] Create architecture diagrams

## Files to Create/Modify
- `README.md` (update)
- `docs/installation.md` (new)
- `docs/configuration.md` (new)
- `docs/api/` (new directory)
- `docs/strategies/` (new directory)

## Success Metrics
- All public APIs documented
- Installation guide tested on fresh machine
- Architecture diagrams up to date

## Definition of Done
- [ ] All documentation written
- [ ] API docs generated
- [ ] Installation guide verified
- [ ] Architecture diagrams complete"""
    },
    {
        "title": "Build Performance Benchmarking System",
        "body": """## Objective
Create a system to benchmark the bot's performance against market indices and other strategies.

## Acceptance Criteria
- [ ] Create benchmark comparison tools
- [ ] Add market index comparisons (S&P 500, NIFTY)
- [ ] Implement risk-adjusted metrics
- [ ] Create performance reports
- [ ] Add statistical significance tests
- [ ] Create performance visualization
- [ ] Add benchmark alerts when underperforming

## Files to Create/Modify
- `analysis/benchmarking.py` (new)
- `reports/performance_report.py` (new)
- `tests/test_benchmarking.py` (new)

## Success Metrics
- Benchmark comparisons accurate
- Reports generated automatically
- Statistical tests valid

## Definition of Done
- [ ] Benchmark comparisons working
- [ ] Reports generating correctly
- [ ] Statistical tests implemented
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Extend Support to Multiple Asset Classes",
        "body": """## Objective
Extend the bot to support trading in multiple asset classes beyond stocks.

## Acceptance Criteria
- [ ] Add cryptocurrency support (Bitcoin, Ethereum, etc.)
- [ ] Implement forex trading support
- [ ] Add commodity futures support
- [ ] Create asset-specific strategies
- [ ] Add cross-asset correlation analysis
- [ ] Implement asset allocation across classes
- [ ] Add asset-specific risk management

## Files to Create/Modify
- `agents/crypto_agent.py` (new)
- `agents/forex_agent.py` (new)
- `trading/strategies/crypto_strategy.py` (new)
- `tests/test_multi_asset.py` (new)

## Success Metrics
- Crypto trading working end-to-end
- Cross-asset correlation accurate
- Asset allocation optimized

## Definition of Done
- [ ] Crypto support implemented
- [ ] Forex support implemented
- [ ] Cross-asset analysis working
- [ ] Tests passing
- [ ] Documentation complete"""
    },
    {
        "title": "Create Community Strategy Sharing Platform",
        "body": """## Objective
Build a platform for users to share and discover trading strategies within the open source community.

## Acceptance Criteria
- [ ] Create strategy upload and sharing system
- [ ] Add strategy rating and review system
- [ ] Implement strategy backtesting before sharing
- [ ] Create strategy marketplace/registry
- [ ] Add strategy versioning
- [ ] Implement strategy import/export
- [ ] Add community features (comments, forks)

## Files to Create/Modify
- `community/strategy_registry.py` (new)
- `community/strategy_validator.py` (new)
- `docs/strategy_sharing.md` (new)

## Success Metrics
- Strategy upload/download working
- Backtesting validation before publishing
- Community engagement metrics

## Definition of Done
- [ ] Strategy registry implemented
- [ ] Validation working
- [ ] Import/export functional
- [ ] Documentation complete"""
    },
    {
        "title": "Build Mobile Application for Bot Monitoring",
        "body": """## Objective
Create a mobile app for monitoring and controlling the trading bot remotely.

## Acceptance Criteria
- [ ] Create React Native mobile app
- [ ] Add real-time portfolio monitoring
- [ ] Implement push notifications for trades
- [ ] Add emergency stop controls
- [ ] Create performance dashboard
- [ ] Add trade history view
- [ ] Implement biometric security
- [ ] Add dark/light mode support

## Files to Create/Modify
- `mobile/` (new directory - React Native project)
- `api/mobile_api.py` (new - REST API for mobile)
- `docs/mobile_setup.md` (new)

## Success Metrics
- App loads in < 3 seconds
- Push notifications delivered within 30 seconds
- Emergency stop works reliably

## Definition of Done
- [ ] Mobile app functional on iOS and Android
- [ ] Push notifications working
- [ ] Emergency controls tested
- [ ] App store ready
- [ ] Documentation complete"""
    },
    {
        "title": "Improve Scalping Strategy Win Rate to 60%+ Target",
        "body": """## Objective
The scalping module infrastructure is complete and working. This issue tracks improving the strategy win rate from current ~33% to the target 60%+ through better signal filtering, ML integration, and strategy optimization.

## Current State
- Scalping module built with 3 strategies: EMA Crossover, VWAP Bounce, RSI Scalp
- Backtesting engine with realistic cost modeling (brokerage, STT, slippage)
- Paper trading engine with validation gate
- Current best: EMA strategy on RELIANCE.NS - 33% win rate, PF 1.40

## Target Benchmarks
Conservative mode:
- Win rate: 60-65%
- Profit per trade: 0.2-0.3%
- Stop loss: 0.1-0.15%
- Daily net: 1-2% after all costs
- Monthly: 20-40% on capital

Aggressive mode:
- Win rate: 65-70%
- Profit per trade: 0.3-0.5%
- Daily net: 2-3% after all costs
- Monthly: 40-60% on capital

## Acceptance Criteria
- [ ] Improve EMA strategy win rate to 60%+ on 3+ months of data
- [ ] Add ML-based signal confirmation layer
- [ ] Implement adaptive stop loss based on volatility regime
- [ ] Add market session filter (only trade 9:15-11:15 AM for NSE)
- [ ] Add news/event filter (avoid trading around major announcements)
- [ ] Implement ensemble signal scoring across all 3 strategies
- [ ] Validate on minimum 500 trades across different market conditions
- [ ] Add BANKNIFTY futures support (higher liquidity, better for scalping)

## Known Issues to Fix
- 7-day data limitation from yfinance (need paid data feed for proper backtesting)
- Risk/reward ratio currently inverted (losses > wins in size)
- Need minimum 3-candle hold causing some TP misses
- VWAP strategy needs VWAP reset per trading day (currently cumulative)

## Technical Approach
1. Integrate ML prediction confidence as signal filter
2. Use ATR-based dynamic position sizing
3. Add regime detection (trending vs choppy market)
4. Only trade when volatility is in optimal range
5. Implement partial profit taking (50% at 1:1, rest at 1:3)

## Files to Modify
- `scalping/strategies/ema_crossover.py`
- `scalping/strategies/vwap_strategy.py`
- `scalping/strategies/rsi_scalp.py`
- `scalping/backtester.py`
- `scalping/config.py`

## References
- Current implementation: `scalping/` directory
- Benchmark targets: `scalping/config.py` (CONSERVATIVE, AGGRESSIVE dicts)
- Run backtest: `python scalping/run_scalping.py --mode backtest --symbol RELIANCE.NS --strategy ema`

## Definition of Done
- [ ] 60%+ win rate validated on 500+ trades
- [ ] Positive daily returns consistently
- [ ] Paper trading passes validation gate
- [ ] Documentation updated with new strategy logic"""
    }
]

def check_gh_cli():
    try:
        result = subprocess.run(['gh', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("GitHub CLI not found. Please ensure it is installed and in PATH.")
            return False
        result = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            print("GitHub CLI not authenticated. Please run: gh auth login")
            return False
        print("GitHub CLI ready.")
        return True
    except FileNotFoundError:
        print("GitHub CLI not found. Please install it from https://cli.github.com/")
        return False

def create_issue(title, body):
    cmd = ['gh', 'issue', 'create', '--title', title, '--body', body, '--repo', REPO]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        url = result.stdout.strip()
        print(f"  Created: {title}")
        print(f"  URL: {url}")
        return True
    else:
        print(f"  Failed: {title}")
        print(f"  Error: {result.stderr.strip()}")
        return False

def main():
    print("GitHub Issues Creator - Stock AI Project")
    print(f"Repository: {REPO}")
    print(f"Total issues to create: {len(ISSUES)}")
    print("=" * 50)

    if not check_gh_cli():
        sys.exit(1)

    response = input(f"\nCreate all {len(ISSUES)} issues? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)

    print("\nCreating issues...\n")
    success = 0
    failed = 0

    for i, issue in enumerate(ISSUES, 1):
        print(f"[{i}/{len(ISSUES)}]", end=" ")
        if create_issue(issue['title'], issue['body']):
            success += 1
        else:
            failed += 1
        print()

    print("=" * 50)
    print(f"Done! Created: {success} | Failed: {failed}")
    print(f"View issues: https://github.com/{REPO}/issues")

if __name__ == "__main__":
    main()
