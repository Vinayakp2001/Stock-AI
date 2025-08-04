"""
Main Application
Stock Prediction Agent SDK - Main entry point
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.data_agent import DataAgent
from agents.prediction_agent import PredictionAgent
from backtesting.engine import BacktestEngine, MovingAverageStrategy, RSIStrategy, MACDStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPredictionApp:
    """Main application class for the Stock Prediction Agent SDK"""
    
    def __init__(self):
        self.data_agent = DataAgent()
        self.prediction_agent = PredictionAgent()
        self.backtest_engine = BacktestEngine(initial_capital=100000)
        
        # Default symbols to analyze (including Indian stocks)
        self.default_symbols = [
            # US Stocks
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX',
            # Indian Stocks
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'AXISBANK.NS',
            # Market Indices
            '^NSEI', '^BSESN', '^GSPC', '^IXIC', '^DJI'
        ]
    
    def analyze_stock(self, symbol: str, period: str = "1y", show_indicators: bool = False):
        """Analyze a single stock"""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {symbol}")
        print(f"{'='*60}")
        
        try:
            # Get stock data
            print("📊 Fetching stock data...")
            stock_data = self.data_agent.get_stock_data(symbol, period=period)
            
            current_price = stock_data.data['Close'].iloc[-1]
            print(f"💰 Current Price: ${current_price:.2f}")
            
            # Show metadata
            if stock_data.metadata:
                print(f"🏢 Company: {stock_data.metadata.get('name', 'Unknown')}")
                print(f"📈 Sector: {stock_data.metadata.get('sector', 'Unknown')}")
                print(f"📊 Market Cap: ${stock_data.metadata.get('market_cap', 0):,.0f}")
                print(f"📉 P/E Ratio: {stock_data.metadata.get('pe_ratio', 0):.2f}")
            
            # Calculate support/resistance
            levels = self.data_agent.get_support_resistance(stock_data)
            print(f"🛡️  Support Level: ${levels['support']:.2f}")
            print(f"🚀 Resistance Level: ${levels['resistance']:.2f}")
            
            # Show technical indicators
            if show_indicators:
                print("\n📈 Technical Indicators:")
                indicators = stock_data.indicators
                latest_values = {name: values.iloc[-1] for name, values in indicators.items() if not pd.isna(values.iloc[-1])}
                
                for name, value in latest_values.items():
                    if 'rsi' in name:
                        print(f"   RSI: {value:.2f}")
                    elif 'macd' in name and 'signal' not in name and 'histogram' not in name:
                        print(f"   MACD: {value:.4f}")
                    elif 'sma_20' in name:
                        print(f"   SMA 20: ${value:.2f}")
                    elif 'sma_50' in name:
                        print(f"   SMA 50: ${value:.2f}")
            
            # Prepare data for ML
            print("\n🤖 Preparing data for machine learning...")
            X, y = self.data_agent.prepare_ml_data(stock_data, target_days=1)
            # Add data check to prevent ML errors
            if X.empty or y.empty or len(X) < 10:
                print("❌ Not enough data for training. Try a longer period or check your data source.")
                logger.error("Not enough data for training. X or y is empty or too small.")
                return None, None
            
            # Train models
            print("🧠 Training prediction models...")
            self.prediction_agent.train_models(X, y, symbol)
            
            # Make prediction
            print("🔮 Making price prediction...")
            latest_features = X.tail(1)
            prediction = self.prediction_agent.predict(latest_features, symbol)
            
            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"   Predicted Price: ${prediction.predicted_price:.2f}")
            print(f"   Confidence: {prediction.confidence:.2%}")
            print(f"   Signal: {prediction.signal}")
            print(f"   Model Used: {prediction.model_name}")
            
            # Calculate expected return
            expected_return = (prediction.predicted_price - current_price) / current_price
            print(f"   Expected Return: {expected_return:.2%}")
            
            return stock_data, prediction
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            return None, None
    
    def backtest_strategies(self, symbol: str, period: str = "1y"):
        """Backtest multiple trading strategies"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING STRATEGIES: {symbol}")
        print(f"{'='*60}")
        
        try:
            # Get stock data
            stock_data = self.data_agent.get_stock_data(symbol, period=period)
            data = stock_data.data
            
            # Create strategies
            strategies = [
                MovingAverageStrategy(20, 50),
                MovingAverageStrategy(10, 30),
                RSIStrategy(14, 30, 70),
                RSIStrategy(14, 20, 80),
                MACDStrategy(12, 26, 9),
                MACDStrategy(8, 21, 5)
            ]
            
            # Run backtests
            print("🔄 Running backtests...")
            results = self.backtest_engine.compare_strategies(strategies, data, symbol)
            
            # Display results
            print(f"\n📊 BACKTEST RESULTS:")
            print(f"{'Strategy':<20} {'Return':<10} {'Win Rate':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8}")
            print("-" * 70)
            
            best_strategy = None
            best_return = -float('inf')
            
            for strategy_name, result in results.items():
                print(f"{strategy_name:<20} {result.total_return_pct:>8.2%} {result.win_rate:>8.2%} "
                      f"{result.sharpe_ratio:>6.2f} {result.max_drawdown_pct:>6.2%} {result.total_trades:>6}")
                
                if result.total_return_pct > best_return:
                    best_return = result.total_return_pct
                    best_strategy = result
            
            if best_strategy:
                print(f"\n🏆 BEST STRATEGY: {best_strategy.strategy_name}")
                print(f"   Total Return: {best_strategy.total_return_pct:.2%}")
                print(f"   Annualized Return: {best_strategy.annualized_return:.2%}")
                print(f"   Sharpe Ratio: {best_strategy.sharpe_ratio:.2f}")
                print(f"   Max Drawdown: {best_strategy.max_drawdown_pct:.2%}")
                print(f"   Win Rate: {best_strategy.win_rate:.2%}")
                print(f"   Total Trades: {best_strategy.total_trades}")
                print(f"   Profit Factor: {best_strategy.profit_factor:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {str(e)}")
            print(f"❌ Error backtesting {symbol}: {str(e)}")
            return {}
    
    def compare_stocks(self, symbols: List[str], period: str = "1y"):
        """Compare multiple stocks"""
        print(f"\n{'='*60}")
        print(f"COMPARING STOCKS: {', '.join(symbols)}")
        print(f"{'='*60}")
        
        results = {}
        
        for symbol in symbols:
            try:
                stock_data = self.data_agent.get_stock_data(symbol, period=period)
                current_price = stock_data.data['Close'].iloc[-1]
                
                # Prepare data for ML
                X, y = self.data_agent.prepare_ml_data(stock_data, target_days=1)
                
                # Train models
                self.prediction_agent.train_models(X, y, symbol)
                
                # Make prediction
                latest_features = X.tail(1)
                prediction = self.prediction_agent.predict(latest_features, symbol)
                
                results[symbol] = {
                    'current_price': current_price,
                    'predicted_price': prediction.predicted_price,
                    'expected_return': (prediction.predicted_price - current_price) / current_price,
                    'confidence': prediction.confidence,
                    'signal': prediction.signal
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Display comparison
        print(f"\n📊 STOCK COMPARISON:")
        print(f"{'Symbol':<8} {'Current':<10} {'Predicted':<10} {'Return':<10} {'Confidence':<12} {'Signal':<6}")
        print("-" * 60)
        
        for symbol, result in results.items():
            print(f"{symbol:<8} ${result['current_price']:<9.2f} ${result['predicted_price']:<9.2f} "
                  f"{result['expected_return']:>8.2%} {result['confidence']:>10.2%} {result['signal']:>6}")
        
        return results
    
    def run_demo(self):
        """Run a comprehensive demo"""
        print("🚀 STOCK PREDICTION AGENT SDK - DEMO")
        print("=" * 60)
        
        # Analyze a popular Indian stock
        symbol = "RELIANCE.NS"
        print(f"\n1️⃣ Analyzing {symbol}...")
        stock_data, prediction = self.analyze_stock(symbol, period="1y", show_indicators=True)
        
        if stock_data and prediction:
            # Backtest strategies
            print(f"\n2️⃣ Backtesting strategies on {symbol}...")
            backtest_results = self.backtest_strategies(symbol, period="1y")
            
            # Compare multiple stocks (mix of US and Indian)
            print(f"\n3️⃣ Comparing multiple stocks...")
            comparison_results = self.compare_stocks(['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'AAPL'], period="1y")
            
            print(f"\n✅ Demo completed successfully!")
            print(f"📈 You can now use these results to make informed trading decisions.")
            print(f"⚠️  Remember: Past performance doesn't guarantee future results!")
        
        else:
            print("❌ Demo failed. Please check your internet connection and try again.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Stock Prediction Agent SDK")
    parser.add_argument("--symbol", "-s", help="Stock symbol to analyze")
    parser.add_argument("--period", "-p", default="1y", help="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)")
    parser.add_argument("--backtest", "-b", action="store_true", help="Run backtesting")
    parser.add_argument("--compare", "-c", nargs="+", help="Compare multiple stocks")
    parser.add_argument("--demo", "-d", action="store_true", help="Run demo")
    parser.add_argument("--indicators", "-i", action="store_true", help="Show technical indicators")
    
    args = parser.parse_args()
    
    app = StockPredictionApp()
    
    try:
        if args.demo:
            app.run_demo()
        elif args.symbol:
            if args.backtest:
                app.backtest_strategies(args.symbol, args.period)
            else:
                app.analyze_stock(args.symbol, args.period, args.indicators)
        elif args.compare:
            app.compare_stocks(args.compare, args.period)
        else:
            # Default: analyze AAPL
            app.analyze_stock("AAPL", "6mo", True)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"❌ Application error: {str(e)}")

if __name__ == "__main__":
    main() 