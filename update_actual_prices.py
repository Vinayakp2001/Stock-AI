"""
Actual Price Update Script
Automatically updates actual prices for completed predictions to calculate accuracy
"""

import os
import sys
import yfinance as yf
from datetime import datetime, timedelta
from prediction_tracker import PredictionTracker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_actual_prices():
    """Update actual prices for all pending predictions that have reached their target date"""
    
    tracker = PredictionTracker()
    
    # Get all pending predictions
    pending_predictions = [p for p in tracker.predictions if p.status.value == 'pending']
    
    if not pending_predictions:
        logger.info("No pending predictions found.")
        return
    
    logger.info(f"Found {len(pending_predictions)} pending predictions to check.")
    
    # Group predictions by symbol for efficient data fetching
    symbols_to_update = set()
    predictions_by_symbol = {}
    
    for pred in pending_predictions:
        if datetime.now() >= pred.target_date:
            symbols_to_update.add(pred.symbol)
            if pred.symbol not in predictions_by_symbol:
                predictions_by_symbol[pred.symbol] = []
            predictions_by_symbol[pred.symbol].append(pred)
    
    if not symbols_to_update:
        logger.info("No predictions have reached their target date yet.")
        return
    
    logger.info(f"Updating prices for symbols: {list(symbols_to_update)}")
    
    # Update prices for each symbol
    for symbol in symbols_to_update:
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('regularMarketPrice')
            
            if current_price is None:
                logger.warning(f"Could not get current price for {symbol}")
                continue
            
            # Update all predictions for this symbol
            predictions = predictions_by_symbol[symbol]
            for pred in predictions:
                if datetime.now() >= pred.target_date:
                    tracker.update_actual_price(symbol, current_price)
                    logger.info(f"Updated {symbol}: Predicted ${pred.predicted_price:.2f}, Actual ${current_price:.2f}")
        
        except Exception as e:
            logger.error(f"Error updating prices for {symbol}: {str(e)}")
            continue
    
    logger.info("Price update completed.")

def get_accuracy_summary():
    """Get a summary of current prediction accuracy"""
    
    tracker = PredictionTracker()
    metrics = tracker.get_accuracy_metrics()
    
    print("\n" + "="*50)
    print("PREDICTION ACCURACY SUMMARY")
    print("="*50)
    print(f"Total Predictions: {metrics.total_predictions}")
    print(f"Accurate Predictions: {metrics.accurate_predictions}")
    print(f"Overall Accuracy: {metrics.overall_accuracy:.1f}%")
    print(f"Mean Error: {metrics.mean_absolute_percentage_error:.2f}%")
    print(f"Confidence Correlation: {metrics.confidence_correlation:.3f}")
    print(f"Root Mean Square Error: {metrics.root_mean_squared_error:.2f}")
    
    if metrics.accuracy_by_confidence:
        print("\nAccuracy by Confidence Level:")
        for level, accuracy in metrics.accuracy_by_confidence.items():
            print(f"  {level}: {accuracy:.1f}%")
    
    if metrics.accuracy_by_timeframe:
        print("\nAccuracy by Timeframe:")
        for timeframe, accuracy in metrics.accuracy_by_timeframe.items():
            print(f"  {timeframe}: {accuracy:.1f}%")
    
    if metrics.accuracy_by_symbol:
        print("\nAccuracy by Symbol:")
        for symbol, accuracy in metrics.accuracy_by_symbol.items():
            print(f"  {symbol}: {accuracy:.1f}%")
    
    print("="*50)

def get_refinement_recommendations():
    """Get current refinement recommendations"""
    
    tracker = PredictionTracker()
    recommendations = tracker.get_refinement_recommendations()
    
    print("\n" + "="*50)
    print("REFINEMENT RECOMMENDATIONS")
    print("="*50)
    print(f"Overall Assessment: {recommendations['overall_assessment']}")
    
    if recommendations['specific_issues']:
        print("\nIdentified Issues:")
        for issue in recommendations['specific_issues']:
            print(f"  • {issue}")
    
    if recommendations['improvement_suggestions']:
        print("\nImprovement Suggestions:")
        for suggestion in recommendations['improvement_suggestions']:
            print(f"  • {suggestion}")
    
    if recommendations['priority_actions']:
        print("\nPriority Actions:")
        for action in recommendations['priority_actions']:
            print(f"  • {action}")
    
    print("="*50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update actual prices and analyze prediction accuracy")
    parser.add_argument("--update", action="store_true", help="Update actual prices for completed predictions")
    parser.add_argument("--summary", action="store_true", help="Show accuracy summary")
    parser.add_argument("--recommendations", action="store_true", help="Show refinement recommendations")
    parser.add_argument("--all", action="store_true", help="Run all operations")
    
    args = parser.parse_args()
    
    if args.all or args.update:
        print("🔄 Updating actual prices...")
        update_actual_prices()
    
    if args.all or args.summary:
        get_accuracy_summary()
    
    if args.all or args.recommendations:
        get_refinement_recommendations()
    
    if not any([args.update, args.summary, args.recommendations, args.all]):
        print("No operation specified. Use --help for options.")
        print("Example: python update_actual_prices.py --all") 