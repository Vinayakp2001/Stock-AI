"""
Prediction Agent
Uses machine learning models to predict stock prices and generate trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Container for prediction results"""
    symbol: str
    predicted_price: float
    confidence: float
    signal: str  # 'BUY', 'SELL', 'HOLD'
    model_name: str
    timestamp: datetime
    features_used: List[str]
    model_performance: Dict[str, float]
    # New fields for multiple timeframes
    timeframe: str = "1d"  # '1d', '1w', '1m'
    prediction_intervals: Dict[str, float] = None  # Confidence intervals
    risk_adjusted_return: float = None

@dataclass
class MultiTimeframePrediction:
    """Container for multiple timeframe predictions"""
    symbol: str
    predictions: Dict[str, PredictionResult]  # timeframe -> prediction
    overall_signal: str
    overall_confidence: float
    risk_metrics: Dict[str, float]

class PredictionAgent:
    """Agent responsible for making stock price predictions"""
    
    def __init__(self, models_dir: str = "models/"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all prediction models with better hyperparameters"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150, 
                max_depth=6, 
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=150, 
                max_depth=6, 
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=150, 
                max_depth=6, 
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, symbol: str):
        """
        Train all prediction models with improved validation
        
        Args:
            X: Feature matrix
            y: Target variable
            symbol: Stock symbol for model identification
        """
        logger.info(f"Training models for {symbol} with {len(X)} samples")
        
        # Ensure we have enough data
        if len(X) < 100:
            logger.warning(f"Insufficient data for training: {len(X)} samples")
            return
        
        # Split data with more recent data for testing
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name} for {symbol}")
                
                # Scale features for non-tree models
                if model_name in ['linear_regression']:
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate performance metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store performance
                self.model_performance[f"{symbol}_{model_name}"] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[f"{symbol}_{model_name}"] = dict(
                        zip(X.columns, model.feature_importances_)
                    )
                
                logger.info(f"{model_name} trained successfully. R²: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Save models
        self._save_models(symbol)
    
    def predict(self, X: pd.DataFrame, symbol: str, model_name: str = None) -> PredictionResult:
        """
        Make prediction using specified or best model
        
        Args:
            X: Feature matrix for prediction
            symbol: Stock symbol
            model_name: Specific model to use (optional)
        
        Returns:
            PredictionResult object
        """
        try:
            # Load models if not already loaded
            if not self.models:
                self._load_models(symbol)
            
            # Select best model if none specified
            if model_name is None:
                model_name = self._get_best_model(symbol)
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            # Scale features if needed
            if model_name in ['linear_regression']:
                X_scaled = self.scalers[model_name].transform(X)
            else:
                X_scaled = X
            
            # Make prediction (this is now a return prediction)
            predicted_return = self.models[model_name].predict(X_scaled)[0]
            
            # Convert return to price prediction
            # We need to get the current price from the data
            current_price = self._get_current_price(X, symbol)
            predicted_price = current_price * (1 + predicted_return)
            
            # Calculate confidence based on model performance
            confidence = self._calculate_confidence(symbol, model_name)
            
            # Generate trading signal
            signal = self._generate_signal(predicted_return, X, symbol)
            
            return PredictionResult(
                symbol=symbol,
                predicted_price=predicted_price,
                confidence=confidence,
                signal=signal,
                model_name=model_name,
                timestamp=datetime.now(),
                features_used=list(X.columns),
                model_performance=self.model_performance.get(f"{symbol}_{model_name}", {})
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_multiple_timeframes(self, X: pd.DataFrame, symbol: str) -> MultiTimeframePrediction:
        """
        Make predictions for multiple timeframes (1 day, 1 week, 1 month)
        
        Args:
            X: Feature matrix for prediction
            symbol: Stock symbol
        
        Returns:
            MultiTimeframePrediction object
        """
        timeframes = {
            '1d': 1,
            '1w': 5,
            '1m': 21
        }
        
        predictions = {}
        
        for timeframe, days in timeframes.items():
            try:
                # Prepare data for specific timeframe
                X_timeframe, y_timeframe = self._prepare_timeframe_data(X, symbol, days)
                
                if X_timeframe.empty or y_timeframe.empty or len(X_timeframe) < 10:
                    continue
                
                # Train models for this timeframe
                self.train_models(X_timeframe, y_timeframe, f"{symbol}_{timeframe}")
                
                # Make prediction
                latest_features = X_timeframe.tail(1)
                prediction = self.predict(latest_features, f"{symbol}_{timeframe}")
                
                # Add timeframe info
                prediction.timeframe = timeframe
                prediction.prediction_intervals = self._calculate_prediction_intervals(prediction, X_timeframe)
                prediction.risk_adjusted_return = self._calculate_risk_adjusted_return(prediction, X_timeframe)
                
                predictions[timeframe] = prediction
                
            except Exception as e:
                logger.warning(f"Error predicting {timeframe} for {symbol}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError(f"No valid predictions for {symbol}")
        
        # Calculate overall metrics
        overall_signal = self._calculate_overall_signal(predictions)
        overall_confidence = np.mean([p.confidence for p in predictions.values()])
        risk_metrics = self._calculate_risk_metrics(predictions)
        
        return MultiTimeframePrediction(
            symbol=symbol,
            predictions=predictions,
            overall_signal=overall_signal,
            overall_confidence=overall_confidence,
            risk_metrics=risk_metrics
        )
    
    def _prepare_timeframe_data(self, X: pd.DataFrame, symbol: str, target_days: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for specific timeframe prediction"""
        # This would need to be implemented based on your data structure
        # For now, we'll use the existing prepare_ml_data method
        from agents.data_agent import DataAgent
        data_agent = DataAgent()
        
        # Get fresh data for the symbol
        stock_data = data_agent.get_stock_data(symbol, period="2y")  # Get more data for longer timeframes
        X_timeframe, y_timeframe = data_agent.prepare_ml_data(stock_data, target_days=target_days)
        
        return X_timeframe, y_timeframe
    
    def _calculate_prediction_intervals(self, prediction: PredictionResult, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate prediction confidence intervals"""
        # Simple confidence intervals based on model performance
        confidence_level = prediction.confidence
        
        # Calculate standard deviation of recent predictions
        if hasattr(self, 'models') and prediction.model_name in self.models:
            model = self.models[prediction.model_name]
            recent_predictions = model.predict(X.tail(30))  # Last 30 predictions
            std_dev = np.std(recent_predictions)
            
            # Calculate intervals
            lower_bound = prediction.predicted_price - (1.96 * std_dev)  # 95% confidence
            upper_bound = prediction.predicted_price + (1.96 * std_dev)
            
            return {
                'lower_95': max(0, lower_bound),
                'upper_95': upper_bound,
                'lower_68': max(0, prediction.predicted_price - std_dev),  # 68% confidence
                'upper_68': prediction.predicted_price + std_dev
            }
        
        return {
            'lower_95': prediction.predicted_price * 0.9,
            'upper_95': prediction.predicted_price * 1.1,
            'lower_68': prediction.predicted_price * 0.95,
            'upper_68': prediction.predicted_price * 1.05
        }
    
    def _calculate_risk_adjusted_return(self, prediction: PredictionResult, X: pd.DataFrame) -> float:
        """Calculate risk-adjusted return (Sharpe ratio)"""
        try:
            # Get the predicted return from the model
            # We need to calculate this from the predicted price
            current_price = self._get_current_price(X, prediction.symbol)
            expected_return = (prediction.predicted_price - current_price) / current_price
            
            # Calculate volatility from price change features if available
            volatility = 0.02  # Default 2% volatility
            if 'price_change' in X.columns:
                volatility = X['price_change'].std()
            elif 'volatility_5d' in X.columns:
                volatility = X['volatility_5d'].iloc[-1]
            
            # Calculate risk-adjusted return (Sharpe ratio)
            if volatility > 0:
                risk_free_rate = 0.05  # 5% risk-free rate
                risk_adjusted_return = (expected_return - risk_free_rate) / volatility
            else:
                risk_adjusted_return = 0
            
            return risk_adjusted_return
            
        except Exception as e:
            logger.warning(f"Error calculating risk-adjusted return: {str(e)}")
            return 0
    
    def _calculate_overall_signal(self, predictions: Dict[str, PredictionResult]) -> str:
        """Calculate overall signal based on multiple timeframes"""
        signals = [p.signal for p in predictions.values()]
        confidences = [p.confidence for p in predictions.values()]
        
        # Weight signals by confidence
        buy_weight = sum(confidences[i] for i, signal in enumerate(signals) if signal == 'BUY')
        sell_weight = sum(confidences[i] for i, signal in enumerate(signals) if signal == 'SELL')
        hold_weight = sum(confidences[i] for i, signal in enumerate(signals) if signal == 'HOLD')
        
        if buy_weight > sell_weight and buy_weight > hold_weight:
            return 'BUY'
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_risk_metrics(self, predictions: Dict[str, PredictionResult]) -> Dict[str, float]:
        """Calculate overall risk metrics"""
        risk_adjusted_returns = [p.risk_adjusted_return for p in predictions.values() if p.risk_adjusted_return is not None]
        confidences = [p.confidence for p in predictions.values()]
        
        return {
            'avg_risk_adjusted_return': np.mean(risk_adjusted_returns) if risk_adjusted_returns else 0,
            'avg_confidence': np.mean(confidences),
            'signal_consistency': len(set(p.signal for p in predictions.values())),  # Lower = more consistent
            'timeframe_coverage': len(predictions)
        }
    
    def ensemble_predict(self, X: pd.DataFrame, symbol: str, weights: Dict[str, float] = None) -> PredictionResult:
        """
        Make ensemble prediction using multiple models
        
        Args:
            X: Feature matrix for prediction
            symbol: Stock symbol
            weights: Weights for each model (optional)
        
        Returns:
            PredictionResult object
        """
        predictions = []
        confidences = []
        
        # Load models if needed
        if not self.models:
            self._load_models(symbol)
        
        # Get predictions from all models
        for model_name in self.models.keys():
            try:
                if model_name in ['linear_regression']:
                    X_scaled = self.scalers[model_name].transform(X)
                else:
                    X_scaled = X
                
                pred = self.models[model_name].predict(X_scaled)[0]
                confidence = self._calculate_confidence(symbol, model_name)
                
                predictions.append(pred)
                confidences.append(confidence)
                
            except Exception as e:
                logger.warning(f"Error with {model_name}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # Calculate weighted average
        if weights is None:
            # Use equal weights
            weights = {name: 1.0/len(predictions) for name in self.models.keys()}
        
        # Calculate ensemble prediction
        ensemble_pred = np.average(predictions, weights=list(weights.values()))
        ensemble_confidence = np.mean(confidences)
        
        # Generate signal
        signal = self._generate_signal(ensemble_pred, X, symbol)
        
        return PredictionResult(
            symbol=symbol,
            predicted_price=ensemble_pred,
            confidence=ensemble_confidence,
            signal=signal,
            model_name="ensemble",
            timestamp=datetime.now(),
            features_used=list(X.columns),
            model_performance={'ensemble_confidence': ensemble_confidence}
        )
    
    def _get_best_model(self, symbol: str) -> str:
        """Get the best performing model for a symbol"""
        best_model = None
        best_r2 = -float('inf')
        
        for model_name in self.models.keys():
            perf_key = f"{symbol}_{model_name}"
            if perf_key in self.model_performance:
                r2 = self.model_performance[perf_key]['r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
        
        return best_model or 'random_forest'
    
    def _calculate_confidence(self, symbol: str, model_name: str) -> float:
        """Calculate prediction confidence based on model performance"""
        perf_key = f"{symbol}_{model_name}"
        if perf_key in self.model_performance:
            r2 = self.model_performance[perf_key]['r2']
            
            # Improved confidence calculation
            if r2 < 0:
                # Negative R² means model is worse than just predicting the mean
                confidence = max(5, 10 + (r2 * 10))  # 5-10% for poor models
            elif r2 < 0.1:
                # Very low R²
                confidence = 10 + (r2 * 100)  # 10-20%
            elif r2 < 0.3:
                # Low R²
                confidence = 20 + (r2 * 100)  # 20-50%
            elif r2 < 0.5:
                # Moderate R²
                confidence = 50 + (r2 * 60)  # 50-80%
            else:
                # Good R²
                confidence = 80 + (r2 * 20)  # 80-100%
            
            return max(5, min(95, confidence))  # Clamp between 5% and 95%
        
        # Default confidence based on model type
        default_confidences = {
            'random_forest': 45,
            'gradient_boosting': 50,
            'xgboost': 55,
            'lightgbm': 52,
            'linear_regression': 35,
            'ensemble': 60
        }
        return default_confidences.get(model_name, 40)
    
    def _get_current_price(self, X: pd.DataFrame, symbol: str) -> float:
        """Get current price for the symbol"""
        try:
            # Try to get from data agent
            from agents.data_agent import DataAgent
            data_agent = DataAgent()
            stock_data = data_agent.get_stock_data(symbol, period="1d")
            return stock_data.data['Close'].iloc[-1]
        except:
            # Fallback: use a reasonable default
            return 1000.0  # Default price
    
    def _generate_signal(self, predicted_return: float, X: pd.DataFrame, symbol: str) -> str:
        """Generate trading signal based on prediction"""
        # predicted_return is already a percentage change (e.g., 0.05 for 5% increase)
        
        # Generate signal based on threshold
        if predicted_return > 0.02:  # 2% increase
            return 'BUY'
        elif predicted_return < -0.02:  # 2% decrease
            return 'SELL'
        else:
            return 'HOLD'
    
    def _save_models(self, symbol: str):
        """Save trained models to disk"""
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                model_path = f"{self.models_dir}/{symbol}_{model_name}.joblib"
                joblib.dump(model, model_path)
                
                # Save scaler if exists
                if model_name in self.scalers:
                    scaler_path = f"{self.models_dir}/{symbol}_{model_name}_scaler.joblib"
                    joblib.dump(self.scalers[model_name], scaler_path)
                
                logger.info(f"Saved {model_name} model for {symbol}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
    
    def _load_models(self, symbol: str):
        """Load trained models from disk"""
        for model_name in self.models.keys():
            try:
                model_path = f"{self.models_dir}/{symbol}_{model_name}.joblib"
                self.models[model_name] = joblib.load(model_path)
                
                # Load scaler if exists
                scaler_path = f"{self.models_dir}/{symbol}_{model_name}_scaler.joblib"
                if os.path.exists(scaler_path):
                    self.scalers[model_name] = joblib.load(scaler_path)
                
                logger.info(f"Loaded {model_name} model for {symbol}")
                
            except Exception as e:
                logger.warning(f"Could not load {model_name} for {symbol}: {str(e)}")
    
    def get_model_performance(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models"""
        performance = {}
        for model_name in self.models.keys():
            perf_key = f"{symbol}_{model_name}"
            if perf_key in self.model_performance:
                performance[model_name] = self.model_performance[perf_key]
        return performance
    
    def get_feature_importance(self, symbol: str, model_name: str = None) -> Dict[str, float]:
        """Get feature importance for a model"""
        if model_name:
            key = f"{symbol}_{model_name}"
            return self.feature_importance.get(key, {})
        else:
            # Return feature importance for best model
            best_model = self._get_best_model(symbol)
            key = f"{symbol}_{best_model}"
            return self.feature_importance.get(key, {})

# Example usage
if __name__ == "__main__":
    from agents.data_agent import DataAgent
    
    # Initialize agents
    data_agent = DataAgent()
    prediction_agent = PredictionAgent()
    
    # Get data and prepare for ML
    apple_data = data_agent.get_stock_data('AAPL', period='1y')
    X, y = data_agent.prepare_ml_data(apple_data, target_days=1)
    
    # Train models
    prediction_agent.train_models(X, y, 'AAPL')
    
    # Make prediction
    latest_features = X.tail(1)
    prediction = prediction_agent.predict(latest_features, 'AAPL')
    
    print(f"Predicted Price: ${prediction.predicted_price:.2f}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print(f"Signal: {prediction.signal}")
    print(f"Model Used: {prediction.model_name}") 