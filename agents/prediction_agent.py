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
        """Initialize all prediction models"""
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, symbol: str):
        """
        Train all prediction models
        
        Args:
            X: Feature matrix
            y: Target variable
            symbol: Stock symbol for model identification
        """
        logger.info(f"Training models for {symbol}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
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
            
            # Make prediction
            predicted_price = self.models[model_name].predict(X_scaled)[0]
            
            # Calculate confidence based on model performance
            confidence = self._calculate_confidence(symbol, model_name)
            
            # Generate trading signal
            signal = self._generate_signal(predicted_price, X, symbol)
            
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
            # Convert R² to confidence (0-1 scale)
            confidence = max(0, min(1, r2))
            return confidence
        return 0.5  # Default confidence
    
    def _generate_signal(self, predicted_price: float, X: pd.DataFrame, symbol: str) -> str:
        """Generate trading signal based on prediction"""
        # Get current price (assuming it's in the features)
        current_price = X['Close'].iloc[-1] if 'Close' in X.columns else X.iloc[-1, 0]
        
        # Calculate predicted change
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Generate signal based on threshold
        if price_change_pct > 0.02:  # 2% increase
            return 'BUY'
        elif price_change_pct < -0.02:  # 2% decrease
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