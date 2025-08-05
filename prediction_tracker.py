"""
Prediction Tracking & Accuracy Analysis System
Tracks predictions vs actual outcomes and provides detailed analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    EXPIRED = "expired"

@dataclass
class PredictionRecord:
    """Individual prediction record"""
    id: str
    symbol: str
    prediction_date: datetime
    target_date: datetime
    timeframe: str  # '1d', '1w', '1m'
    predicted_price: float
    confidence_score: float
    actual_price: Optional[float] = None
    prediction_features: Dict[str, Any] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    status: PredictionStatus = PredictionStatus.PENDING
    error_percentage: Optional[float] = None
    absolute_error: Optional[float] = None
    accuracy_score: Optional[float] = None
    factors_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics"""
    total_predictions: int
    accurate_predictions: int
    overall_accuracy: float
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    confidence_correlation: float
    accuracy_by_confidence: Dict[str, float]
    accuracy_by_timeframe: Dict[str, float]
    accuracy_by_symbol: Dict[str, float]

class PredictionTracker:
    """Main prediction tracking and analysis system"""
    
    def __init__(self, storage_path: str = "data/predictions"):
        self.storage_path = storage_path
        self.predictions_file = os.path.join(storage_path, "predictions.json")
        self.accuracy_file = os.path.join(storage_path, "accuracy_metrics.json")
        self._ensure_storage_directory()
        self.predictions: List[PredictionRecord] = self._load_predictions()
    
    def _ensure_storage_directory(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _load_predictions(self) -> List[PredictionRecord]:
        """Load existing predictions from storage"""
        if os.path.exists(self.predictions_file):
            try:
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    predictions = []
                    for pred_data in data:
                        pred = PredictionRecord(
                            id=pred_data['id'],
                            symbol=pred_data['symbol'],
                            prediction_date=datetime.fromisoformat(pred_data['prediction_date']),
                            target_date=datetime.fromisoformat(pred_data['target_date']),
                            timeframe=pred_data['timeframe'],
                            predicted_price=pred_data['predicted_price'],
                            actual_price=pred_data.get('actual_price'),
                            confidence_score=pred_data['confidence_score'],
                            prediction_features=pred_data.get('prediction_features', {}),
                            market_conditions=pred_data.get('market_conditions', {}),
                            status=PredictionStatus(pred_data.get('status', 'pending')),
                            error_percentage=pred_data.get('error_percentage'),
                            absolute_error=pred_data.get('absolute_error'),
                            accuracy_score=pred_data.get('accuracy_score'),
                            factors_analysis=pred_data.get('factors_analysis', {})
                        )
                        predictions.append(pred)
                    return predictions
            except Exception as e:
                logger.error(f"Error loading predictions: {e}")
        return []
    
    def _save_predictions(self):
        """Save predictions to storage"""
        try:
            data = []
            for pred in self.predictions:
                pred_data = {
                    'id': pred.id,
                    'symbol': pred.symbol,
                    'prediction_date': pred.prediction_date.isoformat(),
                    'target_date': pred.target_date.isoformat(),
                    'timeframe': pred.timeframe,
                    'predicted_price': pred.predicted_price,
                    'actual_price': pred.actual_price,
                    'confidence_score': pred.confidence_score,
                    'prediction_features': pred.prediction_features,
                    'market_conditions': pred.market_conditions,
                    'status': pred.status.value,
                    'error_percentage': pred.error_percentage,
                    'absolute_error': pred.absolute_error,
                    'accuracy_score': pred.accuracy_score,
                    'factors_analysis': pred.factors_analysis
                }
                data.append(pred_data)
            
            with open(self.predictions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def add_prediction(self, symbol: str, timeframe: str, predicted_price: float, 
                      confidence_score: float, prediction_features: Dict[str, Any] = None,
                      market_conditions: Dict[str, Any] = None) -> str:
        """Add a new prediction to track"""
        
        # Generate prediction ID
        pred_id = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate target date based on timeframe
        target_date = self._calculate_target_date(timeframe)
        
        # Create prediction record
        prediction = PredictionRecord(
            id=pred_id,
            symbol=symbol,
            prediction_date=datetime.now(),
            target_date=target_date,
            timeframe=timeframe,
            predicted_price=predicted_price,
            confidence_score=confidence_score,
            prediction_features=prediction_features or {},
            market_conditions=market_conditions or {}
        )
        
        self.predictions.append(prediction)
        self._save_predictions()
        
        logger.info(f"Added prediction {pred_id} for {symbol} {timeframe}")
        return pred_id
    
    def _calculate_target_date(self, timeframe: str) -> datetime:
        """Calculate target date based on timeframe"""
        now = datetime.now()
        if timeframe == '1d':
            return now + timedelta(days=1)
        elif timeframe == '1w':
            return now + timedelta(weeks=1)
        elif timeframe == '1m':
            return now + timedelta(days=30)
        else:
            return now + timedelta(days=1)
    
    def update_actual_price(self, symbol: str, actual_price: float, 
                           target_date: Optional[datetime] = None) -> bool:
        """Update actual price for completed predictions"""
        updated = False
        
        for pred in self.predictions:
            if pred.symbol == symbol and pred.status == PredictionStatus.PENDING:
                # Check if this prediction has reached its target date
                if target_date is None:
                    target_date = datetime.now()
                
                # For demo purposes, mark all predictions as completed immediately
                # In production, you'd check: if target_date >= pred.target_date:
                pred.actual_price = actual_price
                pred.status = PredictionStatus.COMPLETED
                
                # Calculate accuracy metrics
                self._calculate_prediction_accuracy(pred)
                
                updated = True
                logger.info(f"Updated actual price for prediction {pred.id}")
        
        if updated:
            self._save_predictions()
        
        return updated
    
    def force_complete_all_predictions(self):
        """Force complete all pending predictions for demo purposes"""
        updated = False
        
        for pred in self.predictions:
            if pred.status == PredictionStatus.PENDING and pred.actual_price is None:
                # For demo, set a dummy actual price based on predicted price
                # In real usage, this would be the actual market price
                pred.actual_price = pred.predicted_price * (1 + (pred.confidence_score - 50) / 1000)
                pred.status = PredictionStatus.COMPLETED
                
                # Calculate accuracy metrics
                self._calculate_prediction_accuracy(pred)
                
                updated = True
                logger.info(f"Force completed prediction {pred.id}")
        
        if updated:
            self._save_predictions()
        
        return updated
    
    def _calculate_prediction_accuracy(self, prediction: PredictionRecord):
        """Calculate accuracy metrics for a completed prediction"""
        if prediction.actual_price is None:
            return
        
        # Calculate errors
        prediction.absolute_error = abs(prediction.predicted_price - prediction.actual_price)
        prediction.error_percentage = (prediction.absolute_error / prediction.actual_price) * 100
        
        # Calculate accuracy score (0-100, higher is better)
        max_acceptable_error = prediction.actual_price * 0.1  # 10% tolerance
        if prediction.absolute_error <= max_acceptable_error:
            prediction.accuracy_score = max(0, 100 - prediction.error_percentage)
        else:
            prediction.accuracy_score = 0
        
        # Analyze factors that might have caused errors
        prediction.factors_analysis = self._analyze_prediction_factors(prediction)
    
    def _analyze_prediction_factors(self, prediction: PredictionRecord) -> Dict[str, Any]:
        """Analyze factors that might have influenced prediction accuracy"""
        analysis = {
            'confidence_vs_accuracy': 'high' if prediction.confidence_score > 70 else 'medium' if prediction.confidence_score > 50 else 'low',
            'market_volatility': prediction.market_conditions.get('volatility', 'unknown'),
            'trend_strength': prediction.market_conditions.get('trend_strength', 'unknown'),
            'volume_analysis': prediction.market_conditions.get('volume_trend', 'unknown'),
            'technical_indicators': prediction.prediction_features.get('technical_signals', {}),
            'potential_issues': []
        }
        
        # Identify potential issues
        if prediction.error_percentage > 10:
            analysis['potential_issues'].append('High prediction error - model may need retraining')
        
        if prediction.confidence_score > 80 and prediction.error_percentage > 5:
            analysis['potential_issues'].append('High confidence but poor accuracy - overfitting possible')
        
        if prediction.market_conditions.get('volatility') == 'high' and prediction.error_percentage > 8:
            analysis['potential_issues'].append('High volatility period - model struggled with market noise')
        
        return analysis
    
    def get_accuracy_metrics(self, symbol: Optional[str] = None, 
                           timeframe: Optional[str] = None,
                           days_back: int = 30) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics"""
        
        # Filter predictions
        filtered_predictions = self.predictions.copy()
        
        if symbol:
            filtered_predictions = [p for p in filtered_predictions if p.symbol == symbol]
        
        if timeframe:
            filtered_predictions = [p for p in filtered_predictions if p.timeframe == timeframe]
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered_predictions = [p for p in filtered_predictions if p.prediction_date >= cutoff_date]
        
        # Get completed predictions only
        completed_predictions = [p for p in filtered_predictions if p.status == PredictionStatus.COMPLETED]
        
        if not completed_predictions:
            return AccuracyMetrics(
                total_predictions=0,
                accurate_predictions=0,
                overall_accuracy=0.0,
                mean_absolute_error=0.0,
                mean_squared_error=0.0,
                root_mean_squared_error=0.0,
                mean_absolute_percentage_error=0.0,
                confidence_correlation=0.0,
                accuracy_by_confidence={},
                accuracy_by_timeframe={},
                accuracy_by_symbol={}
            )
        
        # Calculate basic metrics
        total_predictions = len(completed_predictions)
        accurate_predictions = len([p for p in completed_predictions if p.accuracy_score >= 70])
        overall_accuracy = (accurate_predictions / total_predictions) * 100
        
        # Calculate error metrics
        errors = [p.absolute_error for p in completed_predictions]
        error_percentages = [p.error_percentage for p in completed_predictions]
        
        mean_absolute_error = np.mean(errors)
        mean_squared_error = np.mean([e**2 for e in errors])
        root_mean_squared_error = np.sqrt(mean_squared_error)
        mean_absolute_percentage_error = np.mean(error_percentages)
        
        # Calculate confidence correlation
        confidence_scores = [p.confidence_score for p in completed_predictions]
        accuracy_scores = [p.accuracy_score for p in completed_predictions]
        confidence_correlation = np.corrcoef(confidence_scores, accuracy_scores)[0, 1] if len(confidence_scores) > 1 else 0
        
        # Calculate accuracy by confidence levels
        accuracy_by_confidence = self._calculate_accuracy_by_confidence(completed_predictions)
        
        # Calculate accuracy by timeframe
        accuracy_by_timeframe = self._calculate_accuracy_by_timeframe(completed_predictions)
        
        # Calculate accuracy by symbol
        accuracy_by_symbol = self._calculate_accuracy_by_symbol(completed_predictions)
        
        return AccuracyMetrics(
            total_predictions=total_predictions,
            accurate_predictions=accurate_predictions,
            overall_accuracy=overall_accuracy,
            mean_absolute_error=mean_absolute_error,
            mean_squared_error=mean_squared_error,
            root_mean_squared_error=root_mean_squared_error,
            mean_absolute_percentage_error=mean_absolute_percentage_error,
            confidence_correlation=confidence_correlation,
            accuracy_by_confidence=accuracy_by_confidence,
            accuracy_by_timeframe=accuracy_by_timeframe,
            accuracy_by_symbol=accuracy_by_symbol
        )
    
    def _calculate_accuracy_by_confidence(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate accuracy for different confidence levels"""
        confidence_ranges = {
            'Low (0-30%)': (0, 30),
            'Medium (30-70%)': (30, 70),
            'High (70-90%)': (70, 90),
            'Very High (90-100%)': (90, 100)
        }
        
        accuracy_by_confidence = {}
        for label, (min_conf, max_conf) in confidence_ranges.items():
            filtered_preds = [p for p in predictions if min_conf <= p.confidence_score < max_conf]
            if filtered_preds:
                accurate_count = len([p for p in filtered_preds if p.accuracy_score >= 70])
                accuracy_by_confidence[label] = (accurate_count / len(filtered_preds)) * 100
            else:
                accuracy_by_confidence[label] = 0.0
        
        return accuracy_by_confidence
    
    def _calculate_accuracy_by_timeframe(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate accuracy for different timeframes"""
        accuracy_by_timeframe = {}
        timeframes = set(p.timeframe for p in predictions)
        
        for timeframe in timeframes:
            filtered_preds = [p for p in predictions if p.timeframe == timeframe]
            accurate_count = len([p for p in filtered_preds if p.accuracy_score >= 70])
            accuracy_by_timeframe[timeframe] = (accurate_count / len(filtered_preds)) * 100
        
        return accuracy_by_timeframe
    
    def _calculate_accuracy_by_symbol(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Calculate accuracy for different symbols"""
        accuracy_by_symbol = {}
        symbols = set(p.symbol for p in predictions)
        
        for symbol in symbols:
            filtered_preds = [p for p in predictions if p.symbol == symbol]
            accurate_count = len([p for p in filtered_preds if p.accuracy_score >= 70])
            accuracy_by_symbol[symbol] = (accurate_count / len(filtered_preds)) * 100
        
        return accuracy_by_symbol
    
    def get_refinement_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for model improvement"""
        metrics = self.get_accuracy_metrics()
        recommendations = {
            'overall_assessment': '',
            'specific_issues': [],
            'improvement_suggestions': [],
            'priority_actions': []
        }
        
        # Overall assessment
        if metrics.overall_accuracy >= 80:
            recommendations['overall_assessment'] = 'Excellent performance - model is working well'
        elif metrics.overall_accuracy >= 60:
            recommendations['overall_assessment'] = 'Good performance with room for improvement'
        else:
            recommendations['overall_assessment'] = 'Poor performance - significant improvements needed'
        
        # Identify specific issues
        if metrics.confidence_correlation < 0.3:
            recommendations['specific_issues'].append('Low correlation between confidence and accuracy')
        
        if metrics.mean_absolute_percentage_error > 10:
            recommendations['specific_issues'].append('High prediction errors - model accuracy needs improvement')
        
        # Generate improvement suggestions
        if metrics.confidence_correlation < 0.3:
            recommendations['improvement_suggestions'].append('Improve confidence scoring mechanism')
        
        if metrics.mean_absolute_percentage_error > 10:
            recommendations['improvement_suggestions'].append('Retrain model with more recent data')
            recommendations['improvement_suggestions'].append('Add more relevant features to the model')
        
        # Priority actions
        if metrics.overall_accuracy < 60:
            recommendations['priority_actions'].append('Immediate model retraining required')
        elif metrics.mean_absolute_percentage_error > 8:
            recommendations['priority_actions'].append('Feature engineering and model optimization')
        else:
            recommendations['priority_actions'].append('Regular monitoring and incremental improvements')
        
        return recommendations 