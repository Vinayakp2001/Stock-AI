"""
Accuracy Learning Engine
Automatically learns from prediction errors and suggests model improvements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from prediction_tracker import PredictionTracker, PredictionRecord
import logging
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccuracyLearningEngine:
    """Advanced learning engine that analyzes prediction patterns and suggests improvements"""
    
    def __init__(self, storage_path: str = "data/learning"):
        self.storage_path = storage_path
        self.learning_file = os.path.join(storage_path, "learning_insights.json")
        self._ensure_storage_directory()
        self.tracker = PredictionTracker()
    
    def _ensure_storage_directory(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def analyze_error_patterns(self, days_back: int = 90) -> Dict[str, Any]:
        """Analyze patterns in prediction errors to identify improvement opportunities"""
        
        # Get completed predictions
        cutoff_date = datetime.now() - timedelta(days=days_back)
        completed_predictions = [
            p for p in self.tracker.predictions 
            if p.status.value == 'completed' and p.prediction_date >= cutoff_date
        ]
        
        if len(completed_predictions) < 10:
            return {"error": "Insufficient data for pattern analysis"}
        
        # Create analysis dataset
        analysis_data = []
        for pred in completed_predictions:
            data_point = {
                'symbol': pred.symbol,
                'timeframe': pred.timeframe,
                'confidence_score': pred.confidence_score,
                'error_percentage': pred.error_percentage,
                'accuracy_score': pred.accuracy_score,
                'prediction_date': pred.prediction_date,
                'target_date': pred.target_date,
                'predicted_price': pred.predicted_price,
                'actual_price': pred.actual_price,
                'market_volatility': pred.market_conditions.get('volatility', 'unknown'),
                'trend_strength': pred.market_conditions.get('trend_strength', 'unknown'),
                'volume_trend': pred.market_conditions.get('volume_trend', 'unknown')
            }
            analysis_data.append(data_point)
        
        df = pd.DataFrame(analysis_data)
        
        # Pattern 1: Error Clustering Analysis
        error_clusters = self._analyze_error_clusters(df)
        
        # Pattern 2: Confidence vs Accuracy Relationship
        confidence_analysis = self._analyze_confidence_accuracy_relationship(df)
        
        # Pattern 3: Time-based Patterns
        time_patterns = self._analyze_time_patterns(df)
        
        # Pattern 4: Symbol-specific Patterns
        symbol_patterns = self._analyze_symbol_patterns(df)
        
        # Pattern 5: Market Condition Impact
        market_patterns = self._analyze_market_condition_patterns(df)
        
        # Generate improvement recommendations
        recommendations = self._generate_improvement_recommendations(
            error_clusters, confidence_analysis, time_patterns, 
            symbol_patterns, market_patterns
        )
        
        insights = {
            'analysis_period': f"{days_back} days",
            'total_predictions_analyzed': len(completed_predictions),
            'error_clusters': error_clusters,
            'confidence_analysis': confidence_analysis,
            'time_patterns': time_patterns,
            'symbol_patterns': symbol_patterns,
            'market_patterns': market_patterns,
            'recommendations': recommendations,
            'analysis_date': datetime.now().isoformat()
        }
        
        # Save insights
        self._save_insights(insights)
        
        return insights
    
    def _analyze_error_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze clusters of prediction errors to identify common characteristics"""
        
        # Prepare features for clustering
        features = ['confidence_score', 'error_percentage']
        X = df[features].values
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['error_cluster'] = kmeans.fit_predict(X)
        
        # Analyze each cluster
        clusters_analysis = {}
        for cluster_id in range(3):
            cluster_data = df[df['error_cluster'] == cluster_id]
            
            clusters_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_error': cluster_data['error_percentage'].mean(),
                'avg_confidence': cluster_data['confidence_score'].mean(),
                'avg_accuracy': cluster_data['accuracy_score'].mean(),
                'common_symbols': cluster_data['symbol'].value_counts().head(3).to_dict(),
                'common_timeframes': cluster_data['timeframe'].value_counts().to_dict(),
                'characteristics': self._identify_cluster_characteristics(cluster_data)
            }
        
        return clusters_analysis
    
    def _identify_cluster_characteristics(self, cluster_data: pd.DataFrame) -> List[str]:
        """Identify key characteristics of a cluster"""
        characteristics = []
        
        # High error cluster
        if cluster_data['error_percentage'].mean() > 10:
            characteristics.append("High prediction errors")
        
        # High confidence but low accuracy
        if cluster_data['confidence_score'].mean() > 70 and cluster_data['accuracy_score'].mean() < 50:
            characteristics.append("Overconfident predictions")
        
        # Low confidence but high accuracy
        if cluster_data['confidence_score'].mean() < 50 and cluster_data['accuracy_score'].mean() > 70:
            characteristics.append("Underconfident predictions")
        
        # Specific timeframe issues
        timeframe_errors = cluster_data.groupby('timeframe')['error_percentage'].mean()
        problematic_timeframes = timeframe_errors[timeframe_errors > 8].index.tolist()
        if problematic_timeframes:
            characteristics.append(f"Poor performance on timeframes: {problematic_timeframes}")
        
        return characteristics
    
    def _analyze_confidence_accuracy_relationship(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the relationship between confidence scores and accuracy"""
        
        # Calculate correlation
        correlation = df['confidence_score'].corr(df['accuracy_score'])
        
        # Analyze by confidence ranges
        confidence_ranges = [
            (0, 30, 'Low'),
            (30, 60, 'Medium'),
            (60, 80, 'High'),
            (80, 100, 'Very High')
        ]
        
        range_analysis = {}
        for min_conf, max_conf, label in confidence_ranges:
            range_data = df[(df['confidence_score'] >= min_conf) & (df['confidence_score'] < max_conf)]
            if len(range_data) > 0:
                range_analysis[label] = {
                    'count': len(range_data),
                    'avg_accuracy': range_data['accuracy_score'].mean(),
                    'avg_error': range_data['error_percentage'].mean(),
                    'reliability': 'Reliable' if range_data['accuracy_score'].mean() > 70 else 'Unreliable'
                }
        
        return {
            'correlation': correlation,
            'range_analysis': range_analysis,
            'insights': self._generate_confidence_insights(correlation, range_analysis)
        }
    
    def _generate_confidence_insights(self, correlation: float, range_analysis: Dict) -> List[str]:
        """Generate insights about confidence-accuracy relationship"""
        insights = []
        
        if correlation < 0.3:
            insights.append("Low correlation between confidence and accuracy - confidence scoring needs improvement")
        elif correlation > 0.7:
            insights.append("Strong correlation between confidence and accuracy - confidence scoring is working well")
        else:
            insights.append("Moderate correlation between confidence and accuracy - some improvement needed")
        
        # Check for overconfidence
        if 'Very High' in range_analysis and range_analysis['Very High']['avg_accuracy'] < 80:
            insights.append("Very high confidence predictions are not as accurate as expected - potential overfitting")
        
        # Check for underconfidence
        if 'Low' in range_analysis and range_analysis['Low']['avg_accuracy'] > 60:
            insights.append("Low confidence predictions are performing better than expected - model may be too conservative")
        
        return insights
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns related to time and prediction accuracy"""
        
        # Add time-based features
        df['prediction_hour'] = df['prediction_date'].dt.hour
        df['prediction_day'] = df['prediction_date'].dt.day_name()
        df['prediction_month'] = df['prediction_date'].dt.month
        
        # Analyze by timeframe
        timeframe_analysis = df.groupby('timeframe').agg({
            'accuracy_score': ['mean', 'std', 'count'],
            'error_percentage': ['mean', 'std']
        }).round(2)
        
        # Analyze by day of week
        day_analysis = df.groupby('prediction_day').agg({
            'accuracy_score': 'mean',
            'error_percentage': 'mean'
        }).round(2)
        
        # Analyze by hour
        hour_analysis = df.groupby('prediction_hour').agg({
            'accuracy_score': 'mean',
            'error_percentage': 'mean'
        }).round(2)
        
        return {
            'timeframe_performance': timeframe_analysis.to_dict(),
            'day_of_week_performance': day_analysis.to_dict(),
            'hour_of_day_performance': hour_analysis.to_dict(),
            'insights': self._generate_time_insights(timeframe_analysis, day_analysis, hour_analysis)
        }
    
    def _generate_time_insights(self, timeframe_analysis, day_analysis, hour_analysis) -> List[str]:
        """Generate insights about time-based patterns"""
        insights = []
        
        # Find best and worst timeframes
        timeframe_accuracy = timeframe_analysis[('accuracy_score', 'mean')]
        best_timeframe = timeframe_accuracy.idxmax()
        worst_timeframe = timeframe_accuracy.idxmin()
        
        insights.append(f"Best performing timeframe: {best_timeframe} ({timeframe_accuracy[best_timeframe]:.1f}% accuracy)")
        insights.append(f"Worst performing timeframe: {worst_timeframe} ({timeframe_accuracy[worst_timeframe]:.1f}% accuracy)")
        
        # Find best and worst days
        day_accuracy = day_analysis['accuracy_score']
        best_day = day_accuracy.idxmax()
        worst_day = day_accuracy.idxmin()
        
        insights.append(f"Best performing day: {best_day} ({day_accuracy[best_day]:.1f}% accuracy)")
        insights.append(f"Worst performing day: {worst_day} ({day_accuracy[worst_day]:.1f}% accuracy)")
        
        return insights
    
    def _analyze_symbol_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns specific to different symbols"""
        
        # Group by symbol
        symbol_analysis = df.groupby('symbol').agg({
            'accuracy_score': ['mean', 'std', 'count'],
            'error_percentage': ['mean', 'std'],
            'confidence_score': 'mean'
        }).round(2)
        
        # Identify best and worst performing symbols
        symbol_accuracy = symbol_analysis[('accuracy_score', 'mean')]
        best_symbols = symbol_accuracy.nlargest(5)
        worst_symbols = symbol_accuracy.nsmallest(5)
        
        # Analyze symbol characteristics
        symbol_characteristics = {}
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            symbol_characteristics[symbol] = {
                'avg_accuracy': symbol_data['accuracy_score'].mean(),
                'avg_error': symbol_data['error_percentage'].mean(),
                'avg_confidence': symbol_data['confidence_score'].mean(),
                'prediction_count': len(symbol_data),
                'best_timeframe': symbol_data.groupby('timeframe')['accuracy_score'].mean().idxmax(),
                'volatility_impact': self._analyze_symbol_volatility_impact(symbol_data)
            }
        
        return {
            'symbol_performance': symbol_analysis.to_dict(),
            'best_performing_symbols': best_symbols.to_dict(),
            'worst_performing_symbols': worst_symbols.to_dict(),
            'symbol_characteristics': symbol_characteristics,
            'insights': self._generate_symbol_insights(best_symbols, worst_symbols, symbol_characteristics)
        }
    
    def _analyze_symbol_volatility_impact(self, symbol_data: pd.DataFrame) -> str:
        """Analyze how volatility affects prediction accuracy for a symbol"""
        if 'market_volatility' not in symbol_data.columns:
            return "Unknown"
        
        volatility_accuracy = symbol_data.groupby('market_volatility')['accuracy_score'].mean()
        
        if len(volatility_accuracy) > 1:
            best_volatility = volatility_accuracy.idxmax()
            worst_volatility = volatility_accuracy.idxmin()
            return f"Best in {best_volatility} volatility, worst in {worst_volatility} volatility"
        
        return "Insufficient volatility data"
    
    def _generate_symbol_insights(self, best_symbols, worst_symbols, symbol_characteristics) -> List[str]:
        """Generate insights about symbol-specific patterns"""
        insights = []
        
        insights.append(f"Top performing symbols: {list(best_symbols.index)}")
        insights.append(f"Challenging symbols: {list(worst_symbols.index)}")
        
        # Find patterns in best vs worst symbols
        best_avg_confidence = np.mean([symbol_characteristics[s]['avg_confidence'] for s in best_symbols.index])
        worst_avg_confidence = np.mean([symbol_characteristics[s]['avg_confidence'] for s in worst_symbols.index])
        
        if best_avg_confidence > worst_avg_confidence:
            insights.append("Better performing symbols tend to have higher confidence scores")
        else:
            insights.append("No clear relationship between confidence and symbol performance")
        
        return insights
    
    def _analyze_market_condition_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how market conditions affect prediction accuracy"""
        
        market_analysis = {}
        
        # Analyze by volatility
        if 'market_volatility' in df.columns:
            volatility_analysis = df.groupby('market_volatility').agg({
                'accuracy_score': 'mean',
                'error_percentage': 'mean',
                'confidence_score': 'mean'
            }).round(2)
            market_analysis['volatility_impact'] = volatility_analysis.to_dict()
        
        # Analyze by trend strength
        if 'trend_strength' in df.columns:
            trend_analysis = df.groupby('trend_strength').agg({
                'accuracy_score': 'mean',
                'error_percentage': 'mean',
                'confidence_score': 'mean'
            }).round(2)
            market_analysis['trend_impact'] = trend_analysis.to_dict()
        
        # Analyze by volume trend
        if 'volume_trend' in df.columns:
            volume_analysis = df.groupby('volume_trend').agg({
                'accuracy_score': 'mean',
                'error_percentage': 'mean',
                'confidence_score': 'mean'
            }).round(2)
            market_analysis['volume_impact'] = volume_analysis.to_dict()
        
        return market_analysis
    
    def _generate_improvement_recommendations(self, error_clusters, confidence_analysis, 
                                           time_patterns, symbol_patterns, market_patterns) -> Dict[str, Any]:
        """Generate specific improvement recommendations based on analysis"""
        
        recommendations = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_optimizations': [],
            'model_adjustments': [],
            'feature_engineering': []
        }
        
        # Analyze error clusters
        for cluster_id, cluster_data in error_clusters.items():
            if cluster_data['avg_error'] > 10:
                recommendations['immediate_actions'].append(
                    f"Address high-error cluster {cluster_id}: {', '.join(cluster_data['characteristics'])}"
                )
        
        # Analyze confidence issues
        if confidence_analysis['correlation'] < 0.3:
            recommendations['model_adjustments'].append(
                "Improve confidence scoring mechanism - low correlation with accuracy"
            )
        
        # Analyze timeframe issues
        timeframe_performance = time_patterns['timeframe_performance']
        for timeframe, performance in timeframe_performance.items():
            try:
                if isinstance(performance, dict) and ('accuracy_score', 'mean') in performance:
                    if performance[('accuracy_score', 'mean')] < 60:
                        recommendations['short_term_improvements'].append(
                            f"Optimize model for {timeframe} predictions - low accuracy"
                        )
            except (KeyError, TypeError):
                # Skip if performance data structure is unexpected
                continue
        
        # Analyze symbol issues
        worst_symbols = list(symbol_patterns['worst_performing_symbols'].keys())
        if worst_symbols:
            recommendations['feature_engineering'].append(
                f"Add symbol-specific features for: {', '.join(worst_symbols[:3])}"
            )
        
        # Market condition improvements
        if 'volatility_impact' in market_patterns:
            volatility_data = market_patterns['volatility_impact']
            try:
                if 'high' in volatility_data and isinstance(volatility_data['high'], dict):
                    if ('accuracy_score', 'mean') in volatility_data['high'] and volatility_data['high'][('accuracy_score', 'mean')] < 60:
                        recommendations['model_adjustments'].append(
                            "Improve model performance during high volatility periods"
                        )
            except (KeyError, TypeError):
                # Skip if volatility data structure is unexpected
                pass
        
        return recommendations
    
    def _save_insights(self, insights: Dict[str, Any]):
        """Save learning insights to storage"""
        try:
            with open(self.learning_file, 'w') as f:
                json.dump(insights, f, indent=2)
            logger.info("Learning insights saved successfully")
        except Exception as e:
            logger.error(f"Error saving insights: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of learning insights"""
        try:
            if os.path.exists(self.learning_file):
                with open(self.learning_file, 'r') as f:
                    insights = json.load(f)
                return insights
            else:
                return {"error": "No learning insights available. Run analysis first."}
        except Exception as e:
            return {"error": f"Error loading insights: {e}"}
    
    def predict_accuracy_improvement(self, improvements_implemented: List[str]) -> Dict[str, float]:
        """Predict accuracy improvement based on implemented improvements"""
        
        # Get current baseline
        current_metrics = self.tracker.get_accuracy_metrics()
        baseline_accuracy = current_metrics.overall_accuracy
        
        # Improvement predictions based on common ML improvements
        improvement_predictions = {
            'feature_engineering': 5.0,  # 5% improvement
            'model_retraining': 3.0,     # 3% improvement
            'hyperparameter_optimization': 2.0,  # 2% improvement
            'ensemble_methods': 4.0,     # 4% improvement
            'confidence_calibration': 2.5,  # 2.5% improvement
            'data_quality_improvement': 3.5,  # 3.5% improvement
        }
        
        total_improvement = 0
        for improvement in improvements_implemented:
            if improvement in improvement_predictions:
                total_improvement += improvement_predictions[improvement]
        
        predicted_accuracy = min(95.0, baseline_accuracy + total_improvement)  # Cap at 95%
        
        return {
            'current_accuracy': baseline_accuracy,
            'predicted_accuracy': predicted_accuracy,
            'improvement': total_improvement,
            'confidence_interval': f"±{total_improvement * 0.2:.1f}%"  # 20% uncertainty
        }

# Example usage and testing
if __name__ == "__main__":
    engine = AccuracyLearningEngine()
    
    print("🔍 Analyzing prediction patterns...")
    insights = engine.analyze_error_patterns(days_back=30)
    
    if 'error' not in insights:
        print("\n📊 Learning Insights Summary:")
        print(f"Predictions analyzed: {insights['total_predictions_analyzed']}")
        print(f"Analysis period: {insights['analysis_period']}")
        
        print("\n🎯 Key Recommendations:")
        for category, recommendations in insights['recommendations'].items():
            if recommendations:
                print(f"\n{category.replace('_', ' ').title()}:")
                for rec in recommendations:
                    print(f"  • {rec}")
        
        print("\n📈 Accuracy Improvement Prediction:")
        improvements = ['feature_engineering', 'model_retraining']
        prediction = engine.predict_accuracy_improvement(improvements)
        print(f"Current: {prediction['current_accuracy']:.1f}%")
        print(f"Predicted: {prediction['predicted_accuracy']:.1f}%")
        print(f"Improvement: +{prediction['improvement']:.1f}%")
        print(f"Confidence: {prediction['confidence_interval']}")
    else:
        print(f"❌ Error: {insights['error']}") 