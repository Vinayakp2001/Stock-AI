"""
Prediction Accuracy Analysis Dashboard
Provides detailed analysis of prediction performance and refinement recommendations
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prediction_tracker import PredictionTracker, AccuracyMetrics
import json

def create_accuracy_dashboard():
    """Create the prediction accuracy analysis dashboard"""
    
    # Initialize tracker
    tracker = PredictionTracker()
    
    # Get metrics
    metrics = tracker.get_accuracy_metrics()
    recommendations = tracker.get_refinement_recommendations()
    
    # Check if we have any predictions
    if metrics.total_predictions == 0:
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Prediction Accuracy Analysis", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H4("🚀 Welcome to Prediction Accuracy Analysis!", className="alert-heading"),
                        html.P("This dashboard will help you track and improve your prediction accuracy over time."),
                        html.Hr(),
                        html.H5("To get started:"),
                        html.Ol([
                            html.Li("Go to the main dashboard and select 'Price Prediction'"),
                            html.Li("Make predictions for different stocks and timeframes"),
                            html.Li("Wait for the predictions to complete (1 day, 1 week, 1 month)"),
                            html.Li("Come back here to see your accuracy analysis"),
                            html.Li("Follow the improvement recommendations to enhance accuracy")
                        ]),
                        html.Hr(),
                        html.P([
                            "💡 Tip: The more predictions you make, the better the system learns and improves!",
                            html.Br(),
                            "📈 Expected accuracy improvement: 5-15% over the first month"
                        ], className="mb-0")
                    ], color="info", className="text-center")
                ])
            ])
        ], fluid=True)
    
    # Create accuracy overview cards
    accuracy_cards = [
        dbc.Card([
            dbc.CardHeader("Overall Accuracy", className="bg-primary text-white"),
            dbc.CardBody([
                html.H3(f"{metrics.overall_accuracy:.1f}%", className="text-primary"),
                html.P(f"{metrics.accurate_predictions}/{metrics.total_predictions} predictions correct")
            ])
        ], className="mb-3"),
        
        dbc.Card([
            dbc.CardHeader("Mean Error", className="bg-info text-white"),
            dbc.CardBody([
                html.H3(f"{metrics.mean_absolute_percentage_error:.2f}%", className="text-info"),
                html.P(f"Average prediction error")
            ])
        ], className="mb-3"),
        
        dbc.Card([
            dbc.CardHeader("Confidence Correlation", className="bg-success text-white"),
            dbc.CardBody([
                html.H3(f"{metrics.confidence_correlation:.3f}", className="text-success"),
                html.P("Confidence vs Accuracy correlation")
            ])
        ], className="mb-3"),
        
        dbc.Card([
            dbc.CardHeader("Sharpe Ratio", className="bg-warning text-dark"),
            dbc.CardBody([
                html.H3(f"{metrics.root_mean_squared_error:.2f}", className="text-warning"),
                html.P("Root Mean Square Error")
            ])
        ], className="mb-3")
    ]
    
    # Create accuracy by confidence chart
    if metrics.accuracy_by_confidence:
        confidence_data = pd.DataFrame([
            {'Confidence Level': k, 'Accuracy (%)': v} 
            for k, v in metrics.accuracy_by_confidence.items()
        ])
        
        confidence_fig = px.bar(
            confidence_data, 
            x='Confidence Level', 
            y='Accuracy (%)',
            title='Accuracy by Confidence Level',
            color='Accuracy (%)',
            color_continuous_scale='RdYlGn'
        )
        confidence_fig.update_layout(height=400)
    else:
        # Create empty chart with message
        confidence_fig = go.Figure()
        confidence_fig.add_annotation(
            text="No confidence data available yet.<br>Make some predictions first!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        confidence_fig.update_layout(
            title='Accuracy by Confidence Level',
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
    
    # Create accuracy by timeframe chart
    if metrics.accuracy_by_timeframe:
        timeframe_data = pd.DataFrame([
            {'Timeframe': k, 'Accuracy (%)': v} 
            for k, v in metrics.accuracy_by_timeframe.items()
        ])
        
        timeframe_fig = px.pie(
            timeframe_data, 
            values='Accuracy (%)', 
            names='Timeframe',
            title='Accuracy by Timeframe'
        )
        timeframe_fig.update_layout(height=400)
    else:
        # Create empty chart with message
        timeframe_fig = go.Figure()
        timeframe_fig.add_annotation(
            text="No timeframe data available yet.<br>Make some predictions first!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        timeframe_fig.update_layout(
            title='Accuracy by Timeframe',
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
    
    # Create accuracy by symbol chart
    if metrics.accuracy_by_symbol:
        symbol_data = pd.DataFrame([
            {'Symbol': k, 'Accuracy (%)': v} 
            for k, v in metrics.accuracy_by_symbol.items()
        ]).sort_values('Accuracy (%)', ascending=True)
        
        symbol_fig = px.bar(
            symbol_data, 
            x='Accuracy (%)', 
            y='Symbol',
            orientation='h',
            title='Accuracy by Symbol',
            color='Accuracy (%)',
            color_continuous_scale='RdYlGn'
        )
        symbol_fig.update_layout(height=400)
    else:
        # Create empty chart with message
        symbol_fig = go.Figure()
        symbol_fig.add_annotation(
            text="No symbol data available yet.<br>Make some predictions first!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        symbol_fig.update_layout(
            title='Accuracy by Symbol',
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
    
    # Create recommendations section
    recommendations_cards = []
    
    # Overall assessment
    recommendations_cards.append(
        dbc.Card([
            dbc.CardHeader("Overall Assessment", className="bg-dark text-white"),
            dbc.CardBody([
                html.P(recommendations['overall_assessment'], className="lead")
            ])
        ], className="mb-3")
    )
    
    # Specific issues
    if recommendations['specific_issues']:
        issues_list = [html.Li(issue) for issue in recommendations['specific_issues']]
        recommendations_cards.append(
            dbc.Card([
                dbc.CardHeader("Identified Issues", className="bg-danger text-white"),
                dbc.CardBody([
                    html.Ul(issues_list)
                ])
            ], className="mb-3")
        )
    
    # Improvement suggestions
    if recommendations['improvement_suggestions']:
        suggestions_list = [html.Li(suggestion) for suggestion in recommendations['improvement_suggestions']]
        recommendations_cards.append(
            dbc.Card([
                dbc.CardHeader("Improvement Suggestions", className="bg-info text-white"),
                dbc.CardBody([
                    html.Ul(suggestions_list)
                ])
            ], className="mb-3")
        )
    
    # Priority actions
    if recommendations['priority_actions']:
        actions_list = [html.Li(action) for action in recommendations['priority_actions']]
        recommendations_cards.append(
            dbc.Card([
                dbc.CardHeader("Priority Actions", className="bg-warning text-dark"),
                dbc.CardBody([
                    html.Ul(actions_list)
                ])
            ], className="mb-3")
        )
    
    # Create the main layout
    layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("📊 Prediction Accuracy Analysis", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        # Accuracy Overview Cards
        dbc.Row([
            dbc.Col(card, width=3) for card in accuracy_cards
        ], className="mb-4"),
        
        # Charts Row 1
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=confidence_fig)
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=timeframe_fig)
            ], width=6)
        ], className="mb-4"),
        
        # Charts Row 2
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=symbol_fig)
            ], width=12)
        ], className="mb-4"),
        
        # Recommendations Section
        dbc.Row([
            dbc.Col([
                html.H3("🔧 Refinement Recommendations", className="mb-3"),
                html.Hr()
            ])
        ]),
        
        dbc.Row([
            dbc.Col(card, width=12) for card in recommendations_cards
        ]),
        
        # Detailed Analysis Section
        dbc.Row([
            dbc.Col([
                html.H3("📈 Detailed Performance Analysis", className="mb-3"),
                html.Hr(),
                dbc.Tabs([
                    dbc.Tab([
                        html.Div(id="detailed-metrics-content")
                    ], label="Detailed Metrics"),
                    dbc.Tab([
                        html.Div(id="factor-analysis-content")
                    ], label="Factor Analysis"),
                    dbc.Tab([
                        html.Div(id="improvement-pipeline-content")
                    ], label="Improvement Pipeline")
                ])
            ])
        ], className="mt-4")
        
    ], fluid=True)
    
    return layout

def create_detailed_metrics_content(tracker: PredictionTracker):
    """Create detailed metrics content"""
    
    # Get recent predictions for detailed analysis
    recent_predictions = [p for p in tracker.predictions 
                         if p.status.value == 'completed' 
                         and p.prediction_date >= datetime.now() - timedelta(days=30)]
    
    if not recent_predictions:
        return dbc.Alert("No completed predictions found for detailed analysis.", color="warning")
    
    # Create detailed metrics table
    table_data = []
    for pred in recent_predictions[-20:]:  # Show last 20 predictions
        table_data.append({
            'Date': pred.prediction_date.strftime('%Y-%m-%d'),
            'Symbol': pred.symbol,
            'Timeframe': pred.timeframe,
            'Predicted': f"${pred.predicted_price:.2f}",
            'Actual': f"${pred.actual_price:.2f}" if pred.actual_price else "N/A",
            'Error %': f"{pred.error_percentage:.2f}%" if pred.error_percentage else "N/A",
            'Confidence': f"{pred.confidence_score:.1f}%",
            'Accuracy': f"{pred.accuracy_score:.1f}" if pred.accuracy_score else "N/A"
        })
    
    df = pd.DataFrame(table_data)
    
    return dbc.Table.from_dataframe(
        df, 
        striped=True, 
        bordered=True, 
        hover=True,
        className="table-sm"
    )

def create_factor_analysis_content(tracker: PredictionTracker):
    """Create factor analysis content"""
    
    # Analyze factors affecting prediction accuracy
    completed_predictions = [p for p in tracker.predictions if p.status.value == 'completed']
    
    if not completed_predictions:
        return dbc.Alert("No completed predictions found for factor analysis.", color="warning")
    
    # Analyze confidence vs accuracy
    high_confidence_preds = [p for p in completed_predictions if p.confidence_score > 70]
    low_confidence_preds = [p for p in completed_predictions if p.confidence_score <= 70]
    
    high_conf_accuracy = np.mean([p.accuracy_score for p in high_confidence_preds]) if high_confidence_preds else 0
    low_conf_accuracy = np.mean([p.accuracy_score for p in low_confidence_preds]) if low_confidence_preds else 0
    
    # Analyze by timeframe
    timeframe_analysis = {}
    for pred in completed_predictions:
        if pred.timeframe not in timeframe_analysis:
            timeframe_analysis[pred.timeframe] = []
        timeframe_analysis[pred.timeframe].append(pred.accuracy_score)
    
    timeframe_avg = {tf: np.mean(scores) for tf, scores in timeframe_analysis.items()}
    
    # Create factor analysis cards
    factor_cards = [
        dbc.Card([
            dbc.CardHeader("Confidence Impact", className="bg-info text-white"),
            dbc.CardBody([
                html.H5("High Confidence (>70%)"),
                html.P(f"Average Accuracy: {high_conf_accuracy:.1f}%"),
                html.H5("Low Confidence (≤70%)"),
                html.P(f"Average Accuracy: {low_conf_accuracy:.1f}%"),
                html.Hr(),
                html.P("Analysis: Higher confidence predictions tend to be more accurate" if high_conf_accuracy > low_conf_accuracy else "Analysis: Confidence doesn't strongly correlate with accuracy")
            ])
        ], className="mb-3"),
        
        dbc.Card([
            dbc.CardHeader("Timeframe Performance", className="bg-success text-white"),
            dbc.CardBody([
                html.Ul([
                    html.Li(f"{tf}: {acc:.1f}% accuracy") 
                    for tf, acc in timeframe_avg.items()
                ])
            ])
        ], className="mb-3")
    ]
    
    return dbc.Row([
        dbc.Col(card, width=6) for card in factor_cards
    ])

def create_improvement_pipeline_content():
    """Create improvement pipeline content"""
    
    improvement_steps = [
        {
            'step': 1,
            'title': 'Data Quality Assessment',
            'description': 'Review and improve data quality, remove outliers, ensure consistency',
            'status': 'pending',
            'priority': 'high'
        },
        {
            'step': 2,
            'title': 'Feature Engineering',
            'description': 'Add new relevant features, optimize existing features',
            'status': 'pending',
            'priority': 'high'
        },
        {
            'step': 3,
            'title': 'Model Retraining',
            'description': 'Retrain models with updated data and features',
            'status': 'pending',
            'priority': 'medium'
        },
        {
            'step': 4,
            'title': 'Hyperparameter Optimization',
            'description': 'Optimize model parameters for better performance',
            'status': 'pending',
            'priority': 'medium'
        },
        {
            'step': 5,
            'title': 'Ensemble Methods',
            'description': 'Combine multiple models for improved accuracy',
            'status': 'pending',
            'priority': 'low'
        },
        {
            'step': 6,
            'title': 'Real-time Monitoring',
            'description': 'Implement continuous monitoring and alerting',
            'status': 'pending',
            'priority': 'low'
        }
    ]
    
    pipeline_cards = []
    for step in improvement_steps:
        status_color = {
            'pending': 'secondary',
            'in_progress': 'warning',
            'completed': 'success'
        }.get(step['status'], 'secondary')
        
        priority_color = {
            'high': 'danger',
            'medium': 'warning',
            'low': 'info'
        }.get(step['priority'], 'secondary')
        
        card = dbc.Card([
            dbc.CardHeader([
                html.Span(f"Step {step['step']}: {step['title']}", className="h6"),
                dbc.Badge(step['priority'].title(), color=priority_color, className="float-right")
            ]),
            dbc.CardBody([
                html.P(step['description']),
                dbc.Progress(
                    value=0 if step['status'] == 'pending' else 50 if step['status'] == 'in_progress' else 100,
                    color=status_color,
                    className="mb-2"
                ),
                html.Small(f"Status: {step['status'].replace('_', ' ').title()}")
            ])
        ], className="mb-3")
        
        pipeline_cards.append(card)
    
    return dbc.Row([
        dbc.Col([
            html.H5("Improvement Pipeline Steps", className="mb-3"),
            html.P("Follow these steps to improve prediction accuracy:"),
            html.Hr()
        ])
    ]) + dbc.Row([
        dbc.Col(card, width=6) for card in pipeline_cards
    ])

# Callback functions for the dashboard
def register_accuracy_callbacks(app):
    """Register callbacks for the accuracy dashboard"""
    
    @app.callback(
        Output("detailed-metrics-content", "children"),
        Input("detailed-metrics-tab", "n_clicks")
    )
    def update_detailed_metrics(n_clicks):
        if n_clicks:
            tracker = PredictionTracker()
            return create_detailed_metrics_content(tracker)
        return ""
    
    @app.callback(
        Output("factor-analysis-content", "children"),
        Input("factor-analysis-tab", "n_clicks")
    )
    def update_factor_analysis(n_clicks):
        if n_clicks:
            tracker = PredictionTracker()
            return create_factor_analysis_content(tracker)
        return ""
    
    @app.callback(
        Output("improvement-pipeline-content", "children"),
        Input("improvement-pipeline-tab", "n_clicks")
    )
    def update_improvement_pipeline(n_clicks):
        if n_clicks:
            return create_improvement_pipeline_content()
        return "" 