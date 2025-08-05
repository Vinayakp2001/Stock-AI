#!/usr/bin/env python3
"""
Fresh Stock Prediction Dashboard - Bypasses all caching issues
"""
import os
import sys
import importlib
import traceback
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Dash components
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import prediction tracking components
from prediction_tracker import PredictionTracker
from prediction_accuracy_dashboard import create_accuracy_dashboard, register_accuracy_callbacks

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Stock Prediction Agent SDK"

# Stock lists
POPULAR_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
INDIAN_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'AXISBANK.NS',
    'KOTAKBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS'
]
MARKET_INDICES = ['^NSEI', '^BSESN', '^GSPC', '^IXIC', '^DJI']
ALL_STOCKS = POPULAR_STOCKS + INDIAN_STOCKS + MARKET_INDICES

def get_fresh_agents():
    """Get completely fresh instances of all agents - bypasses all caching"""
    try:
        # Force reload all modules
        modules_to_reload = [
            'agents.data_agent',
            'agents.prediction_agent', 
            'backtesting.engine'
        ]
        
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # Fresh imports
        from agents.data_agent import DataAgent
        from agents.prediction_agent import PredictionAgent
        from backtesting.engine import BacktestEngine, MovingAverageStrategy
        
        # Create fresh instances
        data_agent = DataAgent()
        prediction_agent = PredictionAgent()
        backtest_engine = BacktestEngine(initial_capital=100000)
        
        return data_agent, prediction_agent, backtest_engine
        
    except Exception as e:
        print(f"Error getting fresh agents: {e}")
        traceback.print_exc()
        return None, None, None

def get_currency_symbol(symbol):
    """Get currency symbol based on stock symbol"""
    if symbol.endswith(('.NS', '.BO')):
        return '₹'
    return '$'

# App Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📈 Stock Prediction Agent SDK", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Stock Selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Stock Analysis", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Stock Symbol:"),
                            dcc.Dropdown(
                                id='stock-dropdown',
                                options=[{'label': symbol, 'value': symbol} for symbol in ALL_STOCKS],
                                value='RELIANCE.NS',
                                placeholder="Select a stock..."
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Time Period:"),
                            dcc.Dropdown(
                                id='period-dropdown',
                                options=[
                                    {'label': '1 Month', 'value': '1mo'},
                                    {'label': '3 Months', 'value': '3mo'},
                                    {'label': '6 Months', 'value': '6mo'},
                                    {'label': '1 Year', 'value': '1y'},
                                    {'label': '2 Years', 'value': '2y'}
                                ],
                                value='6mo'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Analysis Type:"),
                            dcc.Dropdown(
                                id='analysis-dropdown',
                                options=[
                                    {'label': 'Price Prediction', 'value': 'prediction'},
                                    {'label': 'Technical Analysis', 'value': 'technical'},
                                    {'label': 'Backtesting', 'value': 'backtest'},
                                    {'label': 'Prediction Accuracy', 'value': 'accuracy'}
                                ],
                                value='prediction'
                            )
                        ], width=4)
                    ], className="mb-3"),
                    dbc.Button("Analyze Stock", id="analyze-btn", color="primary", className="w-100")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Results Container
    dbc.Row([
        dbc.Col([
            html.Div(id="results-container")
        ])
    ]),
    
    # Loading
    dbc.Row([
        dbc.Col([
            html.Div(id="loading-output")
        ])
    ])
], fluid=True)

@app.callback(
    Output("results-container", "children"),
    Output("loading-output", "children"),
    Input("analyze-btn", "n_clicks"),
    Input("stock-dropdown", "value"),
    Input("period-dropdown", "value"),
    Input("analysis-dropdown", "value"),
    prevent_initial_call=True
)
def update_results(n_clicks, symbol, period, analysis_type):
    """Main callback for analysis"""
    if not symbol:
        return dbc.Alert("Please select a stock symbol.", color="warning"), ""
    
    try:
        # Get fresh agents
        data_agent, prediction_agent, backtest_engine = get_fresh_agents()
        
        if not all([data_agent, prediction_agent, backtest_engine]):
            return dbc.Alert("❌ Failed to initialize agents. Please restart the application.", color="danger"), ""
        
        # Verify the method exists
        if not hasattr(prediction_agent, 'predict_multiple_timeframes'):
            return dbc.Alert(
                f"❌ Method 'predict_multiple_timeframes' not found in PredictionAgent. "
                f"Available methods: {[m for m in dir(prediction_agent) if not m.startswith('_')]}", 
                color="danger"
            ), ""
        
        # Get stock data
        stock_data_obj = data_agent.get_stock_data(symbol, period=period)
        
        if stock_data_obj.data.empty:
            return dbc.Alert(f"❌ No data found for {symbol}. Please check the symbol.", color="warning"), ""
        
        # Prepare data for ML
        X, y = data_agent.prepare_ml_data(stock_data_obj, target_days=1)
        
        if X.empty or y.empty or len(X) < 10:
            return dbc.Alert("❌ Not enough data for prediction. Try a longer period.", color="warning"), ""
        
        # Perform analysis based on type
        if analysis_type == 'prediction':
            return create_prediction_dashboard(symbol, period, data_agent, prediction_agent), ""
        elif analysis_type == 'technical':
            return create_technical_dashboard(symbol, period, data_agent), ""
        elif analysis_type == 'backtest':
            return create_backtest_dashboard(symbol, period, data_agent, backtest_engine), ""
        elif analysis_type == 'accuracy':
            return create_accuracy_dashboard(), ""
        else:
            return dbc.Alert("❌ Invalid analysis type.", color="danger"), ""
            
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return dbc.Alert(error_msg, color="danger"), ""

def create_prediction_dashboard(symbol, period, data_agent, prediction_agent):
    """Create enhanced prediction dashboard"""
    try:
        # Get stock data
        stock_data_obj = data_agent.get_stock_data(symbol, period=period)
        X, y = data_agent.prepare_ml_data(stock_data_obj, target_days=1)
        
        # Get multiple timeframe predictions
        multi_prediction = prediction_agent.predict_multiple_timeframes(X, symbol)
        
        # Track predictions for accuracy analysis
        tracker = PredictionTracker()
        for timeframe, prediction in multi_prediction.predictions.items():
            tracker.add_prediction(
                symbol=symbol,
                timeframe=timeframe,
                predicted_price=prediction.predicted_price,
                confidence_score=prediction.confidence,
                prediction_features={
                    'technical_signals': prediction.signal,
                    'risk_adjusted_return': prediction.risk_adjusted_return
                },
                market_conditions={
                    'volatility': 'medium',  # This could be calculated from data
                    'trend_strength': 'medium',
                    'volume_trend': 'stable'
                }
            )
        
        # Get currency symbol
        currency = get_currency_symbol(symbol)
        
        # Create prediction cards
        prediction_cards = []
        
        # Get current price
        current_price = stock_data_obj.data['Close'].iloc[-1]
        
        # Add current price card
        current_card = dbc.Card([
            dbc.CardHeader("Current Price", className="bg-info text-white"),
            dbc.CardBody([
                html.H4(f"{currency}{current_price:.2f}", className="text-info"),
                html.P(f"Last Updated: {stock_data_obj.data.index[-1].strftime('%Y-%m-%d')}"),
                html.P(f"Volume: {stock_data_obj.data['Volume'].iloc[-1]:,.0f}")
            ])
        ], className="mb-3")
        prediction_cards.append(current_card)
        
        timeframe_labels = {'1d': '1 Day', '1w': '1 Week', '1m': '1 Month'}
        for timeframe, prediction in multi_prediction.predictions.items():
            timeframe_label = timeframe_labels.get(timeframe, timeframe)
            
            # Calculate price change
            price_change = prediction.predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Color coding based on prediction
            if price_change > 0:
                price_color = "text-success"
                change_color = "text-success"
                change_icon = "↗️"
            elif price_change < 0:
                price_color = "text-danger"
                change_color = "text-danger"
                change_icon = "↘️"
            else:
                price_color = "text-primary"
                change_color = "text-muted"
                change_icon = "→"
            
            card = dbc.Card([
                dbc.CardHeader(f"{timeframe_label} Prediction"),
                dbc.CardBody([
                    html.H4(f"{currency}{prediction.predicted_price:.2f}", className=price_color),
                    html.P(f"{change_icon} {currency}{price_change:+.2f} ({price_change_pct:+.2f}%)", className=change_color),
                    html.P(f"Confidence: {prediction.confidence:.1f}%"),
                    html.P(f"Signal: {prediction.signal}"),
                    html.P(f"Risk-Adjusted Return: {prediction.risk_adjusted_return:.2f}" if prediction.risk_adjusted_return else "Risk-Adjusted Return: N/A")
                ])
            ], className="mb-3")
            prediction_cards.append(card)
        
        # Create price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_data_obj.data.index,
            y=stock_data_obj.data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        ))
        
        # Add prediction points
        colors = ['red', 'orange', 'green']
        timeframe_days = {'1d': 1, '1w': 7, '1m': 30}
        for i, (timeframe, prediction) in enumerate(multi_prediction.predictions.items()):
            days = timeframe_days.get(timeframe, 1)
            future_date = datetime.now() + timedelta(days=days)
            timeframe_label = timeframe_labels.get(timeframe, timeframe)
            fig.add_trace(go.Scatter(
                x=[future_date],
                y=[prediction.predicted_price],
                mode='markers',
                name=f'{timeframe_label} Prediction',
                marker=dict(color=colors[i], size=10)
            ))
        
        fig.update_layout(
            title=f'{symbol} Price Prediction',
            xaxis_title='Date',
            yaxis_title=f'Price ({currency})',
            height=500
        )
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3(f"📊 {symbol} Analysis Results", className="mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Prediction Cards
            dbc.Row([
                dbc.Col(card, width=4) for card in prediction_cards
            ], className="mb-4"),
            
            # Chart
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig)
                ])
            ]),
            
            # Risk Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics"),
                        dbc.CardBody([
                            html.P(f"Overall Signal: {multi_prediction.overall_signal}"),
                            html.P(f"Overall Confidence: {multi_prediction.overall_confidence:.2f}"),
                            html.P(f"Avg Risk-Adjusted Return: {multi_prediction.risk_metrics.get('avg_risk_adjusted_return', 0):.2f}")
                        ])
                    ])
                ])
            ])
        ])
        
    except Exception as e:
        return dbc.Alert(f"❌ Error in prediction dashboard: {str(e)}", color="danger")

def create_technical_dashboard(symbol, period, data_agent):
    """Create technical analysis dashboard"""
    try:
        stock_data_obj = data_agent.get_stock_data(symbol, period=period)
        
        # Calculate indicators
        indicators = data_agent._calculate_indicators(stock_data_obj.data)
        
        # Create charts
        fig = go.Figure()
        
        # Price and volume
        fig.add_trace(go.Scatter(
            x=stock_data_obj.data.index,
            y=stock_data_obj.data['Close'],
            mode='lines',
            name='Close Price',
            yaxis='y'
        ))
        
        fig.add_trace(go.Bar(
            x=stock_data_obj.data.index,
            y=stock_data_obj.data['Volume'],
            name='Volume',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Volume', overlaying='y', side='right'),
            height=500
        )
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3(f"📈 {symbol} Technical Analysis", className="mb-4")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig)
                ])
            ])
        ])
        
    except Exception as e:
        return dbc.Alert(f"❌ Error in technical dashboard: {str(e)}", color="danger")

def create_backtest_dashboard(symbol, period, data_agent, backtest_engine):
    """Create backtesting dashboard"""
    try:
        stock_data_obj = data_agent.get_stock_data(symbol, period=period)
        
        # Run backtest
        strategy = MovingAverageStrategy()
        results = backtest_engine.run_backtest(strategy, stock_data_obj.data, symbol)
        
        # Create results display
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3(f"🔄 {symbol} Backtesting Results", className="mb-4")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5(f"Total Return: {results.total_return_pct:.2f}%"),
                            html.P(f"Sharpe Ratio: {results.sharpe_ratio:.2f}"),
                            html.P(f"Max Drawdown: {results.max_drawdown_pct:.2f}%")
                        ])
                    ])
                ])
            ])
        ])
        
    except Exception as e:
        return dbc.Alert(f"❌ Error in backtest dashboard: {str(e)}", color="danger")

if __name__ == '__main__':
    print("🚀 Starting Fresh Stock Prediction Dashboard...")
    print("📍 This version bypasses all Python caching issues!")
    print("🌐 Dashboard will be available at: http://localhost:8050")
    
    # Register accuracy dashboard callbacks
    register_accuracy_callbacks(app)
    
    app.run(debug=True, host='0.0.0.0', port=8050) 