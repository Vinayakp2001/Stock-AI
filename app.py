"""
Web Dashboard
Interactive dashboard for stock prediction and backtesting results
"""

import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_agent import DataAgent
from agents.prediction_agent import PredictionAgent
from backtesting.engine import BacktestEngine, MovingAverageStrategy, RSIStrategy, MACDStrategy

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Stock Prediction Agent SDK"

# Initialize agents
data_agent = DataAgent()
prediction_agent = PredictionAgent()
backtest_engine = BacktestEngine(initial_capital=100000)

# Popular stocks for quick selection
POPULAR_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']

# Indian stocks
INDIAN_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'AXISBANK.NS',
    'KOTAKBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS'
]

# Market indices
MARKET_INDICES = ['^NSEI', '^BSESN', '^GSPC', '^IXIC', '^DJI']

# User's custom stock list (will be stored in session)
user_stocks = []

# All available stocks
ALL_STOCKS = POPULAR_STOCKS + INDIAN_STOCKS + MARKET_INDICES

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📈 Stock Prediction Agent SDK", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Stock management section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Add Custom Stock", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Stock Symbol:"),
                            dcc.Input(
                                id='stock-input',
                                type='text',
                                placeholder='Enter stock symbol (e.g., RELIANCE.NS)',
                                className='form-control'
                            )
                        ], width=8),
                        dbc.Col([
                            html.Label("&nbsp;"),  # Spacer
                            dbc.Button("Add Stock", id="add-stock-btn", color="success", className="w-100")
                        ], width=4)
                    ], className="mb-3"),
                    html.Div(id="add-stock-message"),
                    html.Hr(),
                    html.H6("Stock Suggestions:"),
                    html.Div([
                        dbc.Badge(stock, color="primary", className="me-2 mb-2", 
                                 style={"cursor": "pointer"}) 
                        for stock in INDIAN_STOCKS[:10]
                    ], id="stock-suggestions")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Stock selection and controls
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
                                value='AAPL',
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
                                    {'label': '2 Years', 'value': '2y'},
                                    {'label': '5 Years', 'value': '5y'}
                                ],
                                value='1y'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Analysis Type:"),
                            dcc.Dropdown(
                                id='analysis-dropdown',
                                options=[
                                    {'label': 'Price Prediction', 'value': 'prediction'},
                                    {'label': 'Strategy Backtesting', 'value': 'backtest'},
                                    {'label': 'Technical Analysis', 'value': 'technical'}
                                ],
                                value='prediction'
                            )
                        ], width=4)
                    ], className="mb-3"),
                    dbc.Button("Analyze", id="analyze-btn", color="primary", className="w-100")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Loading spinner
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-1",
                type="default",
                children=html.Div(id="loading-output")
            )
        ])
    ]),
    
    # Results area
    dbc.Row([
        dbc.Col([
            html.Div(id="results-container")
        ])
    ])
], fluid=True)

# Callback to add custom stock
@app.callback(
    Output("add-stock-message", "children"),
    Output("stock-dropdown", "options"),
    Input("add-stock-btn", "n_clicks"),
    State("stock-input", "value"),
    prevent_initial_call=True
)
def add_custom_stock(n_clicks, stock_symbol):
    if not stock_symbol:
        return dbc.Alert("Please enter a stock symbol", color="warning"), dash.no_update
    
    stock_symbol = stock_symbol.upper().strip()
    
    try:
        # Test if stock exists by fetching a small amount of data
        test_data = data_agent.get_stock_data(stock_symbol, period="1mo")
        
        if test_data and not test_data.data.empty:
            # Add to global list
            if stock_symbol not in ALL_STOCKS:
                ALL_STOCKS.append(stock_symbol)
            
            # Update dropdown options
            options = [{'label': symbol, 'value': symbol} for symbol in ALL_STOCKS]
            
            return dbc.Alert(f"✅ Successfully added {stock_symbol}!", color="success"), options
        else:
            return dbc.Alert(f"❌ Could not fetch data for {stock_symbol}. Please check the symbol.", color="danger"), dash.no_update
            
    except Exception as e:
        return dbc.Alert(f"❌ Error adding {stock_symbol}: {str(e)}", color="danger"), dash.no_update

# Callback to handle stock suggestion clicks
@app.callback(
    Output("stock-input", "value"),
    Input("stock-suggestions", "children"),
    prevent_initial_call=True
)
def handle_stock_suggestion_click(children):
    # This will be triggered when a suggestion badge is clicked
    ctx = dash.callback_context
    if ctx.triggered:
        # Extract stock symbol from clicked badge
        # For now, we'll just return the first suggestion
        return INDIAN_STOCKS[0] if INDIAN_STOCKS else ""
    return ""

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
    """Update results based on user selection"""
    if not symbol or not period or not analysis_type:
        return [], ""
    
    try:
        if analysis_type == "prediction":
            return create_prediction_dashboard(symbol, period), ""
        elif analysis_type == "backtest":
            return create_backtest_dashboard(symbol, period), ""
        elif analysis_type == "technical":
            return create_technical_dashboard(symbol, period), ""
        else:
            return [], ""
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger"), ""

def create_prediction_dashboard(symbol, period):
    """Create prediction dashboard"""
    # Get stock data
    stock_data = data_agent.get_stock_data(symbol, period=period)
    
    # Prepare data for ML
    X, y = data_agent.prepare_ml_data(stock_data, target_days=1)
    
    # Train models
    prediction_agent.train_models(X, y, symbol)
    
    # Make prediction
    latest_features = X.tail(1)
    prediction = prediction_agent.predict(latest_features, symbol)
    
    # Create price chart
    fig_price = go.Figure()
    
    # Add historical prices
    fig_price.add_trace(go.Scatter(
        x=stock_data.data.index,
        y=stock_data.data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue')
    ))
    
    # Add predicted price
    current_price = stock_data.data['Close'].iloc[-1]
    fig_price.add_trace(go.Scatter(
        x=[stock_data.data.index[-1]],
        y=[prediction.predicted_price],
        mode='markers',
        name='Predicted Price',
        marker=dict(color='red', size=10)
    ))
    
    fig_price.update_layout(
        title=f"{symbol} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    
    # Create prediction metrics
    expected_return = (prediction.predicted_price - current_price) / current_price
    
    metrics_card = dbc.Card([
        dbc.CardBody([
            html.H5("Prediction Results", className="card-title"),
            dbc.Row([
                dbc.Col([
                    html.H6("Current Price"),
                    html.H4(f"${current_price:.2f}", className="text-primary")
                ], width=3),
                dbc.Col([
                    html.H6("Predicted Price"),
                    html.H4(f"${prediction.predicted_price:.2f}", className="text-success")
                ], width=3),
                dbc.Col([
                    html.H6("Expected Return"),
                    html.H4(f"{expected_return:.2%}", 
                           className="text-success" if expected_return > 0 else "text-danger")
                ], width=3),
                dbc.Col([
                    html.H6("Confidence"),
                    html.H4(f"{prediction.confidence:.1%}", className="text-info")
                ], width=3)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H6("Signal"),
                    html.H4(prediction.signal, 
                           className="text-success" if prediction.signal == "BUY" else 
                                   "text-danger" if prediction.signal == "SELL" else "text-warning")
                ], width=3),
                dbc.Col([
                    html.H6("Model Used"),
                    html.H4(prediction.model_name, className="text-secondary")
                ], width=3)
            ], className="mt-3")
        ])
    ])
    
    return [
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_price)
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                metrics_card
            ], width=12)
        ])
    ]

def create_backtest_dashboard(symbol, period):
    """Create backtesting dashboard"""
    # Get stock data
    stock_data = data_agent.get_stock_data(symbol, period=period)
    data = stock_data.data
    
    # Create strategies
    strategies = [
        MovingAverageStrategy(20, 50),
        RSIStrategy(14, 30, 70),
        MACDStrategy(12, 26, 9)
    ]
    
    # Run backtests
    results = backtest_engine.compare_strategies(strategies, data, symbol)
    
    # Create performance comparison chart
    fig_performance = go.Figure()
    
    for strategy_name, result in results.items():
        fig_performance.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode='lines',
            name=strategy_name
        ))
    
    fig_performance.update_layout(
        title=f"{symbol} Strategy Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400
    )
    
    # Create performance metrics table
    metrics_data = []
    for strategy_name, result in results.items():
        metrics_data.append({
            'Strategy': strategy_name,
            'Total Return': f"{result.total_return_pct:.2%}",
            'Win Rate': f"{result.win_rate:.2%}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Max Drawdown': f"{result.max_drawdown_pct:.2%}",
            'Total Trades': result.total_trades
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    return [
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_performance)
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Strategy Performance", className="card-title"),
                        dbc.Table.from_dataframe(
                            metrics_df, 
                            striped=True, 
                            bordered=True, 
                            hover=True
                        )
                    ])
                ])
            ], width=12)
        ])
    ]

def create_technical_dashboard(symbol, period):
    """Create technical analysis dashboard"""
    # Get stock data
    stock_data = data_agent.get_stock_data(symbol, period=period)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(
        x=stock_data.data.index,
        y=stock_data.data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ), row=1, col=1)
    
    if 'sma_20' in stock_data.indicators:
        fig.add_trace(go.Scatter(
            x=stock_data.data.index,
            y=stock_data.indicators['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange')
        ), row=1, col=1)
    
    if 'sma_50' in stock_data.indicators:
        fig.add_trace(go.Scatter(
            x=stock_data.data.index,
            y=stock_data.indicators['sma_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='red')
        ), row=1, col=1)
    
    # RSI
    if 'rsi' in stock_data.indicators:
        fig.add_trace(go.Scatter(
            x=stock_data.data.index,
            y=stock_data.indicators['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'macd' in stock_data.indicators and 'macd_signal' in stock_data.indicators:
        fig.add_trace(go.Scatter(
            x=stock_data.data.index,
            y=stock_data.indicators['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=stock_data.data.index,
            y=stock_data.indicators['macd_signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red')
        ), row=3, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    
    # Create technical indicators summary
    latest_indicators = {}
    for name, values in stock_data.indicators.items():
        if not pd.isna(values.iloc[-1]):
            latest_indicators[name] = values.iloc[-1]
    
    indicators_card = dbc.Card([
        dbc.CardBody([
            html.H5("Technical Indicators", className="card-title"),
            dbc.Row([
                dbc.Col([
                    html.H6("RSI"),
                    html.H4(f"{latest_indicators.get('rsi', 0):.1f}", 
                           className="text-danger" if latest_indicators.get('rsi', 0) > 70 else
                                   "text-success" if latest_indicators.get('rsi', 0) < 30 else "text-secondary")
                ], width=3),
                dbc.Col([
                    html.H6("MACD"),
                    html.H4(f"{latest_indicators.get('macd', 0):.4f}", className="text-secondary")
                ], width=3),
                dbc.Col([
                    html.H6("SMA 20"),
                    html.H4(f"${latest_indicators.get('sma_20', 0):.2f}", className="text-secondary")
                ], width=3),
                dbc.Col([
                    html.H6("SMA 50"),
                    html.H4(f"${latest_indicators.get('sma_50', 0):.2f}", className="text-secondary")
                ], width=3)
            ])
        ])
    ])
    
    return [
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig)
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                indicators_card
            ], width=12)
        ])
    ]

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=8050)