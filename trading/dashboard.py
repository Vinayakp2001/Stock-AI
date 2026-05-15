#!/usr/bin/env python3
"""
Trading Bot Dashboard (Issue #44)
Surfaces: Market Regime, Position Manager, Portfolio Optimizer,
          Continuous Learner status, and Decision Engine scores.
Run:   python trading/dashboard.py
Access: http://localhost:8052
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                suppress_callback_exceptions=True)
app.title = "Trading Bot Dashboard — Stock AI"

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    'bg': '#0d1117', 'card': '#161b22', 'border': '#30363d',
    'green': '#3fb950', 'red': '#f85149', 'yellow': '#d29922',
    'blue': '#58a6ff', 'purple': '#bc8cff', 'text': '#e6edf3',
    'muted': '#8b949e', 'header': '#1c2128',
}

CARD = {'backgroundColor': C['card'], 'border': f"1px solid {C['border']}",
        'borderRadius': '8px', 'padding': '16px', 'marginBottom': '16px'}

WATCHLIST = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']

REGIME_COLORS = {
    'BULL': C['green'], 'BEAR': C['red'],
    'SIDEWAYS': C['yellow'], 'VOLATILE': C['purple'],
}

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(style={'backgroundColor': C['bg'], 'minHeight': '100vh',
                              'padding': '0 24px 40px'}, children=[

    # Header
    html.Div(style={'backgroundColor': C['header'], 'borderBottom': f"1px solid {C['border']}",
                    'padding': '16px 24px', 'marginBottom': '24px'}, children=[
        dbc.Row([
            dbc.Col(html.Span([
                html.Span("Stock AI", style={'color': C['blue'], 'fontWeight': 700, 'fontSize': '18px'}),
                html.Span("  /  Trading Bot Dashboard",
                          style={'color': C['muted'], 'fontSize': '16px'}),
            ]), width=8),
            dbc.Col(html.Div([
                html.Button("Refresh", id='refresh-btn',
                            style={'backgroundColor': C['blue'], 'color': '#000', 'border': 'none',
                                   'borderRadius': '6px', 'padding': '6px 16px',
                                   'fontWeight': 700, 'cursor': 'pointer', 'float': 'right'}),
            ]), width=4),
        ])
    ]),

    # Auto-refresh interval
    dcc.Interval(id='auto-refresh', interval=60_000, n_intervals=0),

    # ── Row 1: Market Regime + Learner Status ─────────────────────────────
    dbc.Row([
        dbc.Col(html.Div(id='regime-panel', style=CARD), width=6),
        dbc.Col(html.Div(id='learner-panel', style=CARD), width=6),
    ], className='g-3 mb-3'),

    # ── Row 2: Decision Engine Scores ─────────────────────────────────────
    html.Div(id='scores-panel', style=CARD),

    # ── Row 3: Portfolio Optimizer ────────────────────────────────────────
    html.Div(id='portfolio-panel', style=CARD),

    # ── Row 4: Position Manager ───────────────────────────────────────────
    html.Div(id='positions-panel', style=CARD),
])


# ── Callbacks — one per panel to avoid timeout ────────────────────────────────

@app.callback(
    Output('regime-panel', 'children'),
    Input('auto-refresh', 'n_intervals'),
    Input('refresh-btn', 'n_clicks'),
)
def cb_regime(_i, _c):
    return _build_regime_panel()


@app.callback(
    Output('learner-panel', 'children'),
    Input('auto-refresh', 'n_intervals'),
    Input('refresh-btn', 'n_clicks'),
)
def cb_learner(_i, _c):
    return _build_learner_panel()


@app.callback(
    Output('scores-panel', 'children'),
    Input('auto-refresh', 'n_intervals'),
    Input('refresh-btn', 'n_clicks'),
)
def cb_scores(_i, _c):
    return _build_scores_panel()


@app.callback(
    Output('portfolio-panel', 'children'),
    Input('auto-refresh', 'n_intervals'),
    Input('refresh-btn', 'n_clicks'),
)
def cb_portfolio(_i, _c):
    return _build_portfolio_panel()


@app.callback(
    Output('positions-panel', 'children'),
    Input('auto-refresh', 'n_intervals'),
    Input('refresh-btn', 'n_clicks'),
)
def cb_positions(_i, _c):
    return _build_positions_panel()


# ── Panel Builders ────────────────────────────────────────────────────────────

def _build_regime_panel():
    try:
        from trading.market_regime import MarketRegimeDetector
        ticker = yf.Ticker('RELIANCE.NS')
        data = ticker.history(period='1y', interval='1d', auto_adjust=True)
        detector = MarketRegimeDetector()
        signal = detector.detect(data)
        regime_name = signal.regime.value
        color = REGIME_COLORS.get(regime_name, C['blue'])

        rows = [
            _stat("Regime",          regime_name,                          color),
            _stat("Confidence",      f"{signal.confidence:.1%}",           C['text']),
            _stat("Trend Strength",  signal.trend_strength,                C['text']),
            _stat("ADX",             f"{signal.adx:.1f}",                  C['text']),
            _stat("vs SMA50",        f"{signal.price_vs_sma50:+.2f}%",     C['green'] if signal.price_vs_sma50 > 0 else C['red']),
            _stat("vs SMA200",       f"{signal.price_vs_sma200:+.2f}%",    C['green'] if signal.price_vs_sma200 > 0 else C['red']),
            _stat("Volatility",      f"{signal.volatility_pct:.2f}%",      C['yellow']),
            _stat("Recommendation",  signal.recommendation,                C['blue']),
        ]

        return [
            _section_title("Market Regime"),
            html.Div(style={'textAlign': 'center', 'margin': '12px 0'}, children=[
                html.Span(regime_name, style={'color': color, 'fontSize': '28px', 'fontWeight': 700}),
            ]),
            html.Div(rows),
        ]
    except Exception as e:
        return [_section_title("Market Regime"), _err(str(e))]


def _build_learner_panel():
    try:
        from trading.continuous_learner import ContinuousLearner
        learner = ContinuousLearner()
        s = learner.get_status()
        acc_color = C['green'] if s['model_enabled'] else C['yellow']

        history = s.get('performance_history', [])
        fig = go.Figure()
        if history:
            fig.add_trace(go.Scatter(
                x=list(range(len(history))),
                y=[h['accuracy'] for h in history],
                mode='lines+markers',
                line=dict(color=C['blue'], width=2),
                marker=dict(size=5),
            ))
        _dark_fig(fig, "Model Accuracy History", "Retrain #", "Accuracy", 180)

        return [
            _section_title("Continuous Learner"),
            dbc.Row([
                dbc.Col([
                    _stat("Model Enabled",      str(s['model_enabled']),          acc_color),
                    _stat("Current Accuracy",   f"{s['current_accuracy']:.3f}",   acc_color),
                    _stat("Best Accuracy",      f"{s['best_accuracy']:.3f}",      C['green']),
                    _stat("Trades Logged",      str(s['total_trades_logged']),     C['blue']),
                    _stat("Next Retrain In",    f"{s['next_retrain_in']} trades",  C['muted']),
                ], width=5),
                dbc.Col([dcc.Graph(figure=fig, config={'displayModeBar': False})], width=7),
            ]),
        ]
    except Exception as e:
        return [_section_title("Continuous Learner"), _err(str(e))]


def _build_scores_panel():
    try:
        from trading.decision_engine import TradingDecisionEngine
        engine = TradingDecisionEngine()
        rows = []
        for sym in WATCHLIST:
            try:
                ticker = yf.Ticker(sym)
                data = ticker.history(period='5d', interval='1d', auto_adjust=True)
                if data.empty:
                    continue
                decision, score, breakdown = engine.make_decision(sym, data)
                score_color = (C['green'] if score >= 60 else
                               C['yellow'] if score >= 40 else C['red'])
                rows.append(html.Tr([
                    html.Td(sym,      style={'color': C['text'],   'padding': '6px 8px'}),
                    html.Td(f"{score:.1f}",
                            style={'color': score_color, 'fontWeight': 700, 'padding': '6px 8px'}),
                    html.Td(decision, style={'color': score_color, 'padding': '6px 8px'}),
                    html.Td(f"{breakdown['component_scores'].get('technical') or 'N/A'}",
                            style={'color': C['muted'], 'padding': '6px 8px', 'fontSize': '12px'}),
                    html.Td(f"{breakdown['component_scores'].get('ml_prediction') or 'N/A'}",
                            style={'color': C['muted'], 'padding': '6px 8px', 'fontSize': '12px'}),
                ]))
            except Exception:
                continue

        table = html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
            html.Thead(html.Tr([
                html.Th(h, style={'color': C['muted'], 'fontSize': '11px', 'padding': '6px 8px',
                                  'textTransform': 'uppercase', 'borderBottom': f"1px solid {C['border']}"})
                for h in ['Symbol', 'Score', 'Decision', 'Technical', 'ML']
            ])),
            html.Tbody(rows),
        ])

        return [_section_title("Decision Engine — Watchlist Scores"), table]
    except Exception as e:
        return [_section_title("Decision Engine — Watchlist Scores"), _err(str(e))]


def _build_portfolio_panel():
    try:
        from trading.portfolio_optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(WATCHLIST[:4], period='3mo')
        result = opt.optimize()
        ms = result['max_sharpe']
        mv = result['min_volatility']

        def weight_bars(weights):
            bars = []
            for sym, w in weights.items():
                bars.append(html.Div(style={'marginBottom': '6px'}, children=[
                    html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                                   'marginBottom': '2px'}, children=[
                        html.Span(sym,    style={'color': C['muted'],  'fontSize': '12px'}),
                        html.Span(f"{w:.1%}", style={'color': C['text'], 'fontSize': '12px'}),
                    ]),
                    html.Div(style={'backgroundColor': C['border'], 'borderRadius': '3px', 'height': '5px'}, children=[
                        html.Div(style={'backgroundColor': C['blue'], 'width': f"{w*100:.1f}%",
                                        'height': '5px', 'borderRadius': '3px'})
                    ])
                ]))
            return bars

        return [
            _section_title("Portfolio Optimizer (MPT)"),
            dbc.Row([
                dbc.Col([
                    html.P("Max Sharpe", style={'color': C['blue'], 'fontWeight': 600,
                                                'fontSize': '13px', 'marginBottom': '8px'}),
                    _stat("Expected Return", f"{ms['annual_return']:.2%}",
                          C['green'] if ms['annual_return'] > 0 else C['red']),
                    _stat("Volatility",      f"{ms['annual_volatility']:.2%}", C['yellow']),
                    _stat("Sharpe Ratio",    f"{ms['sharpe_ratio']:.3f}",
                          C['green'] if ms['sharpe_ratio'] > 0 else C['red']),
                    html.Div(weight_bars(ms['weights']), style={'marginTop': '10px'}),
                ], width=4),
                dbc.Col([
                    html.P("Min Volatility", style={'color': C['purple'], 'fontWeight': 600,
                                                    'fontSize': '13px', 'marginBottom': '8px'}),
                    _stat("Expected Return", f"{mv['annual_return']:.2%}",
                          C['green'] if mv['annual_return'] > 0 else C['red']),
                    _stat("Volatility",      f"{mv['annual_volatility']:.2%}", C['yellow']),
                    _stat("Sharpe Ratio",    f"{mv['sharpe_ratio']:.3f}",
                          C['green'] if mv['sharpe_ratio'] > 0 else C['red']),
                    html.Div(weight_bars(mv['weights']), style={'marginTop': '10px'}),
                ], width=4),
                dbc.Col([
                    html.P("Recommended", style={'color': C['green'], 'fontWeight': 600,
                                                  'fontSize': '13px', 'marginBottom': '8px'}),
                    html.Span(result['recommended']['strategy'].replace('_', ' ').title(),
                              style={'color': C['green'], 'fontSize': '20px', 'fontWeight': 700}),
                    html.Br(),
                    html.Span(f"Sharpe: {result['recommended']['sharpe_ratio']:.3f}",
                              style={'color': C['muted'], 'fontSize': '13px'}),
                ], width=4),
            ]),
        ]
    except Exception as e:
        return [_section_title("Portfolio Optimizer (MPT)"), _err(str(e))]


def _build_positions_panel():
    try:
        from trading.position_manager import PositionManager, PositionConfig
        from brokers.paper_broker import PaperBroker
        from trading.order_executor import OrderExecutor
        broker = PaperBroker()
        executor = OrderExecutor(broker=broker)
        pm = PositionManager(broker=broker, executor=executor)
        summary = pm.get_summary()

        rows = []
        for sym in summary.get('symbols', []):
            pos = pm._positions.get(sym)
            if pos:
                rows.append(html.Tr([
                    html.Td(sym,                              style={'color': C['text'],  'padding': '6px 8px'}),
                    html.Td(str(pos.side.value if hasattr(pos.side, 'value') else pos.side),
                                                              style={'color': C['blue'],  'padding': '6px 8px'}),
                    html.Td(f"₹{pos.entry_price:.2f}",       style={'color': C['muted'], 'padding': '6px 8px'}),
                    html.Td(f"{pos.quantity}",                style={'color': C['text'],  'padding': '6px 8px'}),
                    html.Td("ACTIVE",                         style={'color': C['green'], 'fontWeight': 700, 'padding': '6px 8px'}),
                ]))

        table = html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
            html.Thead(html.Tr([
                html.Th(h, style={'color': C['muted'], 'fontSize': '11px', 'padding': '6px 8px',
                                  'textTransform': 'uppercase',
                                  'borderBottom': f"1px solid {C['border']}"})
                for h in ['Symbol', 'Side', 'Entry', 'Qty', 'Status']
            ])),
            html.Tbody(rows if rows else [
                html.Tr([html.Td("No open positions", colSpan=5,
                                 style={'color': C['muted'], 'padding': '12px 8px',
                                        'textAlign': 'center', 'fontSize': '13px'})])
            ]),
        ])

        halt_color = C['red'] if summary.get('trading_halted') else C['green']
        return [
            _section_title("Position Manager"),
            dbc.Row([
                dbc.Col([
                    _stat("Active Positions", str(summary.get('active_positions', 0)), C['blue']),
                    _stat("Daily P&L",        f"₹{summary.get('daily_pnl', 0):+,.2f}",
                          C['green'] if summary.get('daily_pnl', 0) >= 0 else C['red']),
                    _stat("Trading Halted",   str(summary.get('trading_halted', False)), halt_color),
                ], width=3),
                dbc.Col([table], width=9),
            ]),
        ]
    except Exception as e:
        return [_section_title("Position Manager"), _err(str(e))]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section_title(text):
    return html.H6(text, style={'color': C['text'], 'fontWeight': 600,
                                 'marginBottom': '12px', 'fontSize': '14px',
                                 'textTransform': 'uppercase', 'letterSpacing': '0.8px'})


def _stat(label, value, color):
    return html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                            'padding': '3px 0', 'borderBottom': f"1px solid {C['border']}"}, children=[
        html.Span(label, style={'color': C['muted'], 'fontSize': '12px'}),
        html.Span(value, style={'color': color,      'fontSize': '12px', 'fontWeight': 600}),
    ])


def _err(msg):
    return html.Div(f"Error: {msg}", style={'color': C['red'], 'fontSize': '12px'})


def _dark_fig(fig, title, xtitle, ytitle, height):
    fig.update_layout(
        title=dict(text=title, font=dict(color=C['text'], size=12)),
        xaxis=dict(title=xtitle, gridcolor=C['border'], color=C['muted'], zeroline=False),
        yaxis=dict(title=ytitle, gridcolor=C['border'], color=C['muted'], zeroline=False),
        paper_bgcolor=C['card'], plot_bgcolor=C['card'],
        font=dict(color=C['text']), height=height,
        margin=dict(l=40, r=10, t=30, b=30), showlegend=False,
    )


if __name__ == '__main__':
    print("Starting Trading Bot Dashboard...")
    print("Access at: http://localhost:8052")
    app.run(debug=True, host='0.0.0.0', port=8052)
