#!/usr/bin/env python3
"""
Scalping Dashboard - Standalone page for scalping backtesting and paper trading
Run: python scalping/dashboard.py
Access: http://localhost:8051
"""
import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from scalping.config import (
    CONSERVATIVE, AGGRESSIVE, VALIDATION_GATE,
    RECOMMENDED_STOCKS
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "Scalping Module - Stock AI"

# Fix dropdown visibility - Dash dropdowns ignore inline styles for the menu
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dropdown menu background and text */
            .Select-menu-outer {
                background-color: #1c2128 !important;
                border: 1px solid #30363d !important;
            }
            .VirtualizedSelectOption {
                background-color: #1c2128 !important;
                color: #e6edf3 !important;
            }
            .VirtualizedSelectFocusedOption {
                background-color: #2d333b !important;
                color: #58a6ff !important;
            }
            .Select-control {
                background-color: #0d1117 !important;
                border: 1px solid #30363d !important;
                color: #e6edf3 !important;
            }
            .Select-value-label {
                color: #e6edf3 !important;
            }
            .Select-placeholder {
                color: #8b949e !important;
            }
            .Select-input input {
                color: #e6edf3 !important;
            }
            .Select-arrow {
                border-top-color: #8b949e !important;
            }
            /* Dash newer dropdown */
            .dash-dropdown .Select-control {
                background-color: #0d1117 !important;
            }
            .Select--single > .Select-control .Select-value {
                color: #e6edf3 !important;
            }
            .Select-option {
                background-color: #1c2128 !important;
                color: #e6edf3 !important;
            }
            .Select-option:hover, .Select-option.is-focused {
                background-color: #2d333b !important;
                color: #58a6ff !important;
            }
            .Select-option.is-selected {
                background-color: #1f3a5f !important;
                color: #58a6ff !important;
            }
            /* Scrollbar */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: #0d1117; }
            ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
            /* Input fields */
            input[type=number] {
                background-color: #0d1117 !important;
                color: #e6edf3 !important;
                border: 1px solid #30363d !important;
            }
            body { background-color: #0d1117; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

ALL_STOCKS = RECOMMENDED_STOCKS['NSE'] + RECOMMENDED_STOCKS['NYSE']

# ─── Color Palette ────────────────────────────────────────────────────────────
C = {
    'bg':        '#0d1117',   # page background
    'card':      '#161b22',   # card background
    'border':    '#30363d',   # card border
    'green':     '#3fb950',   # profit / win
    'red':       '#f85149',   # loss / fail
    'yellow':    '#d29922',   # warning / neutral
    'blue':      '#58a6ff',   # info / primary
    'purple':    '#bc8cff',   # aggressive mode
    'teal':      '#39d353',   # equity curve line
    'text':      '#e6edf3',   # primary text
    'muted':     '#8b949e',   # secondary text
    'header_bg': '#1c2128',   # header/navbar bg
}

INLINE = {
    'page':   {'backgroundColor': C['bg'], 'minHeight': '100vh', 'padding': '0 24px 40px'},
    'card':   {'backgroundColor': C['card'], 'border': f"1px solid {C['border']}", 'borderRadius': '8px'},
    'header': {'backgroundColor': C['header_bg'], 'borderBottom': f"1px solid {C['border']}",
               'padding': '16px 24px', 'marginBottom': '24px'},
}

# ─── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div(style=INLINE['page'], children=[

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div(style=INLINE['header'], children=[
        dbc.Row([
            dbc.Col([
                html.Span("Stock AI", style={'color': C['blue'], 'fontWeight': 700, 'fontSize': '18px'}),
                html.Span("  /  Scalping Module", style={'color': C['muted'], 'fontSize': '16px'}),
            ], width=6),
            dbc.Col([
                html.Span("Benchmark: Conservative 1-2% daily  |  Aggressive 2-3% daily",
                          style={'color': C['muted'], 'fontSize': '12px', 'float': 'right'})
            ], width=6)
        ])
    ]),

    # ── Controls ────────────────────────────────────────────────────────────
    html.Div(style={**INLINE['card'], 'padding': '20px', 'marginBottom': '20px'}, children=[
        dbc.Row([
            dbc.Col([
                html.Label("Symbol", style={'color': C['muted'], 'fontSize': '12px', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='scalp-symbol',
                    options=[{'label': s, 'value': s} for s in ALL_STOCKS],
                    value='RELIANCE.NS',
                    style={'backgroundColor': C['bg'], 'color': C['text']},
                )
            ], width=3),
            dbc.Col([
                html.Label("Strategy", style={'color': C['muted'], 'fontSize': '12px', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='scalp-strategy',
                    options=[
                        {'label': 'EMA Crossover (9/21)', 'value': 'ema'},
                        {'label': 'VWAP Bounce', 'value': 'vwap'},
                        {'label': 'RSI Scalp (35/65)', 'value': 'rsi'},
                        {'label': 'Improved (5-layer filter)', 'value': 'improved'},
                    ],
                    value='ema',
                    style={'backgroundColor': C['bg'], 'color': C['text']},
                )
            ], width=3),
            dbc.Col([
                html.Label("Mode", style={'color': C['muted'], 'fontSize': '12px', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='scalp-mode',
                    options=[
                        {'label': 'Conservative  (1-2% daily)', 'value': 'conservative'},
                        {'label': 'Aggressive  (2-3% daily)', 'value': 'aggressive'},
                    ],
                    value='conservative',
                    style={'backgroundColor': C['bg'], 'color': C['text']},
                )
            ], width=3),
            dbc.Col([
                html.Label("Capital (₹)", style={'color': C['muted'], 'fontSize': '12px', 'marginBottom': '4px'}),
                dbc.Input(id='scalp-capital', type='number', value=100000, min=50000, step=10000,
                          style={'backgroundColor': C['bg'], 'color': C['text'],
                                 'border': f"1px solid {C['border']}"}),
            ], width=2),
            dbc.Col([
                html.Label("\u00a0", style={'color': C['muted'], 'fontSize': '12px', 'marginBottom': '4px',
                                            'display': 'block'}),
                html.Button("Run Backtest", id='scalp-run-btn',
                            style={
                                'backgroundColor': C['blue'], 'color': '#000',
                                'border': 'none', 'borderRadius': '6px',
                                'padding': '8px 16px', 'fontWeight': 700,
                                'width': '100%', 'cursor': 'pointer', 'fontSize': '14px'
                            })
            ], width=1),
        ], align="end")
    ]),

    # ── Results ─────────────────────────────────────────────────────────────
    dcc.Loading(
        id="scalp-loading",
        type="circle",
        color=C['blue'],
        children=html.Div(id='scalp-results')
    ),

    # ── Paper Trading ───────────────────────────────────────────────────────
    html.Div(style={'marginTop': '24px'}, children=[
        html.H6("Paper Trading Validation Progress",
                style={'color': C['muted'], 'fontSize': '13px', 'marginBottom': '12px',
                       'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Div(id='paper-status')
    ])
])


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output('scalp-results', 'children'),
    Input('scalp-run-btn', 'n_clicks'),
    State('scalp-symbol', 'value'),
    State('scalp-strategy', 'value'),
    State('scalp-mode', 'value'),
    State('scalp-capital', 'value'),
    prevent_initial_call=True
)
def run_backtest(n_clicks, symbol, strategy_name, mode, capital):
    if not symbol or not strategy_name:
        return _alert("Please select a symbol and strategy.", C['yellow'])
    try:
        from scalping.backtester import ScalpingBacktester
        from scalping.strategies.ema_crossover import EMACrossoverStrategy
        from scalping.strategies.vwap_strategy import VWAPStrategy
        from scalping.strategies.rsi_scalp import RSIScalpStrategy
        from scalping.strategies.improved_strategy import ImprovedScalpingStrategy

        strats = {
            'ema':      EMACrossoverStrategy(),
            'vwap':     VWAPStrategy(),
            'rsi':      RSIScalpStrategy(),
            'improved': ImprovedScalpingStrategy(),
        }
        backtester = ScalpingBacktester(initial_capital=capital or 100000)
        strategy   = strats[strategy_name]
        result     = backtester.run_backtest(strategy, symbol, period="7d", interval="1m", mode=mode)

        # For improved strategy, also fetch signal DataFrame for rejection breakdown
        extra = {}
        if strategy_name == 'improved':
            try:
                data       = backtester.fetch_data(symbol, period="7d", interval="1m")
                signals_df = strategy.generate_signals(data)
                extra['rejection_breakdown'] = _compute_rejection_breakdown(signals_df)
                extra['ml_accuracy']         = strategy.ml_confirmer.accuracy
                extra['ml_enabled']          = strategy.ml_confirmer._ml_enabled
            except Exception:
                pass

        return build_results(result, symbol, strategy_name, mode, capital or 100000, extra)
    except Exception as e:
        return _alert(f"Error: {str(e)}", C['red'])


@app.callback(
    Output('paper-status', 'children'),
    Input('scalp-run-btn', 'n_clicks'),
    State('scalp-capital', 'value'),
    prevent_initial_call=True
)
def update_paper_status(n_clicks, capital):
    try:
        from scalping.paper_trader import PaperTrader
        pt = PaperTrader(capital=capital or 100000)
        return build_paper_status(pt._get_live_trading_status("all"))
    except Exception as e:
        return _alert(f"Paper trading status error: {e}", C['yellow'])


# ─── UI Builders ──────────────────────────────────────────────────────────────

def _compute_rejection_breakdown(signals_df) -> dict:
    """Count rejection reasons from improved strategy signals DataFrame."""
    reasons = {}
    if 'rejection_reason' not in signals_df.columns:
        return reasons
    for reason in signals_df['rejection_reason']:
        if not reason:
            continue
        key = reason.split('|')[0]   # strip dynamic values like adx=xx
        reasons[key] = reasons.get(key, 0) + 1
    return dict(sorted(reasons.items(), key=lambda x: -x[1]))


def build_results(result, symbol, strategy_name, mode, capital, extra=None):
    extra = extra or {}
    labels = {'ema': 'EMA Crossover', 'vwap': 'VWAP Bounce', 'rsi': 'RSI Scalp', 'improved': 'Improved (5-layer)'}

    # ── Metric cards ──────────────────────────────────────────────────────
    def metric(title, value, color, sub=""):
        return html.Div(style={**INLINE['card'], 'padding': '16px', 'textAlign': 'center'}, children=[
            html.P(title, style={'color': C['muted'], 'fontSize': '11px', 'marginBottom': '4px',
                                  'textTransform': 'uppercase', 'letterSpacing': '0.8px'}),
            html.H4(value, style={'color': color, 'margin': '0', 'fontWeight': 700}),
            html.P(sub, style={'color': C['muted'], 'fontSize': '11px', 'marginTop': '4px', 'marginBottom': 0})
        ])

    wr_color  = C['green'] if result.win_rate >= 0.6 else C['yellow'] if result.win_rate >= 0.4 else C['red']
    pf_color  = C['green'] if result.profit_factor >= 1.5 else C['yellow'] if result.profit_factor >= 1.0 else C['red']
    pnl_color = C['green'] if result.total_net_pnl >= 0 else C['red']
    dd_color  = C['green'] if result.max_drawdown_pct <= 0.05 else C['yellow'] if result.max_drawdown_pct <= 0.10 else C['red']
    day_color = C['green'] if result.avg_daily_return_pct >= 0.01 else C['yellow'] if result.avg_daily_return_pct >= 0 else C['red']

    metrics = dbc.Row([
        dbc.Col(metric("Win Rate",     f"{result.win_rate:.1%}",              wr_color,
                       f"Target {CONSERVATIVE['win_rate_target']:.0%}+"), width=2),
        dbc.Col(metric("Profit Factor", f"{result.profit_factor:.2f}",        pf_color,  "Target 1.5+"), width=2),
        dbc.Col(metric("Net P&L",      f"₹{result.total_net_pnl:+,.0f}",     pnl_color,
                       f"{result.total_return_pct:.2%} return"), width=2),
        dbc.Col(metric("Avg Daily",    f"{result.avg_daily_return_pct:.2%}",  day_color,
                       f"Target {CONSERVATIVE['daily_net_target_pct']:.1%}+"), width=2),
        dbc.Col(metric("Max Drawdown", f"{result.max_drawdown_pct:.1%}",      dd_color,  "Max 10%"), width=2),
        dbc.Col(metric("Trades",       str(result.total_trades),              C['blue'],
                       f"W:{result.winning_trades} L:{result.losing_trades}"), width=2),
    ], className="mb-3 g-2")

    # ── Equity curve ──────────────────────────────────────────────────────
    eq_fig = go.Figure()
    if result.equity_curve:
        above = [v if v >= capital else None for v in result.equity_curve]
        below = [v if v < capital else None for v in result.equity_curve]
        eq_fig.add_trace(go.Scatter(y=above, mode='lines', name='Profit',
                                    line=dict(color=C['green'], width=2),
                                    fill='tozeroy', fillcolor='rgba(63,185,80,0.08)'))
        eq_fig.add_trace(go.Scatter(y=below, mode='lines', name='Loss',
                                    line=dict(color=C['red'], width=2),
                                    fill='tozeroy', fillcolor='rgba(248,81,73,0.08)'))
        eq_fig.add_hline(y=capital, line_dash="dot", line_color=C['muted'],
                         annotation_text="Initial Capital",
                         annotation_font_color=C['muted'])
    _dark_chart(eq_fig, f"{symbol} — {labels[strategy_name]} Equity Curve", "Trade #", "Portfolio Value (₹)", 360)

    # ── Daily returns ─────────────────────────────────────────────────────
    day_fig = go.Figure()
    if result.daily_returns:
        bar_colors = [C['green'] if r >= 0 else C['red'] for r in result.daily_returns]
        day_fig.add_trace(go.Bar(
            y=[r * 100 for r in result.daily_returns],
            marker_color=bar_colors,
            marker_line_width=0,
            name='Daily %'
        ))
        day_fig.add_hline(y=0, line_color=C['border'])
    _dark_chart(day_fig, "Daily Returns (%)", "Day", "Return (%)", 240)

    # ── Validation gate ───────────────────────────────────────────────────
    gate_color = C['green'] if result.validation_passed else C['red']
    gate_items = []
    for k, v in result.validation_details.items():
        if 'check' in k:
            gate_items.append(
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                                'padding': '4px 0', 'borderBottom': f"1px solid {C['border']}"},
                         children=[
                             html.Span(k.replace('_check', '').replace('_', ' ').title(),
                                       style={'color': C['muted'], 'fontSize': '12px'}),
                             html.Span(v, style={'color': C['text'], 'fontSize': '12px'})
                         ])
            )

    validation_card = html.Div(style={**INLINE['card'], 'padding': '16px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '12px'}, children=[
            html.Span("Validation Gate", style={'color': C['text'], 'fontWeight': 600}),
            html.Span("PASSED" if result.validation_passed else "FAILED",
                      style={'color': gate_color, 'fontWeight': 700, 'fontSize': '13px'})
        ]),
        html.Div(gate_items),
        html.Div(style={'display': 'flex', 'gap': '12px', 'marginTop': '12px'}, children=[
            _badge("Conservative", C['green'] if result.meets_conservative_target else C['red']),
            _badge("Aggressive",   C['purple'] if result.meets_aggressive_target else C['red']),
        ])
    ])

    # ── Cost card ─────────────────────────────────────────────────────────
    cost_card = html.Div(style={**INLINE['card'], 'padding': '16px'}, children=[
        html.Span("Cost & Risk Analysis", style={'color': C['text'], 'fontWeight': 600, 'display': 'block',
                                                  'marginBottom': '12px'}),
        _row_stat("Total Costs",    f"₹{result.total_transaction_costs:,.0f}", C['yellow']),
        _row_stat("Cost Impact",    f"{result.cost_impact_pct:.2%}",           C['yellow']),
        _row_stat("Avg Win",        f"{result.avg_win_pct:.3%}",               C['green']),
        _row_stat("Avg Loss",       f"{result.avg_loss_pct:.3%}",              C['red']),
        _row_stat("Risk / Reward",  f"1 : {result.risk_reward_ratio:.1f}",     C['blue']),
        _row_stat("Sharpe Ratio",   f"{result.sharpe_ratio:.2f}",
                  C['green'] if result.sharpe_ratio > 1 else C['yellow']),
    ])

    # ── Benchmark targets ─────────────────────────────────────────────────
    bench_card = html.Div(style={**INLINE['card'], 'padding': '16px'}, children=[
        html.Span("Benchmark Targets", style={'color': C['text'], 'fontWeight': 600,
                                               'display': 'block', 'marginBottom': '12px'}),
        dbc.Row([
            dbc.Col([
                html.P("Conservative", style={'color': C['blue'], 'fontWeight': 600,
                                               'fontSize': '12px', 'marginBottom': '6px'}),
                _row_stat("Win Rate",  f"{CONSERVATIVE['win_rate_target']:.0%}+",       C['blue']),
                _row_stat("Daily Net", f"{CONSERVATIVE['daily_net_target_pct']:.1%}+",  C['blue']),
                _row_stat("Monthly",   f"{CONSERVATIVE['monthly_target_pct']:.0%}+",    C['blue']),
            ], width=4),
            dbc.Col([
                html.P("Aggressive", style={'color': C['purple'], 'fontWeight': 600,
                                             'fontSize': '12px', 'marginBottom': '6px'}),
                _row_stat("Win Rate",  f"{AGGRESSIVE['win_rate_target']:.0%}+",         C['purple']),
                _row_stat("Daily Net", f"{AGGRESSIVE['daily_net_target_pct']:.1%}+",    C['purple']),
                _row_stat("Monthly",   f"{AGGRESSIVE['monthly_target_pct']:.0%}+",      C['purple']),
            ], width=4),
            dbc.Col([
                html.P("Gate", style={'color': C['teal'], 'fontWeight': 600,
                                       'fontSize': '12px', 'marginBottom': '6px'}),
                _row_stat("Min Win",   f"{VALIDATION_GATE['min_win_rate']:.0%}",        C['teal']),
                _row_stat("Min Trades",str(VALIDATION_GATE['min_trades_to_validate']),  C['teal']),
                _row_stat("Paper Days",str(VALIDATION_GATE['min_paper_trading_days']),  C['teal']),
            ], width=4),
        ])
    ])

    return html.Div([
        metrics,
        dbc.Row([
            dbc.Col([dcc.Graph(figure=eq_fig,  config={'displayModeBar': False})], width=8),
            dbc.Col([dcc.Graph(figure=day_fig, config={'displayModeBar': False})], width=4),
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col([validation_card], width=4),
            dbc.Col([cost_card],       width=4),
            dbc.Col([bench_card],      width=4),
        ], className="g-2"),
        _build_improved_extras(extra) if extra else html.Div()
    ])


def _build_improved_extras(extra: dict):
    """Render rejection breakdown and ML accuracy for the improved strategy."""
    children = []

    # ML accuracy card
    ml_acc     = extra.get('ml_accuracy', 0)
    ml_enabled = extra.get('ml_enabled', False)
    ml_color   = C['green'] if ml_enabled else C['yellow']
    ml_label   = f"{ml_acc:.1%}" if ml_acc > 0 else "Not trained"
    ml_status  = "Active" if ml_enabled else "Disabled (accuracy < 55%)"

    ml_card = html.Div(style={**INLINE['card'], 'padding': '16px'}, children=[
        html.Span("ML Signal Confirmer", style={'color': C['text'], 'fontWeight': 600,
                                                 'display': 'block', 'marginBottom': '12px'}),
        _row_stat("Test Accuracy", ml_label,  ml_color),
        _row_stat("Status",        ml_status, ml_color),
        _row_stat("Threshold",     "55%",     C['muted']),
    ])

    # Rejection breakdown card
    breakdown = extra.get('rejection_breakdown', {})
    total_rejected = sum(breakdown.values())
    breakdown_rows = []
    for reason, count in breakdown.items():
        pct = count / total_rejected * 100 if total_rejected > 0 else 0
        breakdown_rows.append(_row_stat(reason.replace('_', ' ').title(), f"{count} ({pct:.0f}%)", C['yellow']))

    rejection_card = html.Div(style={**INLINE['card'], 'padding': '16px'}, children=[
        html.Span("Signal Rejection Breakdown", style={'color': C['text'], 'fontWeight': 600,
                                                        'display': 'block', 'marginBottom': '12px'}),
        html.Div(breakdown_rows if breakdown_rows else [
            html.Span("No rejections recorded", style={'color': C['muted'], 'fontSize': '12px'})
        ])
    ])

    children = dbc.Row([
        dbc.Col([ml_card],        width=4),
        dbc.Col([rejection_card], width=8),
    ], className="g-2 mt-2")

    return children


def build_paper_status(status):
    unlocked = status['live_trading_unlocked']
    bar_color = C['green'] if unlocked else C['yellow']

    def pbar(label, value, total):
        pct = min(100, int(value / max(1, total) * 100))
        return html.Div(style={'marginBottom': '10px'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'}, children=[
                html.Span(label, style={'color': C['muted'], 'fontSize': '12px'}),
                html.Span(f"{value}/{total}", style={'color': C['text'], 'fontSize': '12px'})
            ]),
            html.Div(style={'backgroundColor': C['border'], 'borderRadius': '4px', 'height': '6px'}, children=[
                html.Div(style={'backgroundColor': bar_color, 'width': f"{pct}%",
                                'height': '6px', 'borderRadius': '4px',
                                'transition': 'width 0.4s ease'})
            ])
        ])

    win_pct_val = 0
    try:
        win_pct_val = float(status['current_win_rate'].strip('%'))
    except Exception:
        pass

    return html.Div(style={**INLINE['card'], 'padding': '16px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '16px'}, children=[
            html.Span("Live Trading Status",
                      style={'color': C['text'], 'fontWeight': 600}),
            html.Span("UNLOCKED" if unlocked else "LOCKED",
                      style={'color': C['green'] if unlocked else C['red'],
                             'fontWeight': 700, 'fontSize': '13px'})
        ]),
        dbc.Row([
            dbc.Col([pbar("Days Traded",   status['days_traded'],         status['days_required'])],  width=4),
            dbc.Col([pbar("Paper Trades",  status['total_paper_trades'],  status['trades_required'])], width=4),
            dbc.Col([pbar(f"Win Rate (need {status['required_win_rate']})",
                          int(win_pct_val), 100)], width=4),
        ]),
        html.Div(style={
            'backgroundColor': 'rgba(63,185,80,0.1)' if unlocked else 'rgba(210,153,34,0.1)',
            'border': f"1px solid {C['green'] if unlocked else C['yellow']}",
            'borderRadius': '6px', 'padding': '10px 14px', 'marginTop': '8px'
        }, children=[
            html.Span(status['message'],
                      style={'color': C['green'] if unlocked else C['yellow'], 'fontSize': '13px'})
        ])
    ])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _dark_chart(fig, title, xtitle, ytitle, height):
    fig.update_layout(
        title=dict(text=title, font=dict(color=C['text'], size=13)),
        xaxis=dict(title=xtitle, gridcolor=C['border'], color=C['muted'],
                   showgrid=True, zeroline=False),
        yaxis=dict(title=ytitle, gridcolor=C['border'], color=C['muted'],
                   showgrid=True, zeroline=False),
        paper_bgcolor=C['card'],
        plot_bgcolor=C['card'],
        font=dict(color=C['text']),
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=False
    )


def _row_stat(label, value, color):
    return html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                            'padding': '3px 0', 'borderBottom': f"1px solid {C['border']}"}, children=[
        html.Span(label, style={'color': C['muted'], 'fontSize': '12px'}),
        html.Span(value, style={'color': color,      'fontSize': '12px', 'fontWeight': 600})
    ])


def _badge(label, color):
    return html.Span(label, style={
        'backgroundColor': f"{color}22",
        'color': color,
        'border': f"1px solid {color}",
        'borderRadius': '4px',
        'padding': '2px 10px',
        'fontSize': '12px',
        'fontWeight': 600
    })


def _alert(msg, color):
    return html.Div(msg, style={
        'backgroundColor': f"{color}22",
        'border': f"1px solid {color}",
        'color': color,
        'borderRadius': '6px',
        'padding': '12px 16px',
        'fontSize': '13px'
    })


if __name__ == '__main__':
    print("Starting Scalping Dashboard...")
    print("Access at: http://localhost:8051")
    app.run(debug=True, host='0.0.0.0', port=8051)
