import sys
sys.path.insert(0, '.')
import yfinance as yf
from scalping.strategies.ema_crossover import EMACrossoverStrategy
from scalping.strategies.vwap_strategy import VWAPStrategy
from scalping.strategies.rsi_scalp import RSIScalpStrategy

data = yf.Ticker('RELIANCE.NS').history(period='7d', interval='1m')
print(f'Total candles: {len(data)}')

for name, strat in [('EMA', EMACrossoverStrategy()), ('VWAP', VWAPStrategy()), ('RSI', RSIScalpStrategy())]:
    df = strat.generate_signals(data)
    buys  = (df['signal'] == 'BUY').sum()
    sells = (df['signal'] == 'SELL').sum()
    if buys + sells > 0:
        scores = df[df['signal'].isin(['BUY', 'SELL'])]['signal_score']
        print(f'{name}: BUY={buys} SELL={sells} | avg_score={scores.mean():.1f} min={scores.min():.0f} max={scores.max():.0f}')
    else:
        print(f'{name}: BUY=0 SELL=0 (no signals at all)')
