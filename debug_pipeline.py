"""Diagnose which strategies are contributing signals."""
import sys
sys.path.insert(0, '.')
import yfinance as yf
from scalping.ensemble_scorer import EnsembleScorer
from scalping.filters.session_filter import SessionFilter
from scalping.filters.regime_filter import VolatilityRegimeDetector

data = yf.Ticker('RELIANCE.NS').history(period='7d', interval='1m')
scorer = EnsembleScorer()
sf = SessionFilter()
rd = VolatilityRegimeDetector()

ema_df  = scorer._ema.generate_signals(data)
vwap_df = scorer._vwap.generate_signals(data)
rsi_df  = scorer._rsi.generate_signals(data)

# Count signals per strategy inside session window
for name, df in [('EMA', ema_df), ('VWAP', vwap_df), ('RSI', rsi_df)]:
    sigs = df[df['signal'].isin(['BUY','SELL'])]
    in_sess = [i for i, ts in enumerate(sigs.index) if sf.is_trading_time(ts, 'NSE')]
    print(f"{name}: total={len(sigs)} in_session={len(in_sess)}")

# What does ensemble pass through?
result = scorer.score_all(data)
ens_sigs = result[result['ensemble_signal'].isin(['BUY','SELL'])]
in_sess_ens = [(ts, row['ensemble_signal'], row['ensemble_score'], row['ensemble_agreement'])
               for ts, row in ens_sigs.iterrows() if sf.is_trading_time(ts, 'NSE')]
print(f"\nEnsemble in session: {len(in_sess_ens)}")
for ts, sig, score, ag in in_sess_ens:
    # Check which strategies fired at this candle
    idx = data.index.get_loc(ts)
    e = ema_df.iloc[idx]['signal']
    v = vwap_df.iloc[idx]['signal']
    r = rsi_df.iloc[idx]['signal']
    print(f"  {ts.strftime('%m-%d %H:%M')} | {sig} score={score:.0f} ag={ag} | EMA={e} VWAP={v} RSI={r}")
