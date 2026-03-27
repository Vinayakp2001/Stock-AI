# Implementation Tasks: Improve Scalping Strategy Win Rate to 60%+

- [x] 1. Create scalping filters directory and session filter





  - Create `scalping/filters/__init__.py`
  - Implement `SessionFilter` class in `scalping/filters/session_filter.py`
  - Add NSE (9:15-11:15 IST) and NYSE (9:30-11:30 EST) trading windows
  - Add position close trigger at 15 min before market close
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement volatility regime detector


  - Implement `VolatilityRegimeDetector` in `scalping/filters/regime_filter.py`
  - Add ADX calculation (use `ta` library)
  - Implement regime classification: choppy (ADX<20), trending (20-40), strong (>40)
  - Add position size multiplier based on regime
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 3. Build ensemble scorer


  - Implement `EnsembleScorer` in `scalping/ensemble_scorer.py`
  - Add weighted scoring: EMA 40%, VWAP 35%, RSI 25%
  - Add agreement bonus: +10 for 2/3 agree, +20 for 3/3 agree
  - Set threshold: score >= 70 to generate signal
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 4. Implement adaptive stop loss


  - Create `scalping/risk/__init__.py`
  - Implement `AdaptiveStopLoss` in `scalping/risk/adaptive_sl.py`
  - Add ATR multiplier logic (2.0x normal, 2.5x high volatility)
  - Enforce min 0.15% and max 0.5% SL distance
  - Always set TP at 3x SL distance (1:3 RR)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 5. Build ML signal confirmer


  - Create `scalping/ml/__init__.py`
  - Implement `MLSignalConfirmer` in `scalping/ml/signal_confirmer.py`
  - Use RandomForestClassifier with 8 features (RSI, MACD, EMA diff, volume ratio, ATR, ADX, VWAP dist, momentum)
  - Add 80/20 temporal train/test split
  - Disable ML filter if accuracy < 55%, log warning
  - Save trained model to `models/scalping/signal_classifier.joblib`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [x] 6. Create improved strategy orchestrator


  - Implement `ImprovedScalpingStrategy` in `scalping/strategies/improved_strategy.py`
  - Wire all 5 components in sequence: session → regime → ensemble → ML → adaptive SL
  - Add rejection reason logging for every filtered signal
  - Add `train_ml_model()` method
  - Update `scalping/strategies/__init__.py` to export new strategy
  - _Requirements: 1.1-1.5, 2.1-2.6, 3.1-3.7, 4.1-4.6, 5.1-5.7_

- [x] 7. Update backtester and run_scalping to use improved strategy


  - Add `improved` as a strategy option in `scalping/run_scalping.py`
  - Update `STRATEGIES` dict to include `ImprovedScalpingStrategy`
  - Add `--train-ml` flag to run_scalping.py for ML model training
  - _Requirements: 6.1, 6.2_

- [x] 8. Add validation report generation


  - Add `run_validation()` function in `scalping/run_scalping.py`
  - Run backtest on RELIANCE.NS, TCS.NS, HDFCBANK.NS
  - Compare improved vs baseline strategy win rates
  - Save report to `data/scalping/validation_report.json`
  - Print clear pass/fail against 60% win rate target
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ]* 9. Write unit tests for all new components
  - `tests/test_session_filter.py` - boundary conditions
  - `tests/test_regime_filter.py` - ADX thresholds
  - `tests/test_ensemble_scorer.py` - scoring logic
  - `tests/test_adaptive_sl.py` - SL/TP calculations
  - `tests/test_ml_confirmer.py` - training and prediction
  - _Requirements: All_

- [x] 10. Update scalping dashboard to show improved strategy results



  - Add `improved` option to strategy dropdown in `scalping/dashboard.py`
  - Show rejection reason breakdown (how many signals filtered by each layer)
  - Show ML model accuracy in results
  - _Requirements: 6.2, 6.3_
