# momentum-across-markets

Cross-sectional momentum across the US (S&P 500), India (BSE), and Japan (Nikkei 225) using monthly data (2004–2024).  
Python backtest with performance, drawdowns, rolling Sharpe, and regression-based alpha test.

## Contents
- `src/Momentum.py` — strategy construction + backtest + plots + regressions
- `data/Momentum strategyy-2.xlsx` — input data
- `figures/` — output charts
- `paper/` — report (PDF)

## Strategies tested
- 1M momentum (long winner / short loser, hold 1 month)
- 12M momentum
- 12–1 momentum (skip most recent month)

## How to run
```bash
pip install -r requirements.txt
python src/Momentum.py
