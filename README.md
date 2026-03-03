# Yield curve dislocation (Polymarket)

Calendar-spread strategy: static dislocation signal, backtest + live execution.

## Essential files

| File | Purpose |
|------|--------|
| `curve_pipeline.py` | Core algorithm: universe, panel, signal, hedge weights, trade builder. |
| `live_execution.py` | Fast live runner (signals + optional order placement). No backtest rerun. |
| `curve_dislocation_backtest.ipynb` | Backtest: config, universe, panel, trades, dedup, summary. |
| `analytics.ipynb` | Analytics, diagnostics, calibration. |
| `ALGORITHM.md` | Full algorithm and runbook. |
| `requirements.txt` | Python deps. |
| `.env.example` | Env template; copy to `.env` and fill for live trading. |

## Run

- **Backtest:** Open `curve_dislocation_backtest.ipynb`, run all cells.
- **Analytics:** Open `analytics.ipynb`, run as needed.
- **Live (one shot):** `python3 live_execution.py`
- **Live (loop):** `python3 live_execution.py --execute-live --loop-seconds 300`

Generated at runtime (gitignored): `.cache/`, `*_latest.csv`, `*_latest.json`, `*_log.jsonl`, `execution_attempts_latest.csv`.
