# Static Dislocation Algorithm

This is the complete runbook for the strategy implemented in
`curve_dislocation_backtest.ipynb` and analyzed in `analytics.ipynb`.

## 1) Strategy objective

Trade calendar spreads within a single event when one deadline node deviates
from a lagged fair-value curve built on the event term structure.

- If node residual is negative (cheap): buy node, short hedge basket.
- If node residual is positive (rich): sell node, buy hedge basket.

## 2) Data used

- Events + markets: `https://gamma-api.polymarket.com/events`
- Historical prices: `https://clob.polymarket.com/prices-history`
- Live execution (systematic): order book, order placement, fills, positions,
  and fee/account endpoints from Polymarket CLOB.

## 3) Signal construction

For each event:

1. Build hourly panel across deadline nodes (`tau_days` per node).
2. At timestamp `t`, fit robust polynomial in logit space using only lagged
   data (`t-1` and earlier, optionally smoothed).
3. Predict current node prices and compute raw residuals.
4. Cross-sectionally demean residuals so parallel curve shifts do not trigger.
5. Entry trigger: `|ts_residual| >= STATIC_THRESHOLD`.

Bias controls:

- 1-bar delayed entry and exit execution.
- Poor-fit event filter can be computed on warmup window only (no lookahead).
- Hedge leg timestamp matching is backward-only (no future peeking).

## 4) Hedge model

- Thin curve (`N <= poly_degree + 1`): equal-weight hedge `-1/(N-1)`.
- Overdetermined curve: hat-matrix hedge weights.
- Stabilization caps:
  - `MAX_WEIGHT_PER_LEG`
  - `MAX_GROSS_HEDGE`
  - fallback to equal-weight if exceeded.

## 5) Exit logic

Priority:

1. Signal exit when `|ts_residual| < EXIT_THRESHOLD` (executed next bar).
2. Resolution price if resolved.
3. Pending mark-to-market on latest sampled price.

## 6) Research vs live execution

The project is now intentionally split:

- `curve_dislocation_backtest.ipynb`: research, backtests, diagnostics.
- `live_execution.py`: fast background signal generation for live trading.
- `analytics.ipynb`: analysis and calibration.

## 7) Notebook execution steps

Run `curve_dislocation_backtest.ipynb` top to bottom:

1. Imports + parameters.
2. Universe build (`include_closed=True`).
3. History panel build.
4. Static signal scoring + optional warmup-only poor-fit exclusion.
5. Trade construction.
6. Dedup for symmetric 2-node entries.
7. Results + spread-cost stress section.

Use `analytics.ipynb` for diagnostics, event-level behavior, and threshold
calibration.

## 8) Fast live runner (no backtest rerun)

Run one cycle:

`python3 live_execution.py`

Run continuously in background loop (example: every 5 minutes):

`python3 live_execution.py --loop-seconds 300`

Run with live order placement:

`python3 live_execution.py --execute-live --loop-seconds 300`

Speed/computational choices in the live runner:

- Universe cached and refreshed only every `UNIVERSE_REFRESH_MINUTES`.
- Only recent history is fetched (`LOOKBACK_HOURS`), not full backtest range.
- Parallel token history fetch via thread pool.
- Signals evaluated only on latest timestamp per event.
- Outputs saved to `execution_candidates_latest.csv/json`.
- Live order attempts saved to `execution_attempts_latest.csv` and `execution_log.jsonl`.

Setup once:

1. `python3 -m pip install -r requirements.txt`
2. `cp .env.example .env`
3. Fill `.env` values.

## 9) Requirements for systematic real-money trading

You need:

- A funded wallet authorized for trading.
- Signing private key / signer setup used by your CLOB client.
- CLOB API/session credentials required by your chosen client library.
- Reliable market metadata mapping (event, market, token ids, deadline).
- Live order books, order entry, cancel/replace, fills, positions.
- Fee schedule and account balance endpoints for risk checks.
- Persistent logging + monitoring (orders, fills, rejects, pnl, slippage).

Important constraint from Polymarket auth model:

- Posting orders requires signed payloads, so a wallet private key is required.
- API key/secret/passphrase alone is not sufficient to create signed orders.
- The live runner can derive API creds from the private key if they are not set.

## 10) Conservative liquidity-based sizing (1 cent from top of book)

Execution rule requested:

- Buy legs: only consume asks with `price <= best_ask + 0.01`.
- Sell legs: only consume bids with `price >= best_bid - 0.01`.
- You may consume all liquidity in that 1-cent band.

Spread sizing method:

1. Let `L_dis` be dislocated leg executable shares in the 1-cent band.
2. For hedge leg `i` with weight `w_i`, required shares are `|w_i| * q_dis`.
3. Let `L_i` be executable shares for hedge leg `i` in its 1-cent band.
4. Enforce feasibility:
   - `q_dis <= L_dis`
   - `q_dis <= L_i / |w_i|` for every hedge leg with nonzero weight.
5. Final size:
   - `q_dis = min(L_dis, min_i(L_i/|w_i|), portfolio caps)`

This guarantees each hedge leg can be completed without reaching beyond
1 cent from top of book.

## 11) Production risk controls (minimum)

- Max notional per spread and per event.
- Max daily loss and circuit breaker.
- Stale order book timeout (do not trade stale books).
- Min fill ratio for multi-leg execution (abort if one leg under-fills).
- Reject trading when spread width exceeds allowed threshold.
- Separate realized (`E`) and pending MtM (`P`) performance reporting.

## 12) Practical notes

- Backtest universe is built from current API snapshots (active/closed), not a
  historical as-of universe feed. Treat this as a structural limitation.
- Always evaluate robustness under nonzero spread assumptions (`SPREAD_HALF`).
- Favor exited-only and closed-only metrics over all-trades headline stats.
