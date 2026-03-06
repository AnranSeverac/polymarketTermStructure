from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    OrderType,
    PostOrdersArgs,
)
from py_clob_client.order_builder.constants import BUY, SELL

from curve_pipeline import (
    build_deadline_market_universe,
    compute_hedge_weights,
    event_ids_poor_static_fit_warmup,
    fetch_token_price_history,
    score_time_shifted_dislocations,
)


# -----------------------------
# Fast live-run configuration
# -----------------------------
MAX_EVENTS = 1200
MAX_MARKETS = 600
INCLUDE_CLOSED = True
UNIVERSE_REFRESH_MINUTES = 120

INTERVAL = "1h"
FREQUENCY_MINUTES = 60
LOOKBACK_HOURS = 96
FETCH_WORKERS = 16

STATIC_POLY_DEGREE = 2
STATIC_MIN_NODES = 2
STATIC_LAG = 1
REF_SMOOTH_BARS = 10
STATIC_THRESHOLD = 0.12

EXCLUDE_POOR_FIT = True
POOR_FIT_MIN_OBS = 50
POOR_FIT_EXCLUDE_WORST_PCT = 10.0
POOR_FIT_WARMUP_FRAC = 0.25

MAX_WEIGHT_PER_LEG = 1.5
MAX_GROSS_HEDGE = 5.0

CACHE_DIR = Path(".cache")
UNIVERSE_CACHE_PATH = CACHE_DIR / "universe.parquet"
LOG_DIR = Path("logs")

# Track starting balance for PnL (since process start) when running live.
_start_balance: Optional[float] = None
EXECUTION_LOG_PATH = LOG_DIR / "execution_log.jsonl"
CYCLE_LOG_PATH = LOG_DIR / "cycle_log.jsonl"
EXECUTION_ATTEMPTS_PATH = LOG_DIR / "execution_attempts_latest.csv"

# Live execution controls
MAX_DISLOCATED_SHARES = 500.0
MIN_EXECUTABLE_SHARES = 1.0
MAX_FROM_TOP = 0.01
HEARTBEAT_EVERY_N_RUNS = 3

# Exit: assume we exit when |residual| drops to this (matches backtest EXIT_THRESHOLD).
EXIT_THRESHOLD = 0.03


def top_of_book_liquidity_within_1c(
    levels: Iterable[Tuple[float, float]],
    side: str,
    max_from_top: float = 0.01,
) -> float:
    """Shares executable within 1 cent from top-of-book."""
    lv = [(float(p), float(s)) for p, s in levels if float(s) > 0]
    if not lv:
        return 0.0
    side = side.lower()
    best = lv[0][0]
    if side == "buy":
        return float(sum(sz for px, sz in lv if px <= (best + max_from_top)))
    if side == "sell":
        return float(sum(sz for px, sz in lv if px >= (best - max_from_top)))
    raise ValueError("side must be 'buy' or 'sell'")


def conservative_spread_size(
    dislocated_liq: float,
    hedge_liq_by_deadline: Dict[object, float],
    hedge_weights_by_deadline: Dict[object, float],
    max_dislocated_shares: float,
) -> float:
    """Feasible dislocated-leg shares under hedge liquidity constraints."""
    caps = [float(dislocated_liq), float(max_dislocated_shares)]
    for dd, w in hedge_weights_by_deadline.items():
        req = abs(float(w))
        if req <= 1e-12:
            continue
        liq = float(hedge_liq_by_deadline.get(dd, 0.0))
        caps.append(liq / req)
    return max(0.0, float(min(caps)))


def _book_levels(book: object, side: str) -> List[Tuple[float, float]]:
    levels = getattr(book, "asks" if side == "buy" else "bids", []) or []
    out: List[Tuple[float, float]] = []
    for lvl in levels:
        px = getattr(lvl, "price", None)
        sz = getattr(lvl, "size", None)
        if px is None and isinstance(lvl, dict):
            px = lvl.get("price")
            sz = lvl.get("size")
        if px is None or sz is None:
            continue
        out.append((float(px), float(sz)))
    return out


def _cap_price_from_book(book: object, side: str, max_from_top: float) -> Optional[float]:
    levels = _book_levels(book, side)
    if not levels:
        return None
    best = float(levels[0][0])
    if side == "buy":
        return min(0.9999, best + max_from_top)
    return max(0.0001, best - max_from_top)


def _shares_to_amount(side: str, shares: float, cap_price: float) -> float:
    # For market BUY orders, amount is dollars. For SELL orders, amount is shares.
    if side == BUY:
        return float(shares * cap_price)
    return float(shares)


class PolymarketExecutor:
    def __init__(self) -> None:
        load_dotenv()
        self.host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
        self.chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
        self.funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "").strip()
        self.signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))

        if not self.private_key:
            raise RuntimeError("Missing POLYMARKET_PRIVATE_KEY in environment.")
        if not self.funder:
            raise RuntimeError("Missing POLYMARKET_FUNDER_ADDRESS in environment.")

        self.client = ClobClient(
            host=self.host,
            chain_id=self.chain_id,
            key=self.private_key,
            signature_type=self.signature_type,
            funder=self.funder,
        )

        api_key = os.getenv("POLYMARKET_API_KEY", "").strip()
        api_secret = os.getenv("POLYMARKET_API_SECRET", "").strip()
        api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "").strip()
        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
            self.client.set_api_creds(creds)
            self.api_creds = creds
        else:
            self.api_creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(self.api_creds)
            print("[auth] Derived API creds from private key.", flush=True)

        self._tick_cache: Dict[str, str] = {}
        self._neg_risk_cache: Dict[str, bool] = {}
        self._heartbeat_id: str = ""
        self._runs_since_heartbeat = 0

    def get_tick_size(self, token_id: str) -> str:
        if token_id not in self._tick_cache:
            self._tick_cache[token_id] = str(self.client.get_tick_size(token_id))
        return self._tick_cache[token_id]

    def get_neg_risk(self, token_id: str) -> bool:
        if token_id not in self._neg_risk_cache:
            self._neg_risk_cache[token_id] = bool(self.client.get_neg_risk(token_id))
        return self._neg_risk_cache[token_id]

    def get_order_book(self, token_id: str):
        return self.client.get_order_book(token_id)

    def get_balance(self) -> Dict[str, object]:
        """Return USDC (collateral) balance info for logging."""
        try:
            bal = self.client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            if hasattr(bal, "__dict__"):
                return {k: str(v) if hasattr(v, "isoformat") else v for k, v in bal.__dict__.items()}
            return {"raw": str(bal)}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def maybe_heartbeat(self) -> None:
        self._runs_since_heartbeat += 1
        if self._runs_since_heartbeat < HEARTBEAT_EVERY_N_RUNS:
            return
        resp = self.client.post_heartbeat(self._heartbeat_id or "")
        self._heartbeat_id = resp.get("heartbeat_id", self._heartbeat_id)
        self._runs_since_heartbeat = 0

    def post_market_orders_batch(self, legs: List[dict]) -> list:
        signed: List[PostOrdersArgs] = []
        for leg in legs:
            token_id = str(leg["token_id"])
            side = str(leg["side"])
            shares = float(leg["shares"])
            cap_price = float(leg["cap_price"])
            amount = _shares_to_amount(side, shares, cap_price)
            if amount <= 0:
                continue

            order = self.client.create_market_order(
                MarketOrderArgs(
                    token_id=token_id,
                    amount=amount,
                    side=side,
                    price=cap_price,
                    order_type=OrderType.FOK,
                ),
                options={
                    "tick_size": self.get_tick_size(token_id),
                    "neg_risk": self.get_neg_risk(token_id),
                },
            )
            signed.append(PostOrdersArgs(order=order, orderType=OrderType.FOK, postOnly=False))
        if not signed:
            return []
        return self.client.post_orders(signed)


def load_or_refresh_universe(now_utc: pd.Timestamp) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached: Optional[pd.DataFrame] = None
    if UNIVERSE_CACHE_PATH.exists():
        mtime = pd.Timestamp(UNIVERSE_CACHE_PATH.stat().st_mtime, unit="s", tz="UTC")
        age_min = (now_utc - mtime).total_seconds() / 60.0
        cached = pd.read_parquet(UNIVERSE_CACHE_PATH)
        if age_min <= UNIVERSE_REFRESH_MINUTES and cached is not None and not cached.empty:
            return cached
    try:
        u = build_deadline_market_universe(
            max_events=MAX_EVENTS,
            min_distinct_dates=2,
            include_closed=INCLUDE_CLOSED,
        )
    except Exception:  # noqa: BLE001
        if cached is not None and not cached.empty:
            return cached
        raise
    if MAX_MARKETS is not None and len(u) > MAX_MARKETS:
        u = u.head(MAX_MARKETS).copy()
    u.to_parquet(UNIVERSE_CACHE_PATH, index=False)
    return u


def _fetch_recent_token_panel(
    row: pd.Series,
    start_ts: int,
    end_ts: int,
    min_ts: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    try:
        hist = fetch_token_price_history(
            token_id=row["yes_token_id"],
            start_ts=start_ts,
            end_ts=end_ts,
            interval=INTERVAL,
            fidelity=FREQUENCY_MINUTES,
        )
    except Exception:  # noqa: BLE001
        return None
    if hist.empty:
        return None
    hist = hist[hist["timestamp"] >= min_ts].copy()
    if hist.empty:
        return None

    hist = (
        hist.set_index("timestamp")
        .resample(f"{int(FREQUENCY_MINUTES)}min")
        .last()
        .dropna()
        .reset_index()
    )
    if hist.empty:
        return None

    hist["event_id"] = row["event_id"]
    hist["question"] = row["question"]
    hist["deadline_date"] = row["deadline_date"]
    hist["yes_token_id"] = row["yes_token_id"]
    return hist


def build_recent_panel(universe: pd.DataFrame, now_utc: pd.Timestamp) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "question",
                "deadline_date",
                "yes_token_id",
                "timestamp",
                "probability_yes",
                "tau_days",
            ]
        )

    end_ts = int(now_utc.timestamp())
    start_ts = int((now_utc - pd.Timedelta(hours=LOOKBACK_HOURS)).timestamp())
    min_ts = now_utc - pd.Timedelta(hours=LOOKBACK_HOURS)

    rows: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        futures = [
            ex.submit(_fetch_recent_token_panel, row, start_ts, end_ts, min_ts)
            for _, row in universe.iterrows()
        ]
        for fut in as_completed(futures):
            out = fut.result()
            if out is not None and not out.empty:
                rows.append(out)

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)
    panel["timestamp"] = panel["timestamp"].dt.floor(INTERVAL)
    panel = (
        panel.groupby(
            ["event_id", "question", "deadline_date", "yes_token_id", "timestamp"],
            as_index=False,
        )["probability_yes"]
        .last()
    )
    panel["deadline_date"] = pd.to_datetime(panel["deadline_date"]).dt.date
    panel["tau_days"] = (
        pd.to_datetime(panel["deadline_date"])
        - panel["timestamp"].dt.tz_convert(None).dt.normalize()
    ).dt.days.clip(lower=1)
    panel = panel.sort_values(["event_id", "timestamp", "deadline_date"]).reset_index(drop=True)
    return panel


def latest_signals(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    static_df = score_time_shifted_dislocations(
        panel,
        lag_bars=STATIC_LAG,
        min_nodes=STATIC_MIN_NODES,
        poly_degree=STATIC_POLY_DEGREE,
        ref_smooth_bars=REF_SMOOTH_BARS,
    ).dropna(subset=["ts_predicted_prob"])

    if static_df.empty:
        return static_df, static_df

    if EXCLUDE_POOR_FIT:
        poor_fit_ids = event_ids_poor_static_fit_warmup(
            static_df,
            warmup_frac=POOR_FIT_WARMUP_FRAC,
            min_obs_per_event=POOR_FIT_MIN_OBS,
            exclude_worst_pct=POOR_FIT_EXCLUDE_WORST_PCT,
        )
        static_df = static_df[~static_df["event_id"].astype(str).isin(poor_fit_ids)].copy()

    if static_df.empty:
        return static_df, static_df

    # Signals use CLOB price history (last-trade / API "p"), not top of book or mid.
    # Execution is at top of book within MAX_FROM_TOP (e.g. 1c); opportunity size uses same.
    latest_ts = static_df.groupby("event_id")["timestamp"].transform("max")
    live_slice = static_df[static_df["timestamp"] == latest_ts].copy()
    live_slice["direction"] = np.where(live_slice["ts_residual"] < 0, "BUY", "SELL")
    signals = live_slice[live_slice["ts_residual"].abs() >= STATIC_THRESHOLD].copy()
    signals = signals.sort_values("ts_residual", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return static_df, signals


def build_execution_candidates(signals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "question",
                "timestamp",
                "dis_node",
                "direction",
                "static_resid",
                "n_nodes",
                "dis_token_id",
                "hedge_weights_by_deadline",
                "hedge_weights_by_token",
            ]
        )

    event_deadlines = panel.groupby("event_id")["deadline_date"].apply(lambda x: sorted(x.unique())).to_dict()
    out: List[dict] = []

    for _, sig in signals.iterrows():
        eid = sig["event_id"]
        dd = sig["deadline_date"]
        ts = sig["timestamp"]
        direction = sig["direction"]

        deadlines = event_deadlines.get(eid, [])
        if dd not in deadlines:
            continue
        entry_snap = panel[(panel["event_id"] == eid) & (panel["timestamp"] == ts)]
        if entry_snap.empty:
            continue
        tau_map = entry_snap.groupby("deadline_date")["tau_days"].first()
        available = [(i, d) for i, d in enumerate(deadlines) if d in tau_map.index]
        if len(available) < 2:
            continue
        deadlines_local = [d for _, d in available]
        token_map = (
            entry_snap[["deadline_date", "yes_token_id"]]
            .drop_duplicates(subset=["deadline_date"], keep="last")
            .set_index("deadline_date")["yes_token_id"]
            .to_dict()
        )
        j_idx = next((k for k, (_, d) in enumerate(available) if d == dd), None)
        if j_idx is None:
            continue
        taus = np.asarray([tau_map[d] for _, d in available], dtype=float)

        hedge_idx_weights = compute_hedge_weights(
            j_idx,
            len(deadlines_local),
            taus,
            STATIC_POLY_DEGREE,
            max_weight_per_leg=MAX_WEIGHT_PER_LEG,
            max_gross_hedge=MAX_GROSS_HEDGE,
        )
        if not hedge_idx_weights:
            continue
        hedge_by_deadline = {str(deadlines_local[i]): float(w) for i, w in hedge_idx_weights.items()}
        hedge_by_token = {str(token_map.get(deadlines_local[i])): float(w) for i, w in hedge_idx_weights.items()}
        dis_token_id = token_map.get(dd)
        if dis_token_id is None:
            continue

        out.append(
            {
                "event_id": str(eid),
                "question": str(sig["question"]),
                "timestamp": ts,
                "dis_node": str(dd),
                "direction": direction,
                "static_resid": float(sig["ts_residual"]),
                "n_nodes": int(len(deadlines_local)),
                "dis_token_id": str(dis_token_id),
                "hedge_weights_by_deadline": hedge_by_deadline,
                "hedge_weights_by_token": hedge_by_token,
            }
        )

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("static_resid", key=lambda s: s.abs(), ascending=False)


def _json_safe(obj: object) -> object:
    """Convert payload to JSON-serializable form (e.g. Timestamp -> iso string)."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _log_execution(payload: dict) -> None:
    payload = {"ts": pd.Timestamp.utcnow().isoformat(), **payload}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with EXECUTION_LOG_PATH.open("a") as f:
        f.write(json.dumps(_json_safe(payload)) + "\n")


def compute_opportunity_per_candidate(
    candidates: pd.DataFrame,
    executor: PolymarketExecutor,
) -> List[Dict[str, object]]:
    """For each candidate: max tradeable size (within 1c of top of book), entry price, and
    estimated gross $ opportunity assuming exit when |residual| drops to EXIT_THRESHOLD.
    Returns one dict per candidate (same order): max_shares, entry_price_dis, est_gross_dollars.
    """
    out: List[Dict[str, object]] = []
    for _, cand in candidates.iterrows():
        rec = {"max_shares": 0.0, "entry_price_dis": None, "est_gross_dollars": 0.0}
        try:
            dis_token_id = str(cand["dis_token_id"])
            direction = str(cand["direction"]).upper()
            dis_side = BUY if direction == "BUY" else SELL
            resid = abs(float(cand["static_resid"]))

            dis_book = executor.get_order_book(dis_token_id)
            dis_levels = _book_levels(dis_book, "buy" if dis_side == BUY else "sell")
            dis_liq = top_of_book_liquidity_within_1c(
                dis_levels,
                "buy" if dis_side == BUY else "sell",
                max_from_top=MAX_FROM_TOP,
            )

            hedge_by_token_raw = cand["hedge_weights_by_token"]
            if isinstance(hedge_by_token_raw, str):
                hedge_by_token = json.loads(hedge_by_token_raw)
            else:
                hedge_by_token = dict(hedge_by_token_raw)

            hedge_liq: Dict[str, float] = {}
            hedge_abs_w: Dict[str, float] = {}
            for token_id, w in hedge_by_token.items():
                w = float(w)
                position_sign = w if direction == "BUY" else -w
                side = BUY if position_sign > 0 else SELL
                book = executor.get_order_book(str(token_id))
                levels = _book_levels(book, "buy" if side == BUY else "sell")
                liq = top_of_book_liquidity_within_1c(
                    levels,
                    "buy" if side == BUY else "sell",
                    max_from_top=MAX_FROM_TOP,
                )
                cap = _cap_price_from_book(book, "buy" if side == BUY else "sell", MAX_FROM_TOP)
                if cap is None:
                    liq = 0.0
                hedge_liq[str(token_id)] = liq
                hedge_abs_w[str(token_id)] = abs(w)

            q_dis = conservative_spread_size(
                dislocated_liq=dis_liq,
                hedge_liq_by_deadline=hedge_liq,
                hedge_weights_by_deadline=hedge_abs_w,
                max_dislocated_shares=MAX_DISLOCATED_SHARES,
            )
            dis_cap = _cap_price_from_book(
                dis_book,
                "buy" if dis_side == BUY else "sell",
                MAX_FROM_TOP,
            )
            rec["max_shares"] = float(q_dis)
            rec["entry_price_dis"] = float(dis_cap) if dis_cap is not None else None
            # Gross $: assume exit when |residual| = EXIT_THRESHOLD; profit per share ≈ |resid| - EXIT_THRESHOLD (prob space ≈ $).
            rec["est_gross_dollars"] = max(0.0, (resid - EXIT_THRESHOLD) * q_dis)
        except Exception:  # noqa: BLE001
            pass
        out.append(rec)
    return out

def execute_candidates(candidates: pd.DataFrame, executor: PolymarketExecutor) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["event_id", "dis_token_id", "executed_shares", "status", "details"])

    rows: List[dict] = []
    for _, cand in candidates.iterrows():
        try:
            dis_token_id = str(cand["dis_token_id"])
            direction = str(cand["direction"]).upper()
            dis_side = BUY if direction == "BUY" else SELL
            dis_book = executor.get_order_book(dis_token_id)
            dis_levels = _book_levels(dis_book, "buy" if dis_side == BUY else "sell")
            dis_liq = top_of_book_liquidity_within_1c(
                dis_levels,
                "buy" if dis_side == BUY else "sell",
                max_from_top=MAX_FROM_TOP,
            )

            hedge_by_token_raw = cand["hedge_weights_by_token"]
            if isinstance(hedge_by_token_raw, str):
                hedge_by_token = json.loads(hedge_by_token_raw)
            else:
                hedge_by_token = dict(hedge_by_token_raw)

            hedge_liq: Dict[str, float] = {}
            hedge_abs_w: Dict[str, float] = {}
            hedge_exec_side: Dict[str, str] = {}
            hedge_cap_price: Dict[str, float] = {}
            for token_id, w in hedge_by_token.items():
                w = float(w)
                # Direction flip to mirror backtest hedge PnL conventions.
                position_sign = w if direction == "BUY" else -w
                side = BUY if position_sign > 0 else SELL
                book = executor.get_order_book(str(token_id))
                levels = _book_levels(book, "buy" if side == BUY else "sell")
                liq = top_of_book_liquidity_within_1c(
                    levels,
                    "buy" if side == BUY else "sell",
                    max_from_top=MAX_FROM_TOP,
                )
                cap_price = _cap_price_from_book(book, "buy" if side == BUY else "sell", MAX_FROM_TOP)
                if cap_price is None:
                    liq = 0.0
                hedge_liq[str(token_id)] = liq
                hedge_abs_w[str(token_id)] = abs(w)
                hedge_exec_side[str(token_id)] = side
                if cap_price is not None:
                    hedge_cap_price[str(token_id)] = cap_price

            q_dis = conservative_spread_size(
                dislocated_liq=dis_liq,
                hedge_liq_by_deadline=hedge_liq,
                hedge_weights_by_deadline=hedge_abs_w,
                max_dislocated_shares=MAX_DISLOCATED_SHARES,
            )
            if q_dis < MIN_EXECUTABLE_SHARES:
                rows.append(
                    {
                        "event_id": str(cand["event_id"]),
                        "dis_token_id": dis_token_id,
                        "executed_shares": 0.0,
                        "status": "SKIP_NO_SIZE",
                        "details": f"q_dis={q_dis:.4f}",
                    }
                )
                continue

            dis_cap = _cap_price_from_book(
                dis_book,
                "buy" if dis_side == BUY else "sell",
                MAX_FROM_TOP,
            )
            if dis_cap is None:
                rows.append(
                    {
                        "event_id": str(cand["event_id"]),
                        "dis_token_id": dis_token_id,
                        "executed_shares": 0.0,
                        "status": "SKIP_NO_BOOK",
                        "details": "dislocated book empty",
                    }
                )
                continue

            legs = [
                {"token_id": dis_token_id, "side": dis_side, "shares": q_dis, "cap_price": dis_cap}
            ]
            for token_id, abs_w in hedge_abs_w.items():
                cap = hedge_cap_price.get(token_id)
                if cap is None:
                    continue
                legs.append(
                    {
                        "token_id": token_id,
                        "side": hedge_exec_side[token_id],
                        "shares": q_dis * float(abs_w),
                        "cap_price": cap,
                    }
                )

            responses = executor.post_market_orders_batch(legs)
            rows.append(
                {
                    "event_id": str(cand["event_id"]),
                    "dis_token_id": dis_token_id,
                    "executed_shares": float(q_dis),
                    "status": "SENT",
                    "details": json.dumps(responses),
                }
            )
            _log_execution(
                {
                    "event_id": str(cand["event_id"]),
                    "direction": direction,
                    "dis_token_id": dis_token_id,
                    "q_dis": q_dis,
                    "legs": legs,
                    "response": responses,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "event_id": str(cand.get("event_id", "")),
                    "dis_token_id": str(cand.get("dis_token_id", "")),
                    "executed_shares": 0.0,
                    "status": "ERROR",
                    "details": str(exc),
                }
            )
            _log_execution({"status": "ERROR", "error": str(exc), "candidate": cand.to_dict()})

    return pd.DataFrame(rows)


def run_once(execute_live: bool = False) -> pd.DataFrame:
    t0 = time.time()
    executor: Optional[PolymarketExecutor] = PolymarketExecutor() if execute_live else None
    now_utc = pd.Timestamp.utcnow()
    universe = load_or_refresh_universe(now_utc)
    panel = build_recent_panel(universe, now_utc)
    static_df, signals = latest_signals(panel)
    candidates = build_execution_candidates(signals, panel)

    # Candidate output is the printed table below; no CSV/JSON files by default.
    executed = pd.DataFrame()
    balance: Dict[str, object] = {}
    first_book: Dict[str, object] = {}
    opportunity_per_candidate: List[Dict[str, object]] = []
    if execute_live and executor is not None:
        try:
            balance = executor.get_balance()
        except Exception:  # noqa: BLE001
            balance = {"error": "balance_fetch_failed"}
        if not candidates.empty:
            opportunity_per_candidate = compute_opportunity_per_candidate(candidates, executor)
            executed = execute_candidates(candidates, executor)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            executed.to_csv(EXECUTION_ATTEMPTS_PATH, index=False)
            # Snapshot top-of-book for first candidate (dislocated leg only)
            try:
                row = candidates.iloc[0]
                tid = str(row["dis_token_id"])
                direction = str(row["direction"]).upper()
                side = "buy" if direction == "BUY" else "sell"
                book = executor.get_order_book(tid)
                ask_levels = _book_levels(book, "buy")
                bid_levels = _book_levels(book, "sell")
                best_ask = float(ask_levels[0][0]) if ask_levels else None
                best_bid = float(bid_levels[0][0]) if bid_levels else None
                liq = top_of_book_liquidity_within_1c(
                    ask_levels if side == "buy" else bid_levels,
                    side,
                    max_from_top=MAX_FROM_TOP,
                )
                first_book = {
                    "event_id": str(row["event_id"]),
                    "dis_node": str(row["dis_node"]),
                    "dis_token_id": tid,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "liq_1c_trade_side": liq,
                }
            except Exception:  # noqa: BLE001
                first_book = {"error": "book_snapshot_failed"}
        executor.maybe_heartbeat()

    elapsed = time.time() - t0
    cycle_entry = {
        "ts": now_utc.isoformat(),
        "n_universe": len(universe),
        "n_panel": len(panel),
        "n_signals": len(signals),
        "n_candidates": len(candidates),
        "n_executed": len(executed),
        "elapsed_s": round(elapsed, 2),
        "balance": balance,
        "first_candidate_book": first_book if first_book else None,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with CYCLE_LOG_PATH.open("a") as f:
        f.write(json.dumps(_json_safe(cycle_entry)) + "\n")

    # --- Terminal output: algo summary every run (flush so it shows in IDE/terminal) ---
    def _out(msg: str = "") -> None:
        print(msg, flush=True)
    _out()
    _out(f"--- cycle {now_utc.isoformat()} ---")
    _out(f"  Loop time: {elapsed:.2f}s")
    _out(
        f"  universe={len(universe)}  panel_rows={len(panel)}  "
        f"signals={len(signals)}  candidates={len(candidates)}"
    )
    if not candidates.empty:
        _out("  candidates (algo output):")
        for i, (_, row) in enumerate(candidates.head(15).iterrows(), 1):
            q = str(row.get("question", ""))[:55] + ("..." if len(str(row.get("question", ""))) > 55 else "")
            direction = str(row["direction"]).upper()
            dis_node = str(row["dis_node"])
            resid = abs(float(row["static_resid"]))
            _out(f"    {i}. {q}  |resid|={resid:.4f}  nodes={int(row['n_nodes'])}")
            _out(f"        Trade: {direction} at node (deadline) {dis_node} (dislocated leg, 1 unit)")
            hw = row.get("hedge_weights_by_deadline")
            if hw is not None and isinstance(hw, dict):
                parts = [f"{d}: w={w:+.3f}" for d, w in sorted(hw.items())]
                _out(f"        Hedge weights by deadline: {', '.join(parts)}")
            elif hw is not None and isinstance(hw, str):
                try:
                    hwd = json.loads(hw)
                    parts = [f"{d}: w={float(w):+.3f}" for d, w in sorted(hwd.items())]
                    _out(f"        Hedge weights by deadline: {', '.join(parts)}")
                except (json.JSONDecodeError, TypeError):
                    _out(f"        Hedge: {hw}")
            # Dollar opportunity: size within 1c of top of book, exit at EXIT_THRESHOLD
            if i <= len(opportunity_per_candidate):
                opp = opportunity_per_candidate[i - 1]
                max_sh = opp.get("max_shares", 0.0)
                entry_p = opp.get("entry_price_dis")
                gross = opp.get("est_gross_dollars", 0.0)
                if entry_p is not None:
                    _out(f"        Opportunity: max_shares={max_sh:.1f} (within 1c of top)  entry_dis={entry_p:.3f}  est_gross_dollars=${gross:.2f} (exit when |resid|<{EXIT_THRESHOLD})")
                else:
                    _out(f"        Opportunity: max_shares={max_sh:.1f}  est_gross_dollars=${gross:.2f} (exit when |resid|<{EXIT_THRESHOLD})")
            else:
                _out("        Opportunity: N/A (run with --execute-live for tradeable size)")
        if len(candidates) > 15:
            _out(f"    ... and {len(candidates) - 15} more")
    else:
        _out("  candidates (algo output): none")
    if not executed.empty:
        _out("  execution:")
        for _, row in executed.iterrows():
            details = str(row.get("details", ""))[:80]
            if len(str(row.get("details", ""))) > 80:
                details += "..."
            _out(
                f"    {str(row['event_id'])[:12]}... {row['status']} "
                f"shares={float(row['executed_shares']):.2f}  {details}"
            )
    if balance:
        _out(f"  Account balance: {balance}")
        # PnL since process start (when we have numeric balance)
        current = None
        if isinstance(balance, dict) and "error" not in balance:
            for key in ("balance", "amount", "size"):
                raw = balance.get(key)
                if raw is not None:
                    try:
                        current = float(raw)
                        break
                    except (TypeError, ValueError):
                        continue
        if current is not None:
            global _start_balance
            if _start_balance is None:
                _start_balance = current
            pnl = current - _start_balance
            _out(f"  Strategy PnL (since process start): ${pnl:+.2f} USDC")
    _out(f"  logs: {CYCLE_LOG_PATH} | {EXECUTION_LOG_PATH}" + (f" | {EXECUTION_ATTEMPTS_PATH}" if not executed.empty else ""))
    _out()
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast live signal runner (no backtest rerun).")
    parser.add_argument("--loop-seconds", type=int, default=300, help="Run continuously every N seconds. Use 0 to run once and exit.")
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="Send live FOK marketable orders for generated candidates.",
    )
    args = parser.parse_args()

    if args.loop_seconds <= 0:
        run_once(execute_live=args.execute_live)
        return

    while True:
        try:
            run_once(execute_live=args.execute_live)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_once] error: {exc}", flush=True)
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    main()

