from __future__ import annotations

import datetime as dt
import json
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
CLOB_PRICES_HISTORY_URL = "https://clob.polymarket.com/prices-history"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad)


def _huber_weights(r: np.ndarray, c: float = 1.5) -> np.ndarray:
    scale = _mad(r)
    if scale < 1e-10:
        return np.ones_like(r)
    z = np.abs(r) / scale
    w = np.ones_like(r)
    mask = z > c
    w[mask] = c / z[mask]
    return w


def _robust_polyfit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    max_iter: int = 20,
    tol: float = 1e-7,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size <= degree:
        return np.polyfit(x, y, deg=min(1, max(0, x.size - 1)))

    coeff = np.polyfit(x, y, deg=degree)
    for _ in range(max_iter):
        pred = np.polyval(coeff, x)
        resid = y - pred
        w = _huber_weights(resid)
        coeff_new = np.polyfit(x, y, deg=degree, w=w)
        if np.max(np.abs(coeff_new - coeff)) < tol:
            coeff = coeff_new
            break
        coeff = coeff_new
    return coeff


# ── Date parsing ──────────────────────────────────────────────

def _parse_datetime_maybe(date_str: str) -> Optional[dt.datetime]:
    if not date_str:
        return None
    try:
        return dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_DATE_PATTERNS = [
    re.compile(
        r"\b(?:by|before|in)\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:by|before)\s+([A-Za-z]+)\s+(\d{1,2})\b(?!\s*,?\s*\d{4})",
        re.IGNORECASE,
    ),
    re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"\bbefore\s+(\d{4})\b", re.IGNORECASE),
]


def _parse_deadline_from_question(question: str) -> Optional[dt.date]:
    if not question:
        return None

    m = _DATE_PATTERNS[0].search(question)
    if m:
        month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
        month = _MONTH_MAP.get(month_str.lower())
        if month:
            try:
                return dt.date(int(year_str), month, int(day_str))
            except ValueError:
                pass

    m = _DATE_PATTERNS[1].search(question)
    if m:
        month_str, day_str = m.group(1), m.group(2)
        month = _MONTH_MAP.get(month_str.lower())
        if month:
            now = dt.date.today()
            candidate = dt.date(now.year, month, int(day_str))
            if candidate < now:
                candidate = dt.date(now.year + 1, month, int(day_str))
            return candidate

    m = _DATE_PATTERNS[2].search(question)
    if m:
        return dt.date(int(m.group(1)), 12, 31)

    m = _DATE_PATTERNS[3].search(question)
    if m:
        return dt.date(int(m.group(1)) - 1, 12, 31)

    return None


def _extract_yes_token_id(market: dict) -> Optional[str]:
    outcomes_raw = market.get("outcomes")
    token_ids_raw = market.get("clobTokenIds")
    if outcomes_raw is None or token_ids_raw is None:
        return None
    try:
        outcomes = outcomes_raw if isinstance(outcomes_raw, list) else json.loads(outcomes_raw)
        token_ids = token_ids_raw if isinstance(token_ids_raw, list) else json.loads(token_ids_raw)
    except Exception:
        return None
    if len(outcomes) != len(token_ids):
        return None
    for i, outcome in enumerate(outcomes):
        if str(outcome).strip().lower() == "yes":
            return str(token_ids[i])
    return None


# ── Universe construction ─────────────────────────────────────

def fetch_events(max_events: int = 1200, active: bool = True, closed: bool = False) -> List[dict]:
    events: List[dict] = []
    offset = 0
    page_size = 200
    while len(events) < max_events:
        limit = min(page_size, max_events - len(events))
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(GAMMA_EVENTS_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        events.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return events


def build_deadline_market_universe(
    max_events: int = 1200,
    min_distinct_dates: int = 2,
    include_closed: bool = False,
) -> pd.DataFrame:
    title_has_by = lambda s: isinstance(s, str) and ("by" in s.lower())
    market_deadline_like = lambda s: isinstance(s, str) and any(
        k in s.lower() for k in [" by ", "before", " in "]
    )

    events = fetch_events(max_events=max_events, active=True, closed=False)
    if include_closed:
        events.extend(fetch_events(max_events=max_events, active=False, closed=True))

    rows: List[dict] = []
    for event in events:
        event_id = event.get("id")
        event_title = event.get("title", "")
        event_slug = event.get("slug", "")
        if not title_has_by(event_title):
            continue
        for market in event.get("markets", []):
            question = market.get("question", "")
            if not market_deadline_like(question):
                continue
            deadline = _parse_deadline_from_question(question)
            if deadline is None:
                parsed_end = _parse_datetime_maybe(market.get("endDate", ""))
                if parsed_end is not None:
                    deadline = parsed_end.date() if isinstance(parsed_end, dt.datetime) else parsed_end
            yes_token = _extract_yes_token_id(market)
            if deadline is None or yes_token is None:
                continue

            outcome_prices = market.get("outcomePrices")
            resolution = None
            if outcome_prices:
                try:
                    prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                    if prices and len(prices) >= 1:
                        r = float(prices[0])
                        if r in (0.0, 1.0):
                            resolution = r
                except Exception:
                    pass

            rows.append({
                "event_id": event_id,
                "event_slug": event_slug,
                "question": event_title,
                "market_id": market.get("id"),
                "market_question": question,
                "deadline_date": deadline,
                "yes_token_id": yes_token,
                "resolution": resolution,
            })

    universe = pd.DataFrame(rows)
    if universe.empty:
        return universe

    # When multiple markets map to the same (event_id, deadline_date), keep the *last* so we
    # tend to get the more specific market (e.g. "by December 31, 2025") rather than a
    # vaguer one (e.g. "by end of 2025") that may have wrong/stale CLOB data.
    universe = universe.drop_duplicates(subset=["event_id", "deadline_date"], keep="last")

    distinct_counts = (
        universe.groupby(["event_id", "question"], dropna=False)["deadline_date"]
        .nunique()
        .reset_index(name="num_distinct_dates")
    )
    keep = distinct_counts[distinct_counts["num_distinct_dates"] >= min_distinct_dates]
    universe = universe.merge(keep[["event_id", "question"]], on=["event_id", "question"], how="inner")
    universe = universe.sort_values(["event_id", "deadline_date"]).reset_index(drop=True)
    return universe


# ── Price history ─────────────────────────────────────────────

def fetch_token_price_history(
    token_id: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    interval: str = "1h",
    fidelity: int = 60,
) -> pd.DataFrame:
    """Fetch price history for a single token.

    When start_ts/end_ts are provided AND fidelity > 0, uses the CLOB
    fidelity endpoint (minute-level, ~1 month max).

    When fidelity == 0 (or start_ts/end_ts are None), uses the interval
    endpoint which returns longer history at coarser granularity.
    """
    params: Dict[str, object] = {"market": str(token_id)}
    if start_ts is not None and end_ts is not None and fidelity > 0:
        params["startTs"] = int(start_ts)
        params["endTs"] = int(end_ts)
        params["fidelity"] = int(fidelity)
    else:
        # Always fetch full history; caller resamples to desired frequency
        params["interval"] = "max"

    resp = requests.get(CLOB_PRICES_HISTORY_URL, params=params, timeout=30)
    if resp.status_code >= 400:
        fallback_params = {"market": str(token_id), "interval": "max"}
        resp = requests.get(CLOB_PRICES_HISTORY_URL, params=fallback_params, timeout=30)
        if resp.status_code >= 400:
            return pd.DataFrame(columns=["timestamp", "probability_yes"])

    data = resp.json()
    hist = data.get("history", []) if isinstance(data, dict) else []
    if not hist:
        return pd.DataFrame(columns=["timestamp", "probability_yes"])
    out = pd.DataFrame(hist)
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "probability_yes"])
    out["timestamp"] = pd.to_datetime(out["t"], unit="s", utc=True)
    out["probability_yes"] = out["p"].astype(float).clip(0.0, 1.0)
    out = out[["timestamp", "probability_yes"]].sort_values("timestamp").reset_index(drop=True)
    return out


def build_history_panel(
    universe: pd.DataFrame,
    lookback_days: int = 45,
    interval: str = "1h",
    fidelity: int = 60,
    max_markets: Optional[int] = None,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    empty_cols = [
        "event_id", "question", "deadline_date",
        "yes_token_id", "timestamp", "probability_yes",
    ]
    if universe.empty:
        return pd.DataFrame(columns=empty_cols)

    now_utc = pd.Timestamp.utcnow()
    end_ts = int(now_utc.timestamp())
    start_ts = int((now_utc - pd.Timedelta(days=lookback_days)).timestamp())
    min_ts = now_utc - pd.Timedelta(days=lookback_days)

    rows: List[pd.DataFrame] = []
    iter_df = universe.copy()
    if max_markets is not None:
        iter_df = iter_df.head(max_markets).copy()

    for _, row in iter_df.iterrows():
        hist = fetch_token_price_history(
            token_id=row["yes_token_id"],
            start_ts=start_ts,
            end_ts=end_ts,
            interval=interval,
            fidelity=fidelity,
        )
        if hist.empty:
            continue
        hist = hist[hist["timestamp"] >= min_ts].copy()
        if hist.empty:
            continue
        if fidelity and fidelity > 1:
            hist = (
                hist.set_index("timestamp")
                .resample(f"{int(fidelity)}min")
                .last()
                .dropna()
                .reset_index()
            )
            if hist.empty:
                continue
        hist["event_id"] = row["event_id"]
        hist["question"] = row["question"]
        hist["deadline_date"] = row["deadline_date"]
        hist["yes_token_id"] = row["yes_token_id"]
        rows.append(hist)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    panel = pd.concat(rows, ignore_index=True)

    # Align timestamps across tokens: floor to the interval grid so that
    # markets within the same event share identical timestamps.
    _INTERVAL_FREQ = {"1m": "1min", "5m": "5min", "1h": "1h", "6h": "6h",
                      "1d": "1D", "1w": "1W", "max": "1D"}
    freq = _INTERVAL_FREQ.get(interval, interval)
    panel["timestamp"] = panel["timestamp"].dt.floor(freq)
    panel = (
        panel.groupby(["event_id", "question", "deadline_date",
                        "yes_token_id", "timestamp"], as_index=False)
        ["probability_yes"].last()
    )

    panel["deadline_date"] = pd.to_datetime(panel["deadline_date"]).dt.date
    panel["tau_days"] = (
        pd.to_datetime(panel["deadline_date"]) - panel["timestamp"].dt.tz_convert(None).dt.normalize()
    ).dt.days.clip(lower=1)
    panel = panel.sort_values(["event_id", "timestamp", "deadline_date"]).reset_index(drop=True)
    return panel


def _last_monotonic_index_series(p: np.ndarray) -> int:
    """Last index (inclusive) such that p is non-decreasing. Returns len(p)-1 if all good."""
    if len(p) <= 1:
        return len(p) - 1
    for i in range(1, len(p)):
        if p[i] < p[i - 1]:
            return i - 1
    return len(p) - 1


def truncate_panel_to_monotonic(panel: pd.DataFrame) -> pd.DataFrame:
    """For each (event_id, timestamp), drop rows after the first decrease in probability_yes
    (when ordered by deadline_date). Use this so the algo never sees invalid long-dated prices."""
    if panel.empty:
        return panel
    rows: List[pd.DataFrame] = []
    for (eid, ts), grp in panel.groupby(["event_id", "timestamp"], dropna=False):
        grp = grp.sort_values("deadline_date").reset_index(drop=True)
        p = grp["probability_yes"].values.astype(float)
        k = _last_monotonic_index_series(p)
        rows.append(grp.iloc[: k + 1])
    return pd.concat(rows, ignore_index=True)


def report_non_monotonic_slices(
    panel: pd.DataFrame,
    max_report: int = 20,
) -> pd.DataFrame:
    """Report (event_id, timestamp, deadline_date, yes_token_id, prob, prev_prob) where
    probability_yes first decreases vs previous tenor. Use to investigate wrong/stale long-dated data."""
    out: List[dict] = []
    for (eid, ts), grp in panel.groupby(["event_id", "timestamp"], dropna=False):
        grp = grp.sort_values("deadline_date").reset_index(drop=True)
        p = grp["probability_yes"].values.astype(float)
        for i in range(1, len(p)):
            if p[i] < p[i - 1]:
                row = grp.iloc[i]
                out.append({
                    "event_id": eid,
                    "timestamp": ts,
                    "deadline_date": row["deadline_date"],
                    "tau_days": row["tau_days"],
                    "yes_token_id": row.get("yes_token_id"),
                    "probability_yes": float(p[i]),
                    "prev_probability_yes": float(p[i - 1]),
                })
                if len(out) >= max_report:
                    return pd.DataFrame(out)
                break
    return pd.DataFrame(out)


# ── Static dislocation signal (time-shifted, cross-sectionally demeaned) ──

def score_time_shifted_dislocations(
    panel: pd.DataFrame,
    lag_bars: int = 1,
    min_nodes: int = 2,
    poly_degree: int = 2,
    ref_smooth_bars: int = 1,
) -> pd.DataFrame:
    """Fit a logit polynomial at t-lag (optionally smoothed over ref_smooth_bars),
    predict each node at t, compute raw residual, then cross-sectionally demean
    so that parallel shifts produce zero signal."""
    empty_cols = [
        "event_id", "question", "timestamp", "deadline_date",
        "tau_days", "probability_yes", "ts_predicted_prob",
        "ts_residual", "ts_residual_z", "lag_bars",
    ]
    if panel.empty:
        return pd.DataFrame(columns=empty_cols)

    min_hist = lag_bars + ref_smooth_bars - 1

    out_rows: List[pd.DataFrame] = []
    for event_id, event_df in panel.groupby("event_id", dropna=False):
        timestamps = sorted(event_df["timestamp"].unique())
        if len(timestamps) <= min_hist:
            continue

        for t_idx in range(min_hist, len(timestamps)):
            ts_now = timestamps[t_idx]
            now_slice = event_df[event_df["timestamp"] == ts_now].sort_values("deadline_date")
            if now_slice.empty:
                continue

            win_start = t_idx - lag_bars - ref_smooth_bars + 1
            win_end = t_idx - lag_bars + 1
            ref_timestamps = timestamps[win_start:win_end]

            ref_slice = event_df[event_df["timestamp"].isin(ref_timestamps)]
            ref_avg = (
                ref_slice
                .groupby("deadline_date", as_index=False)["probability_yes"]
                .mean()
                .sort_values("deadline_date")
            )

            if len(ref_avg) < min_nodes:
                continue

            tau_map = now_slice.groupby("deadline_date")["tau_days"].first()
            ref_avg = ref_avg.copy()
            ref_avg["tau_days"] = ref_avg["deadline_date"].map(tau_map)
            ref_avg = ref_avg.dropna(subset=["tau_days"])
            if len(ref_avg) < min_nodes:
                continue
            x_past = ref_avg["tau_days"].to_numpy(dtype=float)
            p_past = ref_avg["probability_yes"].to_numpy(dtype=float)
            y_past = _logit(p_past)
            deg = min(poly_degree, len(ref_avg) - 1)
            coeff = _robust_polyfit(x_past, y_past, degree=deg)

            x_now = now_slice["tau_days"].to_numpy(dtype=float)
            p_now = now_slice["probability_yes"].to_numpy(dtype=float)
            y_pred_now = np.polyval(coeff, x_now)
            p_pred_now = _sigmoid(y_pred_now)

            resid_raw = p_now - p_pred_now
            resid = resid_raw - resid_raw.mean()
            scale = max(_mad(resid), 1e-6)

            chunk = now_slice[["event_id", "question", "timestamp", "deadline_date",
                               "tau_days", "probability_yes"]].copy()
            chunk["ts_predicted_prob"] = p_pred_now
            chunk["ts_residual"] = resid
            chunk["ts_residual_z"] = resid / scale
            chunk["lag_bars"] = lag_bars
            out_rows.append(chunk)

    if not out_rows:
        return pd.DataFrame(columns=empty_cols)

    out = pd.concat(out_rows, ignore_index=True)
    out = out.sort_values(["event_id", "timestamp", "deadline_date"]).reset_index(drop=True)
    return out


def event_ids_poor_static_fit(
    static_df: pd.DataFrame,
    min_obs_per_event: int = 50,
    exclude_worst_pct: float = 10.0,
) -> set:
    """Event IDs that consistently have poor fit with the static (logit-polynomial) model.

    Uses per-event mean absolute residual; higher = worse fit. Excludes events in the
    worst `exclude_worst_pct` percentile (e.g. 10 = drop worst 10% of events by fit).
    Events with fewer than `min_obs_per_event` rows are ignored (not marked poor).
    """
    if static_df.empty or "ts_residual" not in static_df.columns:
        return set()
    counts = static_df.groupby("event_id").size()
    by_event = (
        static_df.groupby("event_id")["ts_residual"]
        .agg(lambda x: np.abs(x).mean())
        .reset_index()
    )
    by_event["_n"] = by_event["event_id"].map(counts)
    by_event = by_event[by_event["_n"] >= min_obs_per_event].drop(columns=["_n"])
    if by_event.empty:
        return set()
    thresh = np.nanpercentile(by_event["ts_residual"], 100.0 - exclude_worst_pct)
    poor = set(by_event.loc[by_event["ts_residual"] > thresh, "event_id"].astype(str))
    return poor


def event_ids_poor_static_fit_warmup(
    static_df: pd.DataFrame,
    warmup_frac: float = 0.25,
    min_obs_per_event: int = 50,
    exclude_worst_pct: float = 10.0,
) -> set:
    """Warmup-only poor-fit filter to avoid lookahead in event exclusion.

    Computes poor-fit IDs from the earliest `warmup_frac` portion of timestamps,
    then returns the same type of output as `event_ids_poor_static_fit`.
    """
    if static_df.empty or "timestamp" not in static_df.columns:
        return set()

    s = static_df.dropna(subset=["timestamp"]).copy()
    if s.empty:
        return set()

    ts_min = s["timestamp"].min()
    ts_max = s["timestamp"].max()
    if pd.isna(ts_min) or pd.isna(ts_max):
        return set()

    frac = float(np.clip(warmup_frac, 0.0, 1.0))
    warmup_end = ts_min + (ts_max - ts_min) * frac
    warmup = s[s["timestamp"] <= warmup_end].copy()
    if warmup.empty:
        return set()

    return event_ids_poor_static_fit(
        warmup,
        min_obs_per_event=min_obs_per_event,
        exclude_worst_pct=exclude_worst_pct,
    )


def compute_hedge_weights(
    j_idx: int,
    n_nodes: int,
    taus: np.ndarray,
    poly_degree: int,
    max_weight_per_leg: Optional[float] = None,
    max_gross_hedge: Optional[float] = None,
) -> Dict[int, float]:
    """Equal-weight hedge for thin curves, hat-matrix hedge otherwise.

    Falls back to equal-weight when matrix issues occur or hedge weights exceed
    stabilization caps.
    """
    if n_nodes < 2:
        return {}
    w_each = -1.0 / (n_nodes - 1)
    equal_weight = {i: w_each for i in range(n_nodes) if i != j_idx}
    if n_nodes <= poly_degree + 1:
        return equal_weight

    taus = np.asarray(taus, dtype=float)
    deg = min(poly_degree, n_nodes - 1)
    X = np.vander(taus, N=deg + 1, increasing=False)
    try:
        H = X @ np.linalg.solve(X.T @ X, X.T)
    except np.linalg.LinAlgError:
        H = X @ np.linalg.lstsq(X.T @ X, X.T, rcond=None)[0]

    h_row = H[j_idx]
    denom = 1.0 - h_row[j_idx]
    if abs(denom) < 1e-10:
        return equal_weight

    weights: Dict[int, float] = {}
    for i in range(n_nodes):
        if i == j_idx:
            continue
        w = -h_row[i] / denom
        if abs(w) > 1e-8:
            weights[i] = float(w)
    if not weights:
        return equal_weight

    w_vals = list(weights.values())
    max_w = max(abs(x) for x in w_vals)
    gross = sum(abs(x) for x in w_vals)
    if (max_weight_per_leg is not None and max_w > max_weight_per_leg) or (
        max_gross_hedge is not None and gross > max_gross_hedge
    ):
        return equal_weight
    return weights


def _price_at_or_before(
    series_df: pd.DataFrame,
    ts: pd.Timestamp,
    max_lag_seconds: int = 7200,
) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
    """Return last available price at or before `ts` within lag tolerance.

    This is intentionally backward-only to prevent accidental future peeking.
    """
    if series_df.empty:
        return None, None
    idx = series_df.index.searchsorted(ts, side="right") - 1
    if idx < 0 or idx >= len(series_df):
        return None, None
    matched_ts = series_df.index[idx]
    lag = (ts - matched_ts).total_seconds()
    if lag < 0 or lag > max_lag_seconds:
        return None, None
    return float(series_df.iloc[idx]["probability_yes"]), matched_ts


def build_trades_static_dislocation(
    signals_df: pd.DataFrame,
    panel: pd.DataFrame,
    static_signal_df: pd.DataFrame,
    exit_threshold: float,
    poly_degree: int,
    resolution_map: Optional[Dict[Tuple[object, object], float]] = None,
    event_deadlines: Optional[Dict[object, List[object]]] = None,
    latest_prices: Optional[Dict[Tuple[object, object], float]] = None,
    spread_half: float = 0.0,
    max_weight_per_leg: Optional[float] = None,
    max_gross_hedge: Optional[float] = None,
    max_match_lag_seconds: int = 7200,
    shares_per_trade: Optional[float] = None,
) -> pd.DataFrame:
    """Build trades with 1-bar entry/exit delay and controlled re-entry.

    Notes:
    - Entries/exits are executed one bar after signal timestamps.
    - At most one concurrent trade per event, and one per (event, deadline) pair.
    - Hedge fills use backward-only timestamp matching (no future peeking).
    - If shares_per_trade is set (e.g. 500), PnLs are scaled to dollar PnL assuming
      that many shares per trade (1 share ≈ $1 at resolution). Columns shares and
      spread_pnl_dollars are always present; when shares_per_trade is None, shares=1.
    """
    mult = 1.0 if shares_per_trade is None else float(shares_per_trade)
    cols = [
        "event", "event_id", "entry_ts", "exit_ts", "hold_time", "spread_type",
        "status", "dis_node", "dis_entry", "dis_exit", "dis_pnl", "hedge_legs",
        "hedge_pnl", "spread_pnl", "spread_pnl_dollars", "shares", "static_resid", "n_nodes",
    ]
    if signals_df.empty or panel.empty or static_signal_df.empty:
        return pd.DataFrame(columns=cols)

    p = panel.copy()
    if resolution_map is None:
        resolution_map = {}
    if event_deadlines is None:
        event_deadlines = p.groupby("event_id")["deadline_date"].apply(lambda x: sorted(x.unique())).to_dict()
    if latest_prices is None:
        latest_prices = (
            p.sort_values("timestamp")
            .groupby(["event_id", "deadline_date"])
            .last()["probability_yes"]
            .to_dict()
        )

    evt_ts = {eid: sorted(g["timestamp"].unique()) for eid, g in p.groupby("event_id")}
    event_last_ts = p.groupby("event_id")["timestamp"].max().to_dict()
    earliest_ts = p["timestamp"].min()

    event_next_available: Dict[object, pd.Timestamp] = {}
    pair_next_available: Dict[Tuple[object, object], pd.Timestamp] = {}
    trades: List[dict] = []

    signals_sorted = signals_df.sort_values("timestamp").reset_index(drop=True)
    for _, sig in signals_sorted.iterrows():
        eid = sig["event_id"]
        dd = sig["deadline_date"]
        ts_signal = sig["timestamp"]
        direction = sig["direction"]

        if ts_signal < event_next_available.get(eid, earliest_ts):
            continue
        if ts_signal < pair_next_available.get((eid, dd), earliest_ts):
            continue

        ts_list = evt_ts.get(eid, [])
        sig_idx = next((i for i, t in enumerate(ts_list) if t >= ts_signal), None)
        if sig_idx is None or sig_idx + 1 >= len(ts_list):
            continue
        ts_entry = ts_list[sig_idx + 1]

        dis_series = (
            p[(p["event_id"] == eid) & (p["deadline_date"] == dd)]
            .drop_duplicates(subset=["timestamp"], keep="last")
            .set_index("timestamp")
            .sort_index()
        )
        dis_entry, _ = _price_at_or_before(dis_series, ts_entry, max_lag_seconds=max_match_lag_seconds)
        if dis_entry is None:
            continue

        deadlines = event_deadlines.get(eid, [])
        if dd not in deadlines:
            continue
        j_idx = deadlines.index(dd)

        entry_snap = p[(p["event_id"] == eid) & (p["timestamp"] == ts_entry)]
        if entry_snap.empty:
            continue
        tau_map = entry_snap.groupby("deadline_date")["tau_days"].first()
        available = [(i, d) for i, d in enumerate(deadlines) if d in tau_map.index]
        if len(available) < 2:
            continue
        deadlines_local = [d for _, d in available]
        taus = np.asarray([tau_map[d] for _, d in available], dtype=float)
        j_local = next((k for k, (_, d) in enumerate(available) if d == dd), None)
        if j_local is None:
            continue

        hedge_weights = compute_hedge_weights(
            j_local,
            len(deadlines_local),
            taus,
            poly_degree,
            max_weight_per_leg=max_weight_per_leg,
            max_gross_hedge=max_gross_hedge,
        )
        if not hedge_weights:
            continue

        if spread_half > 0:
            dis_entry_eff = (dis_entry + spread_half) if direction == "BUY" else (dis_entry - spread_half)
        else:
            dis_entry_eff = dis_entry

        future = static_signal_df[
            (static_signal_df["event_id"] == eid)
            & (static_signal_df["deadline_date"] == dd)
            & (static_signal_df["timestamp"] > ts_entry)
        ].sort_values("timestamp")

        exit_rows = future[future["ts_residual"].abs() < exit_threshold]
        exit_ts: Optional[pd.Timestamp] = None
        dis_exit: Optional[float] = None
        exit_status: Optional[str] = None
        if not exit_rows.empty:
            exit_sig_ts = exit_rows.iloc[0]["timestamp"]
            xi = next((i for i, t in enumerate(ts_list) if t >= exit_sig_ts), None)
            if xi is not None and xi + 1 < len(ts_list):
                ts_exit = ts_list[xi + 1]
                dis_exit, _ = _price_at_or_before(dis_series, ts_exit, max_lag_seconds=max_match_lag_seconds)
                if dis_exit is not None:
                    exit_ts = ts_exit
                    exit_status = "E"

        if exit_status is None:
            dis_res = resolution_map.get((eid, dd))
            dis_resolved = dis_res is not None and not (isinstance(dis_res, float) and np.isnan(dis_res))
            if dis_resolved:
                dis_exit = float(dis_res)
                exit_status = "R"
            else:
                dis_exit = float(latest_prices.get((eid, dd), dis_entry))
                exit_status = "P"

        effective_end_ts = exit_ts if (exit_status == "E" and exit_ts is not None) else event_last_ts.get(eid, ts_entry)
        if effective_end_ts is None:
            effective_end_ts = ts_entry

        if spread_half > 0:
            dis_exit_eff = (dis_exit - spread_half) if direction == "BUY" else (dis_exit + spread_half)
            pnl_dis = (dis_exit_eff - dis_entry_eff) if direction == "BUY" else (dis_entry_eff - dis_exit_eff)
        else:
            pnl_dis = (dis_exit - dis_entry) if direction == "BUY" else (dis_entry - dis_exit)

        legs = []
        all_ok = True
        for h_idx, w in hedge_weights.items():
            hedge_dd = deadlines_local[h_idx]
            h_sub = (
                p[(p["event_id"] == eid) & (p["deadline_date"] == hedge_dd)]
                .drop_duplicates(subset=["timestamp"], keep="last")
                .set_index("timestamp")
                .sort_index()
            )
            h_entry, _ = _price_at_or_before(h_sub, ts_entry, max_lag_seconds=max_match_lag_seconds)
            if h_entry is None:
                all_ok = False
                break

            if exit_status == "E" and exit_ts is not None:
                h_exit, _ = _price_at_or_before(h_sub, exit_ts, max_lag_seconds=max_match_lag_seconds)
                if h_exit is not None:
                    h_stat = "E"
                else:
                    h_exit = float(latest_prices.get((eid, hedge_dd), h_entry))
                    h_stat = "P"
            elif exit_status == "R":
                hr = resolution_map.get((eid, hedge_dd))
                if hr is not None and not (isinstance(hr, float) and np.isnan(hr)):
                    h_exit = float(hr)
                    h_stat = "R"
                else:
                    h_exit = float(latest_prices.get((eid, hedge_dd), h_entry))
                    h_stat = "P"
            else:
                h_exit = float(latest_prices.get((eid, hedge_dd), h_entry))
                h_stat = "P"

            if direction == "BUY":
                pnl_h = abs(w) * ((h_entry - h_exit) if w < 0 else (h_exit - h_entry))
            else:
                pnl_h = abs(w) * ((h_exit - h_entry) if w < 0 else (h_entry - h_exit))
            if spread_half > 0:
                pnl_h -= 2.0 * spread_half * abs(w)

            legs.append(
                {
                    "hedge_dd": hedge_dd,
                    "weight": round(float(w), 4),
                    "entry": round(float(h_entry), 4),
                    "status": h_stat,
                    "exit_val": round(float(h_exit), 4),
                    "pnl": round(float(pnl_h), 4),
                }
            )

        if not all_ok or not legs:
            continue

        total_pnl = float(pnl_dis + sum(l["pnl"] for l in legs))
        if exit_status == "E" and exit_ts is not None:
            hold_h = (exit_ts - ts_entry).total_seconds() / 3600.0
            hold_str = f"{hold_h:.1f}h"
        else:
            hold_str = "—"

        hedge_summary = "; ".join(
            f"{l['hedge_dd']}({l['weight']:+.3f})@{l['entry']:.3f}{l['status']}" for l in legs
        )
        primary_h = max(legs, key=lambda l: abs(l["weight"]))["hedge_dd"]
        spread_dir = (
            ("steepen" if direction == "SELL" else "flatten")
            if dd < primary_h
            else ("steepen" if direction == "BUY" else "flatten")
        )
        q_rows = p[p["event_id"] == eid]
        event_name = q_rows["question"].iloc[0][:50] if not q_rows.empty else eid

        trades.append(
            {
                "event": event_name,
                "event_id": eid,
                "entry_ts": ts_entry,
                "exit_ts": exit_ts,
                "hold_time": hold_str,
                "spread_type": spread_dir,
                "status": exit_status,
                "dis_node": dd,
                "dis_entry": round(float(dis_entry), 4),
                "dis_exit": round(float(dis_exit), 4),
                "dis_pnl": round(float(pnl_dis), 4),
                "hedge_legs": hedge_summary,
                "hedge_pnl": round(float(sum(l["pnl"] for l in legs)), 4),
                "spread_pnl": round(total_pnl, 4),
                "spread_pnl_dollars": round(total_pnl * mult, 2),
                "shares": mult,
                "static_resid": round(float(sig["ts_residual"]), 4),
                "n_nodes": len(deadlines_local),
            }
        )

        event_next_available[eid] = effective_end_ts
        pair_next_available[(eid, dd)] = effective_end_ts

    return pd.DataFrame(trades, columns=cols) if trades else pd.DataFrame(columns=cols)
