from __future__ import annotations

import datetime as dt
import json
import re
import time
from typing import Dict, List, Optional

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

    universe = universe.drop_duplicates(subset=["event_id", "deadline_date"], keep="first")

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
