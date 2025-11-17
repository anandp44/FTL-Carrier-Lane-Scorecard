from __future__ import annotations

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Carrier Lane Analyzer & Scorecard",
    layout="wide",
)

st.title("🚚 Carrier Lane Analyzer & Scorecard")
st.caption(
    "Upload a CSV with carrier performance. The app recommends the best carrier(s) per lane "
    "and builds a carrier scorecard. Weights: Milestone Completeness 50%, Shipment Volume 30% "
    "(min–max scaled within lane), Tracked % 20%. Tie-breaker: lowest Avg Ping Frequency (mins). "
    "All numbers are rounded to 1 decimal place. Multi-carrier lanes apply a min 5 shipments rule with fallback."
)

# ---------------------------
# Canonical columns & aliases (case-insensitive)
# ---------------------------
CANON = {
    "carrier": "Carrier Name",
    "pickup": "Pickup Location",
    "dropoff": "Dropoff Location",
    "volume": "Shipment Volume",
    "tracked": "Tracked Percentage",
    "avg_ping": "Avg Ping Frequency Mins",
    "mc": "Milestone Completeness Percent",
    "oa": "Origin Arrival Milestones Percent",
    "od": "Origin Departure Milestones Percent",
    "da": "Destination Arrival Milestones Percent",
    "dd": "Destination Departure Milestones Percent",
    "pu30": "Pickup Arrival Within 30 Min Percent",
    "do30": "Dropoff Arrival Within 30 Min Percent",
}

ALIASES = {
    "carrier": ["carrier name", "carrier", "provider", "carriername"],
    "pickup": ["pickup location", "pickup", "origin", "origin location", "origin city", "origin_city"],
    "dropoff": ["dropoff location", "dropoff", "destination", "destination location", "destination city", "dest"],
    "volume": ["shipment volume", "volume", "shipments", "loads", "shipment count", "num shipments", "no of shipments"],
    "tracked": ["tracked percentage", "tracked %", "tracking percentage", "tracking %", "tracked", "track %"],
    "avg_ping": ["avg ping frequency mins", "avg ping", "avg ping (mins)", "avg ping mins", "ping frequency", "ping mins"],
    "mc": ["milestone completeness percent", "milestone completeness", "milestone %", "mc %"],
    "oa": ["origin arrival milestones percent", "origin arrival %", "origin arrival pct"],
    "od": ["origin departure milestones percent", "origin departure %", "origin departure pct"],
    "da": ["destination arrival milestones percent", "destination arrival %", "destination arrival pct"],
    "dd": ["destination departure milestones percent", "destination departure %", "destination departure pct"],
    "pu30": ["pickup arrival within 30 min percent", "pickup arrival within 30 minutes percent", "pickup arrival ≤30 min %"],
    "do30": ["dropoff arrival within 30 min percent", "dropoff arrival within 30 minutes percent", "dropoff arrival ≤30 min %"],
}

NUMERIC_KEYS = ["volume", "tracked", "avg_ping", "mc", "oa", "od", "da", "dd", "pu30", "do30"]
PCT_KEYS = ["tracked", "mc", "oa", "od", "da", "dd", "pu30", "do30"]
DECIMALS = 1  # global rounding

# ---------------------------
# Helpers
# ---------------------------
def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def suggest_mapping(cols: List[str]) -> Dict[str, str]:
    norm_cols = {normalize(c): c for c in cols}
    mapping = {}
    for key, alias_list in ALIASES.items():
        sel = None
        for alias in alias_list:
            n = normalize(alias)
            if n in norm_cols:
                sel = norm_cols[n]
                break
        if sel is None:
            canon = CANON[key]
            if normalize(canon) in norm_cols:
                sel = norm_cols[normalize(canon)]
        mapping[key] = sel
    return mapping

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def ensure_percent_scale(s: pd.Series) -> pd.Series:
    s = to_numeric(s)
    if s.dropna().max() is not None and s.dropna().max() <= 1.0:
        return s * 100.0
    return s

def minmax_scale_0_100(x: pd.Series) -> pd.Series:
    x = to_numeric(x).astype(float)
    if x.empty:
        return x
    xmin, xmax = x.min(), x.max()
    if pd.isna(xmin) or pd.isna(xmax):
        return x * np.nan
    if np.isclose(xmin, xmax):
        return pd.Series(100.0, index=x.index)
    return 100.0 * (x - xmin) / (xmax - xmin)

def lane_key(row, pickup_col, dropoff_col) -> str:
    return f"{row[pickup_col]} ⟶ {row[dropoff_col]}"

def round_numeric(df: pd.DataFrame, decimals: int = DECIMALS) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(decimals)
    return out

# ---------------------------
# Core compute (with min-5 rule)
# ---------------------------
def compute_recommendations(df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    MIN_SHIPMENTS = 5  # apply only for lanes with >1 carrier

    # Rename to canonical for internal calc
    rename_map = {mapping[k]: CANON[k] for k in mapping if mapping[k]}
    work = df.rename(columns=rename_map).copy()

    # Required columns present?
    required = [CANON["carrier"], CANON["pickup"], CANON["dropoff"], CANON["volume"],
                CANON["tracked"], CANON["avg_ping"], CANON["mc"]]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # Numeric conversions (auto-scale percentage fields)
    for k in NUMERIC_KEYS:
        col = CANON[k]
        if col in work.columns:
            if k in PCT_KEYS:
                work[col] = ensure_percent_scale(work[col])
            else:
                work[col] = to_numeric(work[col])

    # Lane key + counts
    work["Lane"] = work.apply(lambda r: lane_key(r, CANON["pickup"], CANON["dropoff"]), axis=1)
    lane_counts = work.groupby("Lane")[CANON["carrier"]].nunique().rename("Carriers on this lane")
    work = work.merge(lane_counts, on="Lane", how="left")
    work["Single Carrier Lane"] = np.where(work["Carriers on this lane"] == 1, "Yes", "No")

    # Scale Shipment Volume within each lane (for scoring only)
    work["Scaled Shipment Volume (Lane)"] = work.groupby("Lane")[CANON["volume"]].transform(minmax_scale_0_100)

    # Weighted score (0–100)
    work["Weighted Score"] = (
        0.5 * work[CANON["mc"]].fillna(0) +
        0.3 * work["Scaled Shipment Volume (Lane)"].fillna(0) +
        0.2 * work[CANON["tracked"]].fillna(0)
    )

    # Sort within lane: score desc, ping asc (tie-breaker)
    work_sorted = work.sort_values(["Lane", "Weighted Score", CANON["avg_ping"]],
                                   ascending=[True, False, True]).copy()

    # ---------- Lane Recommendations with min-5 rule ----------
    rec_rows = []
    for lane, g in work_sorted.groupby("Lane", sort=False):
        g = g.reset_index(drop=True)
        multi_carrier = (g["Carriers on this lane"].iat[0] or 0) > 1

        # For multi-carrier lanes, consider only rows with >= 5 shipments; fallback to all if none
        if multi_carrier:
            g_candidates = g[g[CANON["volume"]].fillna(0) >= MIN_SHIPMENTS]
            if g_candidates.empty:
                g_candidates = g.copy()
        else:
            g_candidates = g.copy()

        if g_candidates.empty:
            continue

        # Recommendation #1
        top = g_candidates.iloc[0]
        top_score = top["Weighted Score"]
        r1 = top.copy()
        r1["Recommendation Rank"] = 1
        rec_rows.append(r1)

        # Recommendation #2 within 10% of top (still within the candidate set)
        threshold = 0.9 * top_score
        candidates2 = g_candidates.iloc[1:]
        candidates2 = candidates2[candidates2["Weighted Score"] >= threshold]
        if not candidates2.empty:
            r2 = candidates2.iloc[0].copy()
