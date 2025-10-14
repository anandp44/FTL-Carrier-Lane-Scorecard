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
    "(min–max scaled within lane), Tracked % 20%. Tie-breaker: lowest Avg Ping Frequency (mins)."
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
    "carrier": [
        "carrier name", "carrier", "provider", "carriername"
    ],
    "pickup": [
        "pickup location", "pickup", "origin", "origin location", "origin city", "origin_city"
    ],
    "dropoff": [
        "dropoff location", "dropoff", "destination", "destination location", "destination city", "dest"
    ],
    "volume": [
        "shipment volume", "volume", "shipments", "loads", "shipment count", "num shipments", "no of shipments"
    ],
    "tracked": [
        "tracked percentage", "tracked %", "tracking percentage", "tracking %", "tracked", "track %"
    ],
    "avg_ping": [
        "avg ping frequency mins", "avg ping", "avg ping (mins)", "avg ping mins", "ping frequency", "ping mins"
    ],
    "mc": [
        "milestone completeness percent", "milestone completeness", "milestone %", "mc %"
    ],
    "oa": [
        "origin arrival milestones percent", "origin arrival %", "origin arrival pct"
    ],
    "od": [
        "origin departure milestones percent", "origin departure %", "origin departure pct"
    ],
    "da": [
        "destination arrival milestones percent", "destination arrival %", "destination arrival pct"
    ],
    "dd": [
        "destination departure milestones percent", "destination departure %", "destination departure pct"
    ],
    "pu30": [
        "pickup arrival within 30 min percent", "pickup arrival within 30 minutes percent", "pickup arrival ≤30 min %"
    ],
    "do30": [
        "dropoff arrival within 30 min percent", "dropoff arrival within 30 minutes percent", "dropoff arrival ≤30 min %"
    ],
}

NUMERIC_KEYS = ["volume", "tracked", "avg_ping", "mc", "oa", "od", "da", "dd", "pu30", "do30"]
PCT_KEYS = ["tracked", "mc", "oa", "od", "da", "dd", "pu30", "do30"]

# ---------------------------
# Helpers
# ---------------------------
def normalize(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def suggest_mapping(cols: List[str]) -> Dict[str, str]:
    """Try to map incoming columns to canonical keys."""
    norm_cols = {normalize(c): c for c in cols}
    mapping = {}
    for key, alias_list in ALIASES.items():
        sel = None
        for alias in alias_list:
            n = normalize(alias)
            if n in norm_cols:
                sel = norm_cols[n]
                break
        # exact canonical match?
        if sel is None:
            canon = CANON[key]
            if normalize(canon) in norm_cols:
                sel = norm_cols[normalize(canon)]
        mapping[key] = sel
    return mapping

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def ensure_percent_scale(s: pd.Series) -> pd.Series:
    """If data appears to be 0-1, scale to 0-100. Otherwise pass-through."""
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
        return x * np.nan  # all NaN
    if np.isclose(xmin, xmax):
        return pd.Series(100.0, index=x.index)  # single value -> 100
    return 100.0 * (x - xmin) / (xmax - xmin)

def lane_key(row, pickup_col, dropoff_col) -> str:
    return f"{row[pickup_col]} ⟶ {row[dropoff_col]}"

def compute_recommendations(df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Rename to canonical for internal calc, keep originals for export
    rename_map = {mapping[k]: CANON[k] for k in mapping if mapping[k]}
    work = df.rename(columns=rename_map).copy()

    # Required columns present?
    required = [CANON["carrier"], CANON["pickup"], CANON["dropoff"], CANON["volume"],
                CANON["tracked"], CANON["avg_ping"], CANON["mc"]]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # Numeric conversions
    for k in NUMERIC_KEYS:
        col = CANON[k]
        if col in work.columns:
            if k in PCT_KEYS:
                work[col] = ensure_percent_scale(work[col])
            else:
                work[col] = to_numeric(work[col])

    # Build lane key and per-lane counts
    work["Lane"] = work.apply(lambda r: lane_key(r, CANON["pickup"], CANON["dropoff"]), axis=1)
    lane_counts = work.groupby("Lane")[CANON["carrier"]].nunique().rename("Num Carriers In Lane")
    work = work.merge(lane_counts, on="Lane", how="left")
    work["Single Carrier Lane"] = np.where(work["Num Carriers In Lane"] == 1, "Yes", "No")

    # Scale Shipment Volume within each lane
    work["Scaled Shipment Volume (Lane)"] = work.groupby("Lane")[CANON["volume"]].transform(minmax_scale_0_100)

    # Weighted score: 0-100 scale
    work["Weighted Score"] = (
        0.5 * work[CANON["mc"]].fillna(0) +
        0.3 * work["Scaled Shipment Volume (Lane)"].fillna(0) +
        0.2 * work[CANON["tracked"]].fillna(0)
    )

    # Sort within lane: score desc, tie-breaker ping asc
    sort_cols = ["Weighted Score", CANON["avg_ping"]]
    work_sorted = work.sort_values(["Lane", "Weighted Score", CANON["avg_ping"]],
                                   ascending=[True, False, True]).copy()

    # Pick recommendations
    rec_rows = []
    for lane, g in work_sorted.groupby("Lane", sort=False):
        g = g.reset_index(drop=True)
        top = g.iloc[0]
        top_score = top["Weighted Score"]
        # Recommendation #1
        r1 = top.copy()
        r1["Recommendation Rank"] = 1
        rec_rows.append(r1)

        # Recommendation #2 (within 10% of top score)
        threshold = 0.9 * top_score
        candidates = g.iloc[1:].copy()
        candidates = candidates[candidates["Weighted Score"] >= threshold]
        if not candidates.empty:
            # already sorted by score desc, ping asc because inherited sort
            r2 = candidates.iloc[0].copy()
            r2["Recommendation Rank"] = 2
            rec_rows.append(r2)

    lane_recs = pd.DataFrame(rec_rows)

    # Carrier scorecard (one row per carrier)
    # Counts of lanes where carrier is #1 / #2
    rank_counts = lane_recs.groupby([CANON["carrier"], "Recommendation Rank"])["Lane"].nunique().unstack(fill_value=0)
    rank_counts = rank_counts.rename(columns={1: "Lanes as #1", 2: "Lanes as #2"}).reset_index()

    # Aggregates from full dataset (not just recommended rows)
    agg_map = {
        CANON["volume"]: "sum",
        CANON["avg_ping"]: "mean",
        CANON["mc"]: "mean",
        CANON["tracked"]: "mean",
    }
    # Add other percentage columns if present
    for k in ["oa", "od", "da", "dd", "pu30", "do30"]:
        col = CANON[k]
        if col in work.columns:
            agg_map[col] = "mean"

    base_agg = work.groupby(CANON["carrier"]).agg(agg_map).reset_index()
    base_agg = base_agg.rename(columns={
        CANON["volume"]: "Total Shipment Volume",
        CANON["avg_ping"]: "Avg Ping Frequency (mins) – Mean",
        CANON["mc"]: "Milestone Completeness % – Mean",
        CANON["tracked"]: "Tracked % – Mean",
        CANON.get("oa", "Origin Arrival Milestones Percent"): "Origin Arrival % – Mean",
        CANON.get("od", "Origin Departure Milestones Percent"): "Origin Departure % – Mean",
        CANON.get("da", "Destination Arrival Milestones Percent"): "Destination Arrival % – Mean",
        CANON.get("dd", "Destination Departure Milestones Percent"): "Destination Departure % – Mean",
        CANON.get("pu30", "Pickup Arrival Within 30 Min Percent"): "Pickup Arrival ≤30 Min % – Mean",
        CANON.get("do30", "Dropoff Arrival Within 30 Min Percent"): "Dropoff Arrival ≤30 Min % – Mean",
    })

    # Lane coverage metrics per carrier
    lanes_by_carrier = work.groupby(CANON["carrier"])["Lane"].nunique().rename("Unique Lanes Served")
    pick_uniq = work.groupby(CANON["carrier"])[CANON["pickup"]].nunique().rename("Unique Pickup Locations")
    drop_uniq = work.groupby(CANON["carrier"])[CANON["dropoff"]].nunique().rename("Unique Dropoff Locations")

    carrier_scorecard = (
        base_agg.merge(rank_counts, on=CANON["carrier"], how="left")
                .merge(lanes_by_carrier, on=CANON["carrier"], how="left")
                .merge(pick_uniq, on=CANON["carrier"], how="left")
                .merge(drop_uniq, on=CANON["carrier"], how="left")
                .fillna({"Lanes as #1": 0, "Lanes as #2": 0})
                .sort_values(["Lanes as #1", "Lanes as #2", "Total Shipment Volume"], ascending=[False, False, False])
    )

    # Order and visibility for Lane Recommendations
    display_cols = [
        "Lane",
        CANON["pickup"],
        CANON["dropoff"],
        CANON["carrier"],
        "Recommendation Rank",
        "Num Carriers In Lane",
        "Single Carrier Lane",
        CANON["volume"],
        "Scaled Shipment Volume (Lane)",
        CANON["mc"],
        CANON["tracked"],
        CANON["avg_ping"],
        CANON.get("oa", ""),
        CANON.get("od", ""),
        CANON.get("da", ""),
        CANON.get("dd", ""),
        CANON.get("pu30", ""),
        CANON.get("do30", ""),
        "Weighted Score",
    ]
    display_cols = [c for c in display_cols if c and c in lane_recs.columns]
    lane_recs = lane_recs[display_cols]

    return lane_recs, carrier_scorecard

def autosize_and_write_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            # Freeze header
            ws.freeze_panes(1, 0)
            # Simple autofit by measuring string lengths
            for idx, col in enumerate(df.columns):
                series = df[col].astype(str)
                max_len = max([len(col)] + series.map(len).tolist())
                ws.set_column(idx, idx, min(max_len + 2, 60))
    return output.getvalue()

# ---------------------------
# Sidebar: file input + mapping
# ---------------------------
with st.sidebar:
    st.header("Upload & Column Mapping")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            raw = pd.read_csv(uploaded, encoding_errors="ignore")
        st.success(f"Loaded {raw.shape[0]:,} rows × {raw.shape[1]} columns")

        # Auto-suggest mapping
        mapping = suggest_mapping(list(raw.columns))

        st.write("**Detected mapping** (edit if needed):")
        for key in CANON.keys():
            options = ["-- Unmapped --"] + list(raw.columns)
            default = mapping.get(key)
            idx = options.index(default) if default in options else 0
            sel = st.selectbox(f"{CANON[key]}", options=options, index=idx, key=f"map_{key}")
            mapping[key] = None if sel == "-- Unmapped --" else sel

        # Validate required fields
        required_keys = ["carrier", "pickup", "dropoff", "volume", "tracked", "avg_ping", "mc"]
        missing_keys = [k for k in required_keys if not mapping.get(k)]
        if missing_keys:
            st.error(
                "Please map all required fields: " +
                ", ".join(CANON[k] for k in missing_keys)
            )
        else:
            st.success("All required fields mapped. You’re good to go!")

# ---------------------------
# Main: processing & output
# ---------------------------
if uploaded is None:
    st.info("⬅️ Upload your CSV to get started. The sample you mentioned will also work as-is.")
else:
    # Get mapping back from sidebar widgets
    mapping_live = {}
    for key in CANON.keys():
        sel = st.session_state.get(f"map_{key}")
        if sel and sel != "-- Unmapped --":
            mapping_live[key] = sel

    with st.expander("Preview uploaded data"):
        st.dataframe(raw.head(25), use_container_width=True)

    # Button to run analysis
    if st.button("🔍 Run Lane Analysis & Build Scorecard", type="primary"):
        try:
            lane_recs, carrier_scorecard = compute_recommendations(raw, mapping_live)

            st.subheader("✅ Lane Recommendations")
            st.caption("One or two rows per lane (Rank 1 always; Rank 2 if within 10% of Rank 1 score).")
            st.dataframe(lane_recs, use_container_width=True, height=400)

            st.subheader("📊 Carrier Scorecard")
            st.caption("One row per carrier with counts of lanes where they are Rank #1 and #2, plus aggregates.")
            st.dataframe(carrier_scorecard, use_container_width=True, height=400)

            # Build XLSX for download
            xlsx_bytes = autosize_and_write_excel({
                "Lane Recommendations": lane_recs,
                "Carrier Scorecard": carrier_scorecard
            })

            st.download_button(
                label="⬇️ Download XLSX (2 sheets)",
                data=xlsx_bytes,
                file_name="carrier_lane_analysis_and_scorecard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.success("Done! XLSX is ready.")
        except Exception as e:
            st.exception(e)

# Footer hint
st.markdown(
    "<br><small>Tip: If a percentage column in your data is 0–1 instead of 0–100, "
    "the app will auto-scale it to 0–100.</small>",
    unsafe_allow_html=True,
)
