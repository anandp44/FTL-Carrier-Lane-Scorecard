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
    "Upload 'Truckload Analytics output file- filtered by lane' from Movement. The app recommends the best carrier(s) per lane "
    "and builds a carrier scorecard. Weights: Milestone Completeness 50%, Shipment Volume 30% "
    "(min–max scaled within lane), Tracked % 20%. Tie-breaker: lowest Avg Ping Frequency (mins). "
    "All numbers are rounded to 2 decimal places."
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
        return x * np.nan
    if np.isclose(xmin, xmax):
        return pd.Series(100.0, index=x.index)  # single value -> 100
    return 100.0 * (x - xmin) / (xmax - xmin)

def lane_key(row, pickup_col, dropoff_col) -> str:
    return f"{row[pickup_col]} ⟶ {row[dropoff_col]}"

def round_numeric(df: pd.DataFrame, decimals: int = DECIMALS) -> pd.DataFrame:
    """Round all numeric columns to N decimals."""
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(decimals)
    return out

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

    # Lane key + counts
    work["Lane"] = work.apply(lambda r: lane_key(r, CANON["pickup"], CANON["dropoff"]), axis=1)
    lane_counts = work.groupby("Lane")[CANON["carrier"]].nunique().rename("Num Carriers In Lane")
    work = work.merge(lane_counts, on="Lane", how="left")
    work["Single Carrier Lane"] = np.where(work["Num Carriers In Lane"] == 1, "Yes", "No")

    # Scale Shipment Volume within each lane
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

    # Pick recommendations
    rec_rows = []
    for lane, g in work_sorted.groupby("Lane", sort=False):
        g = g.reset_index(drop=True)
        top = g.iloc[0]
        top_score = top["Weighted Score"]

        r1 = top.copy()
        r1["Recommendation Rank"] = 1
        rec_rows.append(r1)

        threshold = 0.9 * top_score  # within 10% of top
        candidates = g.iloc[1:].copy()
        candidates = candidates[candidates["Weighted Score"] >= threshold]
        if not candidates.empty:
            r2 = candidates.iloc[0].copy()  # already sorted by score desc, ping asc
            r2["Recommendation Rank"] = 2
            rec_rows.append(r2)

    lane_recs = pd.DataFrame(rec_rows)

    # --- Carrier Scorecard (exact columns requested) ---
    # Counts of lanes where carrier is #1 or #2
    rank_counts = lane_recs.groupby([CANON["carrier"], "Recommendation Rank"])["Lane"].nunique().unstack(fill_value=0)
    rank_counts = rank_counts.rename(columns={1: "Lanes as Recommendation #1", 2: "Lanes as Recommendation #2"}).reset_index()

    # Aggregates from full dataset (not just recommended rows)
    agg_map = {
        CANON["volume"]: "sum",                  # Shipment Volume (sum)
        CANON["tracked"]: "mean",                # Visibility Percentage (mean)
        CANON["mc"]: "mean",                     # Milestone Completeness (mean)
        CANON["avg_ping"]: "mean",               # Avg Ping Frequency Mins (mean)
    }
    for k in ["oa", "od", "da", "dd", "pu30", "do30"]:
        col = CANON[k]
        if col in work.columns:
            agg_map[col] = "mean"

    base_agg = work.groupby(CANON["carrier"]).agg(agg_map).reset_index()

    # Build final scorecard with exact headers
    scorecard = base_agg.rename(columns={
        CANON["carrier"]: "Carrier Name",
        CANON["volume"]: "Shipment Volume",
        CANON["tracked"]: "Visibility Percentage",
        CANON["mc"]: "Milestone Completeness Percent",
        CANON["avg_ping"]: "Avg Ping Frequency Mins",
        CANON.get("oa", "Origin Arrival Milestones Percent"): "Origin Arrival Milestones Percent",
        CANON.get("od", "Origin Departure Milestones Percent"): "Origin Departure Milestones Percent",
        CANON.get("da", "Destination Arrival Milestones Percent"): "Destination Arrival Milestones Percent",
        CANON.get("dd", "Destination Departure Milestones Percent"): "Destination Departure Milestones Percent",
        CANON.get("pu30", "Pickup Arrival Within 30 Min Percent"): "Pickup Arrival Within 30 Min Percent",
        CANON.get("do30", "Dropoff Arrival Within 30 Min Percent"): "Dropoff Arrival Within 30 Min Percent",
    })

    scorecard = (
        scorecard.merge(rank_counts, left_on="Carrier Name", right_on=CANON["carrier"], how="left")
                 .drop(columns=[CANON["carrier"]])
                 .fillna({"Lanes as Recommendation #1": 0, "Lanes as Recommendation #2": 0})
    )

    # Ensure exact column order (include optional cols even if absent)
    desired_cols = [
        "Carrier Name",
        "Lanes as Recommendation #1",
        "Lanes as Recommendation #2",
        "Shipment Volume",
        "Visibility Percentage",
        "Milestone Completeness Percent",
        "Avg Ping Frequency Mins",
        "Origin Arrival Milestones Percent",
        "Origin Departure Milestones Percent",
        "Destination Arrival Milestones Percent",
        "Destination Departure Milestones Percent",
        "Pickup Arrival Within 30 Min Percent",
        "Dropoff Arrival Within 30 Min Percent",
    ]
    # Add any missing optional columns as NaN so the header is always present
    for c in desired_cols:
        if c not in scorecard.columns:
            scorecard[c] = np.nan
    scorecard = scorecard[desired_cols]

    # --- Lane Recommendations sheet: include ALL original input columns + computed fields ---
    input_cols = [c for c in df.columns]  # user's original columns
    computed_cols = [
        "Lane",
        "Recommendation Rank",
        "Num Carriers In Lane",
        "Single Carrier Lane",
        "Scaled Shipment Volume (Lane)",
        "Weighted Score",
    ]
    # Map back: if we renamed some columns to canonical names that differ from input, bring both
    # Start from the 'work' copy for accurate values but ensure we select the user's columns if present
    lane_recs_subset = lane_recs.copy()
    # Ensure all original columns appear (pull from lane_recs which is based on 'work')
    for col in input_cols:
        if col not in lane_recs_subset.columns and col in work.columns:
            lane_recs_subset[col] = lane_recs[col] if col in lane_recs.columns else work[col]

    # Compose final columns order: user input cols (as-is), then our computed columns (if present)
    final_lane_cols = [c for c in input_cols if c in lane_recs_subset.columns] + [c for c in computed_cols if c in lane_recs_subset.columns]
    lane_recs_final = lane_recs_subset[final_lane_cols].copy()

    # Round all numerics to DECIMALS
    lane_recs_final = round_numeric(lane_recs_final, DECIMALS)
    scorecard = round_numeric(scorecard, DECIMALS)

    return lane_recs_final, scorecard

def autosize_and_write_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
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
    st.info("⬅️ Upload your CSV to get started.")
else:
    # Get mapping back from sidebar widgets
    mapping_live = {}
    for key in CANON.keys():
        sel = st.session_state.get(f"map_{key}")
        if sel and sel != "-- Unmapped --":
            mapping_live[key] = sel

    with st.expander("Preview uploaded data"):
        st.dataframe(raw.head(25), use_container_width=True)

    if st.button("🔍 Run Lane Analysis & Build Scorecard", type="primary"):
        try:
            lane_recs, carrier_scorecard = compute_recommendations(raw, mapping_live)

            st.subheader("✅ Lane Recommendations (Truckload Analytics)")
            st.caption("Rank 1 is always present per lane; Rank 2 is included if within 10% of Rank 1’s score. All numbers shown are rounded to 1 decimal place.")
            st.dataframe(lane_recs, use_container_width=True, height=420)

            # ---------------------------
            # Filter & Download by Lanes (instructions + tool)
            # ---------------------------
            st.markdown("### 🎯 Download Truckload Analytics filtered by Lanes")
            st.info(
                "Instructions:\n"
                "1) Use the **lane filter** below to select one or more lanes.\n"
                "2) The table will update to show only the selected lanes.\n"
                "3) Click **Download filtered lanes (XLSX)** to save the filtered Truckload Analytics data."
            )

            lane_options = sorted(lane_recs["Lane"].unique().tolist()) if "Lane" in lane_recs.columns else []
            selected_lanes = st.multiselect("Filter by Lane", lane_options, placeholder="Select lanes…")
            if selected_lanes:
                lane_filtered = lane_recs[lane_recs["Lane"].isin(selected_lanes)].copy()
            else:
                lane_filtered = lane_recs.copy()

            st.dataframe(lane_filtered, use_container_width=True, height=300)

            # Build filtered XLSX for download (only the lane sheet)
            filtered_xlsx = autosize_and_write_excel({"Lane Recommendations (Filtered)": lane_filtered})
            st.download_button(
                label="⬇️ Download filtered lanes (XLSX)",
                data=filtered_xlsx,
                file_name="truckload_analytics_filtered_by_lanes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.subheader("📊 Carrier Scorecard")
            st.caption("One row per carrier. Counts of lanes where they are Recommendation #1 and #2, plus aggregated metrics (means/sums). All numbers are rounded to 1 decimal place.")
            st.dataframe(carrier_scorecard, use_container_width=True, height=420)

            # Full XLSX (two sheets)
            xlsx_bytes = autosize_and_write_excel({
                "Lane Recommendations": lane_recs,
                "Carrier Scorecard": carrier_scorecard
            })
            st.download_button(
                label="⬇️ Download XLSX (Lane Recommendations + Carrier Scorecard)",
                data=xlsx_bytes,
                file_name="carrier_lane_analysis_and_scorecard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.success("Done! Downloads are ready.")
        except Exception as e:
            st.exception(e)

# Footer hint
st.markdown(
    "<br><small>Percent fields are auto-scaled to 0–100 if your data is 0–1. "
    "All numbers in outputs are rounded to 1 decimal place.</small>",
    unsafe_allow_html=True,
)

