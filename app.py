def compute_recommendations(df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    MIN_SHIPMENTS = 5  # new rule: apply only when a lane has > 1 carrier

    # Rename to canonical for internal calc
    rename_map = {mapping[k]: CANON[k] for k in mapping if mapping[k]}
    work = df.rename(columns=rename_map).copy()

    # Required columns present?
    required = [CANON["carrier"], CANON["pickup"], CANON["dropoff"], CANON["volume"],
                CANON["tracked"], CANON["avg_ping"], CANON["mc"]]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # Numeric conversions (percent fields auto-scale to 0–100)
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

    # ---------- Build Lane Recommendations with min-5 rule ----------
    rec_rows = []
    for lane, g in work_sorted.groupby("Lane", sort=False):
        g = g.reset_index(drop=True)
        multi_carrier = (g["Carriers on this lane"].iat[0] or 0) > 1

        # Apply min-5 filter only for multi-carrier lanes; fallback to all rows if none meet it
        if multi_carrier:
            g_candidates = g[g[CANON["volume"]].fillna(0) >= MIN_SHIPMENTS]
            if g_candidates.empty:
                g_candidates = g.copy()
        else:
            g_candidates = g.copy()

        # If still empty (edge-case), skip lane
        if g_candidates.empty:
            continue

        # Recommendation #1
        top = g_candidates.iloc[0]
        top_score = top["Weighted Score"]
        r1 = top.copy()
        r1["Recommendation Rank"] = 1
        rec_rows.append(r1)

        # Recommendation #2 (within 10% of top score, from the candidate set)
        threshold = 0.9 * top_score
        candidates2 = g_candidates.iloc[1:]
        candidates2 = candidates2[candidates2["Weighted Score"] >= threshold]
        if not candidates2.empty:
            r2 = candidates2.iloc[0].copy()  # already sorted by score desc, ping asc
            r2["Recommendation Rank"] = 2
            rec_rows.append(r2)

    lane_recs = pd.DataFrame(rec_rows)

    # ---------- Carrier Scorecard ----------
    # Counts of lanes by rank
    rank_counts = (
        lane_recs.pivot_table(
            index=CANON["carrier"],
            columns="Recommendation Rank",
            values="Lane",
            aggfunc=lambda s: s.nunique(),
            fill_value=0,
        )
        .rename(columns={1: "Lanes as Recommendation #1", 2: "Lanes as Recommendation #2"})
        .reset_index()
    )

    # Aggregates from the full dataset
    agg_map = {
        CANON["volume"]: "sum",      # Shipment Volume (sum)
        CANON["tracked"]: "mean",    # Visibility Percentage (mean)
        CANON["mc"]: "mean",         # Milestone Completeness (mean)
        CANON["avg_ping"]: "mean",   # Avg Ping Frequency Mins (mean)
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
    scorecard = scorecard.merge(rank_counts, on="Carrier Name", how="left").fillna(
        {"Lanes as Recommendation #1": 0, "Lanes as Recommendation #2": 0}
    )

    # Exact column order for scorecard
    scorecard_cols = [
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
    for c in scorecard_cols:
        if c not in scorecard.columns:
            scorecard[c] = np.nan
    scorecard = scorecard[scorecard_cols]

    # ---------- Lane Recommendations output (your exact order) ----------
    lane_cols = [
        CANON["pickup"],                         # Pickup Location
        CANON["dropoff"],                        # Dropoff Location
        "Lane",
        CANON["carrier"],                        # Carrier Name
        "Carriers on this lane",
        "Recommendation Rank",
        CANON["volume"],                         # Shipment Volume
        CANON["tracked"],                        # Tracked Percentage
        CANON["mc"],                             # Milestone Completeness Percent
        CANON["avg_ping"],                       # Avg Ping Frequency Mins
        CANON.get("oa", "Origin Arrival Milestones Percent"),
        CANON.get("od", "Origin Departure Milestones Percent"),
        CANON.get("da", "Destination Arrival Milestones Percent"),
        CANON.get("dd", "Destination Departure Milestones Percent"),
        CANON.get("pu30", "Pickup Arrival Within 30 Min Percent"),
        CANON.get("do30", "Dropoff Arrival Within 30 Min Percent"),
    ]
    for c in lane_cols:
        if c not in lane_recs.columns:
            lane_recs[c] = np.nan
    lane_recs_final = lane_recs[lane_cols].copy()

    # Round numerics and sort both sheets by Shipment Volume (desc)
    lane_recs_final = round_numeric(lane_recs_final, DECIMALS)
    scorecard = round_numeric(scorecard, DECIMALS)

    if CANON["volume"] in lane_recs_final.columns:
        lane_recs_final = lane_recs_final.sort_values(by=CANON["volume"], ascending=False, kind="mergesort")
    if "Shipment Volume" in scorecard.columns:
        scorecard = scorecard.sort_values(by="Shipment Volume", ascending=False, kind="mergesort")

    return lane_recs_final, scorecard
