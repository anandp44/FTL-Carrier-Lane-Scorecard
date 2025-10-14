# Carrier Lane Analyzer & Scorecard (Streamlit)

A Streamlit app that:
- Recommends the best carrier(s) for each **lane** (Pickup Location × Dropoff Location).
- Builds a **Carrier Scorecard** (one row per carrier) with counts of lanes where they are #1 and #2, plus helpful aggregates.
- Exports a **single XLSX** with two sheets: `Lane Recommendations` and `Carrier Scorecard`.

## Scoring & Rules
- **Weighted Score (0–100)** per row:
  - `0.5 * Milestone Completeness %`
  + `0.3 * Scaled Shipment Volume (0–100 within lane)`
  + `0.2 * Tracked %`
- **Rank #1**: Highest weighted score in lane (tie-breaker: **lowest Avg Ping Frequency (mins)**).
- **Rank #2**: Next-best carrier if its score is within **10%** of Rank #1.
- Each lane always has at least one recommendation. Single-carrier lanes are flagged.

## Input Columns (case-insensitive, flexible mapping)
Required:
- Carrier Name
- Pickup Location
- Dropoff Location
- Shipment Volume
- Tracked Percentage
- Avg Ping Frequency Mins
- Milestone Completeness Percent

Optional (if present, shown/aggregated):
- Origin Arrival Milestones Percent
- Origin Departure Milestones Percent
- Destination Arrival Milestones Percent
- Destination Departure Milestones Percent
- Pickup Arrival Within 30 Min Percent
- Dropoff Arrival Within 30 Min Percent

Percent fields are auto-scaled to 0–100 if your data is 0–1.

## Local Run
```bash
# Python 3.10+ recommended
pip install -r requirements.txt
streamlit run app.py
