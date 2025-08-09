import json
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/workspaces/nfl_project/data")
FEATS_CSV = DATA_DIR / "nfl_games_2020_2024_features.csv"
CALIB_JSON = DATA_DIR / "calibrated_weights_2020_2024.json"

FEATURES = [
    "roll3_pd_diff", "roll5_pd_diff", "roll3_pf_diff", "roll3_pa_diff",
    "is_dome", "temp", "wind",
]

def load_data():
    df = pd.read_csv(FEATS_CSV)
    for c in ["season", "week", "home_score", "away_score", "temp", "wind"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_dome" in df.columns:
        df["is_dome"] = df["is_dome"].astype(int)
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df

def load_calibration():
    j = json.loads(CALIB_JSON.read_text())
    # Try to extract coefficients from typical structure
    if "ols" in j and "coefficients" in j["ols"]:
        return j["ols"]["coefficients"]
    # fallback: return as-is
    return j

def predict(df, calib):
    # Example scoring logic
    df["model_spread"] = (
        calib["roll3_pd_diff"] * df["roll3_pd_diff"] +
        calib["roll5_pd_diff"] * df["roll5_pd_diff"] +
        calib["roll3_pf_diff"] * df["roll3_pf_diff"] +
        calib["roll3_pa_diff"] * df["roll3_pa_diff"] +
        calib["is_dome"] * df["is_dome"] +
        calib["temp"] * df["temp"] +
        calib["wind"] * df["wind"]
    )
    df["home_win_prob"] = 1 / (1 + np.exp(-df["model_spread"]))
    df["pick_side"] = np.where(df["home_win_prob"] >= 0.5, "HOME", "AWAY")
    df["pick_confidence_0_1"] = abs(df["home_win_prob"] - 0.5) * 2
    return df

def merge_odds(picks, odds_csv):
    odds = pd.read_csv(odds_csv)
    merged = picks.merge(
        odds,
        on=["week", "home_team", "away_team"],
        how="left"
    )
    merged["spread_edge_pts"] = merged["model_spread"] - merged["home_spread"]
    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--odds_csv", type=str, default=None)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    df = load_data()
    calib = load_calibration()

    # Filter for requested week/season
    week_df = df[(df["season"] == args.season) & (df["week"] == args.week)].copy()

    picks = predict(week_df, calib)

    if args.odds_csv:
        picks = merge_odds(picks, args.odds_csv)

    picks.to_csv(args.out_csv, index=False)

    print(f"Saved picks -> {args.out_csv}")

