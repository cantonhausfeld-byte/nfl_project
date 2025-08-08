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
    "roll3_pd_diff","roll5_pd_diff","roll3_pf_diff","roll3_pa_diff",
    "is_dome","temp","wind",
]

def load_data():
    df = pd.read_csv(FEATS_CSV)
    for c in ["season","week","home_score","away_score","temp","wind"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_dome" in df.columns:
        df["is_dome"] = df["is_dome"].astype(int)
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df

def load_calibration():
    j = json.loads(CALIB_JSON.read_text())
    ols = j["ols"]
    logi = j["logistic"]
    w_lin = ols["coefficients"]
    b_lin = ols["intercept"]
    w_log = logi["coefficients"]
    b_log = logi["intercept"]
    return w_lin, b_lin, w_log, b_log

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def score_row(row, w_lin, b_lin, w_log, b_log):
    x = np.array([row.get(f,0.0) for f in FEATURES], dtype=float)
    lin = b_lin + float(np.sum([w_lin[f]*x[i] for i,f in enumerate(FEATURES)]))
    logit = b_log + float(np.sum([w_log[f]*x[i] for i,f in enumerate(FEATURES)]))
    p_home = float(sigmoid(logit))
    return float(lin), p_home

def load_odds_csv(path):
    # Expected cols: week, home_team, away_team, home_spread, over_under, home_moneyline, away_moneyline
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    for col in ["week","home_spread","over_under","home_moneyline","away_moneyline"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fetch_odds_api(week):
    # Placeholder: wire up your provider here using ODDS_BASE_URL + ODDS_API_KEY envs
    base = os.getenv("ODDS_BASE_URL")
    key  = os.getenv("ODDS_API_KEY")
    if not base or not key:
        print("No ODDS_BASE_URL/ODDS_API_KEY in environment; skipping API odds.")
        return pd.DataFrame(columns=["week","home_team","away_team","home_spread"])
    # TODO: implement your odds API call here
    return pd.DataFrame(columns=["week","home_team","away_team","home_spread"])

def kelly_frac(p, odds_dec):
    if pd.isna(odds_dec):
        return 0.0
    b = odds_dec - 1.0
    edge = p*b - (1-p)
    if b <= 0:
        return 0.0
    f = edge / b
    return max(0.0, min(f, 1.0))

def american_to_decimal(ml):
    if pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml > 0:
        return 1.0 + ml/100.0
    else:
        return 1.0 + 100.0/abs(ml)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True, help="NFL week (REG)")
    ap.add_argument("--season", type=int, default=2024, help="Season")
    ap.add_argument("--odds_csv", type=str, default="", help="Path to odds CSV")
    ap.add_argument("--out_csv", type=str, default="", help="Output CSV path")
    args = ap.parse_args()

    df = load_data()
    w_lin, b_lin, w_log, b_log = load_calibration()

    if "game_type" in df.columns:
        games = df[(df["season"]==args.season) & (df["week"]==args.week) & (df["game_type"]=="REG")].copy()
    else:
        games = df[(df["season"]==args.season) & (df["week"]==args.week)].copy()

    if games.empty:
        print(f"No games found for season={args.season}, week={args.week}")
        return

    preds = [score_row(row, w_lin, b_lin, w_log, b_log) for _, row in games.iterrows()]
    games["model_spread"] = [p[0] for p in preds]       # home minus away
    games["home_win_prob"] = [p[1] for p in preds]

    # odds merge
    if args.odds_csv:
        odds = load_odds_csv(args.odds_csv)
    else:
        odds = fetch_odds_api(args.week)

    picks = games.copy()
    if not odds.empty:
        picks["_home_team_l"] = picks["home_team"].str.lower()
        picks["_away_team_l"] = picks["away_team"].str.lower()
        odds["_home_team_l"] = odds["home_team"].str.lower()
        odds["_away_team_l"] = odds["away_team"].str.lower()
        picks = picks.merge(
            odds[["week","_home_team_l","_away_team_l","home_spread","over_under","home_moneyline","away_moneyline"]].rename(
                columns={"home_spread":"market_home_spread"}
            ),
            on=["week","_home_team_l","_away_team_l"],
            how="left"
        )
        picks.drop(columns=["_home_team_l","_away_team_l"], inplace=True, errors="ignore")

    if "market_home_spread" in picks.columns:
        picks["spread_edge_pts"] = picks["model_spread"] - picks["market_home_spread"]

    if "home_moneyline" in picks.columns and "away_moneyline" in picks.columns:
        picks["home_dec"] = picks["home_moneyline"].apply(american_to_decimal)
        picks["away_dec"] = picks["away_moneyline"].apply(american_to_decimal)
        picks["kelly_home"] = picks.apply(
            lambda r: kelly_frac(r["home_win_prob"], r["home_dec"]) if pd.notna(r.get("home_dec",np.nan)) else 0.0,
            axis=1
        )
        picks["kelly_away"] = picks.apply(
            lambda r: kelly_frac(1.0 - r["home_win_prob"], r["away_dec"]) if pd.notna(r.get("away_dec",np.nan)) else 0.0,
            axis=1
        )

    rec = []
    for _, r in picks.iterrows():
        side = "HOME" if r["home_win_prob"] >= 0.5 else "AWAY"
        conf = abs(r["home_win_prob"] - 0.5) * 2.0
        rec.append((side, conf))
    picks["pick_side"] = [x[0] for x in rec]
    picks["pick_confidence_0_1"] = [x[1] for x in rec]

    # Define output columns, only including those that exist in picks
    out_cols = [
        "season", "week", "gameday", "home_team", "away_team",
        "home_score", "away_score", "model_spread", "home_win_prob",
        "pick_side", "pick_confidence_0_1"
    ]
    if "market_home_spread" in picks.columns:
        out_cols += ["market_home_spread", "spread_edge_pts", "over_under"]
        for col in ["home_moneyline", "away_moneyline", "kelly_home", "kelly_away"]:
            if col in picks.columns:
                out_cols.append(col)

    out = picks[out_cols].sort_values(["gameday", "home_team"])
    print(out.to_string(index=False, max_cols=200))

    out_path = Path(args.out_csv) if args.out_csv else DATA_DIR / f"picks_{args.season}_week_{args.week}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved picks -> {out_path}")

if __name__ == "__main__":
    main()
