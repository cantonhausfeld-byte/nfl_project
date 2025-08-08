import pandas as pd
import numpy as np

IN_CSV = "nfl_games_2020_2024.csv"
OUT_CSV = "nfl_games_2020_2024_features.csv"
OUT_PARQ = "nfl_games_2020_2024_features.parquet"

DOMES = {
    "ATL97","ARI02","HOU02","IND02","MIN02","NO01","DET02","DAL02","LAR30","LAC30","LV01"
}

def safe_dt(s):
    return pd.to_datetime(s, errors="coerce")

def load_data(path):
    df = pd.read_csv(path)
    if "gameday" in df.columns:
        df["gameday"] = safe_dt(df["gameday"])
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
    for c in ["home_score","away_score","temp","wind"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_basic_features(df):
    df["point_diff"] = df["home_score"] - df["away_score"]
    df["total_points"] = df["home_score"] + df["away_score"]
    df["home_win"] = (df["point_diff"] > 0).astype(int)
    df["home_loss"] = (df["point_diff"] < 0).astype(int)
    df["home_tie"] = (df["point_diff"] == 0).astype(int)

    df["is_dome"] = df.get("stadium_id", pd.Series(index=df.index, dtype=object)).isin(DOMES)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    def impute_weather(g):
        g = g.copy()
        if "temp" in g.columns:
            dome_default = 70.0
            med_temp = g["temp"].median()
            g.loc[g["is_dome"] & g["temp"].isna(), "temp"] = dome_default
            g["temp"] = g["temp"].fillna(med_temp)
        if "wind" in g.columns:
            med_wind = g["wind"].median()
            g.loc[g["is_dome"] & g["wind"].isna(), "wind"] = 0.0
            g["wind"] = g["wind"].fillna(med_wind)
        return g

    if "temp" in df.columns or "wind" in df.columns:
        df = df.groupby("season", group_keys=False).apply(impute_weather)

    if "temp" in df.columns:
        df["temp_bucket"] = pd.cut(df["temp"], bins=[-10,32,50,70,85,200],
                                   labels=["frigid","cold","mild","warm","hot"])
    if "wind" in df.columns:
        df["wind_bucket"] = pd.cut(df["wind"], bins=[-1,3,10,18,100],
                                   labels=["calm","breeze","windy","gusty"])
    return df

def build_team_long(df):
    common = [c for c in ["game_id","season","game_type","week","gameday","stadium_id","is_dome","temp","wind"] if c in df.columns]

    home = df.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = True
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]
    if "home_qb_name" in home.columns: home["team_qb"] = home["home_qb_name"]
    if "away_qb_name" in home.columns: home["opp_qb"] = home["away_qb_name"]

    away = df.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = False
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]
    if "away_qb_name" in away.columns: away["team_qb"] = away["away_qb_name"]
    if "home_qb_name" in away.columns: away["opp_qb"] = away["home_qb_name"]

    keep = common + ["team","opponent","is_home","points_for","points_against","team_qb","opp_qb","gameday","week","season"]

    keep = [k for k in keep if k in home.columns]
    long = pd.concat([home[keep], away[keep]], ignore_index=True)
    long["point_diff"] = long["points_for"] - long["points_against"]
    return long

def add_rolling(long):
    # Print any duplicated columns for debugging
    print("Duplicated columns before dropping:", long.columns[long.columns.duplicated()].tolist())
    # Remove duplicate columns (keep first occurrence)
    long = long.loc[:, ~long.columns.duplicated()]
    long = long.sort_values(["season", "team", "gameday", "week"], na_position="last")
    def add_roll(g):
        g = g.copy()
        g["roll3_pf"] = g["points_for"].rolling(3, min_periods=1).mean()
        g["roll3_pa"] = g["points_against"].rolling(3, min_periods=1).mean()
        g["roll3_pd"] = g["point_diff"].rolling(3, min_periods=1).mean()
        g["roll5_pd"] = g["point_diff"].rolling(5, min_periods=1).mean()
        return g
    return long.groupby(["season","team"], group_keys=False).apply(add_roll)

def merge_back(game_df, long):
    base = ["season","gameday","week","team","roll3_pf","roll3_pa","roll3_pd","roll5_pd"]
    home_feats = long[base].rename(columns={
        "team":"home_team","roll3_pf":"home_roll3_pf","roll3_pa":"home_roll3_pa","roll3_pd":"home_roll3_pd","roll5_pd":"home_roll5_pd"
    })
    away_feats = long[base].rename(columns={
        "team":"away_team","roll3_pf":"away_roll3_pf","roll3_pa":"away_roll3_pa","roll3_pd":"away_roll3_pd","roll5_pd":"away_roll5_pd"
    })

    m = game_df.merge(home_feats, on=["season","home_team","gameday","week"], how="left")
    m = m.merge(away_feats, on=["season","away_team","gameday","week"], how="left")

    m["roll3_pd_diff"] = m["home_roll3_pd"] - m["away_roll3_pd"]
    m["roll5_pd_diff"] = m["home_roll5_pd"] - m["away_roll5_pd"]
    m["roll3_pf_diff"] = m["home_roll3_pf"] - m["away_roll3_pf"]
    m["roll3_pa_diff"] = m["home_roll3_pa"] - m["away_roll3_pa"]
    return m

def main():
    df = load_data(IN_CSV)
    if "game_type" in df.columns:
        df = df[df["game_type"] == "REG"].copy()

    df = add_basic_features(df)
    long = build_team_long(df)
    long = add_rolling(long)
    enriched = merge_back(df, long)

    useful = [
        "game_id","season","week","gameday","home_team","away_team","home_score","away_score",
        "point_diff","total_points","home_win","is_dome","temp","wind","temp_bucket","wind_bucket",
        "home_roll3_pd","away_roll3_pd","roll3_pd_diff","roll5_pd_diff",
        "home_roll3_pf","away_roll3_pf","home_roll3_pa","away_roll3_pa","roll3_pf_diff","roll3_pa_diff",
        "home_qb_name","away_qb_name","stadium_id"
    ]
    have = [c for c in useful if c in enriched.columns]
    final_cols = have + [c for c in enriched.columns if c not in have]

    enriched = enriched[final_cols].sort_values(["season","week","gameday","home_team"])

    print("Rows:", len(enriched))
    print("Columns:", len(enriched.columns))
    print("Preview:")
    print(enriched.head(8))

    enriched.to_csv(OUT_CSV, index=False)
    try:
        enriched.to_parquet(OUT_PARQ, index=False)
    except Exception as e:
        print("Parquet save skipped (install pyarrow or fastparquet to enable). Error:", e)

    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {OUT_PARQ} (if backend available)")

if __name__ == "__main__":
    main()
