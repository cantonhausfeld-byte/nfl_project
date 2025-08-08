import json
import pathlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, log_loss, roc_auc_score

IN_CSV  = "nfl_games_2020_2024_features.csv"
OUT_JSON = "calibrated_weights_2020_2024.json"

FEATURES = [
    # Recent performance diffs
    "roll3_pd_diff", "roll5_pd_diff", "roll3_pf_diff", "roll3_pa_diff",
    # Simple weather flags / nums
    "is_dome", "temp", "wind",
]

def load():
    df = pd.read_csv(IN_CSV)
    # ensure types
    for c in ["season","week","home_score","away_score","temp","wind"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_dome" in df.columns:
        df["is_dome"] = df["is_dome"].astype(int)
    # drop rows without target
    df = df.dropna(subset=["home_score","away_score"])
    df["point_diff"] = df["home_score"] - df["away_score"]
    # keep only rows with all features present (or fillna)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df

def split_time(df):
    train = df[df["season"] <= 2023].copy()
    valid = df[df["season"] == 2024].copy()
    return train, valid

def train_point_diff(train, valid):
    X_tr = train[FEATURES].values
    y_tr = train["point_diff"].values
    X_va = valid[FEATURES].values
    y_va = valid["point_diff"].values

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    pred_tr = lr.predict(X_tr)
    pred_va = lr.predict(X_va)

    metrics = {
        "train_mae": float(mean_absolute_error(y_tr, pred_tr)),
        "valid_mae": float(mean_absolute_error(y_va, pred_va)),
        "train_r2": float(r2_score(y_tr, pred_tr)),
        "valid_r2": float(r2_score(y_va, pred_va)),
        "coefficients": {f: float(c) for f, c in zip(FEATURES, lr.coef_)},
        "intercept": float(lr.intercept_),
    }
    return lr, metrics, pred_va

def train_home_win(train, valid):
    train = train.copy()
    valid = valid.copy()
    train["home_win"] = (train["point_diff"] > 0).astype(int)
    valid["home_win"] = (valid["point_diff"] > 0).astype(int)

    X_tr = train[FEATURES].values
    y_tr = train["home_win"].values
    X_va = valid[FEATURES].values
    y_va = valid["home_win"].values

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_tr, y_tr)
    p_tr = clf.predict_proba(X_tr)[:,1]
    p_va = clf.predict_proba(X_va)[:,1]

    yhat_tr = (p_tr >= 0.5).astype(int)
    yhat_va = (p_va >= 0.5).astype(int)

    metrics = {
        "train_acc": float(accuracy_score(y_tr, yhat_tr)),
        "valid_acc": float(accuracy_score(y_va, yhat_va)),
        "train_logloss": float(log_loss(y_tr, p_tr)),
        "valid_logloss": float(log_loss(y_va, p_va)),
        "train_roc_auc": float(roc_auc_score(y_tr, p_tr)) if len(np.unique(y_tr))>1 else None,
        "valid_roc_auc": float(roc_auc_score(y_va, p_va)) if len(np.unique(y_va))>1 else None,
        "coefficients": {f: float(c) for f, c in zip(FEATURES, clf.coef_[0])},
        "intercept": float(clf.intercept_[0]),
    }
    return clf, metrics, p_va

def main():
    df = load()
    train, valid = split_time(df)

    ols, ols_metrics, pred_va_diff = train_point_diff(train, valid)
    logi, logi_metrics, p_win_va = train_home_win(train, valid)

    print("\n=== OLS (Point Diff) ===")
    for k,v in ols_metrics.items():
        if k == "coefficients": continue
        print(f"{k}: {v}")
    print("coefficients:")
    for k,v in ols_metrics["coefficients"].items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Logistic (Home Win) ===")
    for k,v in logi_metrics.items():
        if k == "coefficients": continue
        print(f"{k}: {v}")
    print("coefficients:")
    for k,v in logi_metrics["coefficients"].items():
        print(f"  {k}: {v:.4f}")

    out = {
        "features": FEATURES,
        "ols": ols_metrics,
        "logistic": logi_metrics,
        "note": "Trained on 2020-2023, validated on 2024.",
    }
    pathlib.Path(OUT_JSON).write_text(json.dumps(out, indent=2))
    print(f"\nSaved calibration to {OUT_JSON}")

if __name__ == "__main__":
    main()
