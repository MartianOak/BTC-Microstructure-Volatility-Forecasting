#!/usr/bin/env python3
"""
train_rv_5s_xgb.py

Train an XGBoost model to predict 5-second realized variance (RV)
using BTC 1-second limit order book features.

Target:
    y_rv_5s_log  (forward 5s RV in log-space, produced by feature_builder_1sec.py)

Baseline:
    log(rv_5s_past + eps)  (past 5s RV in log-space)

Input:
    BTC_1sec_features.csv

Outputs:
    - Printed metrics for baseline and XGB (TRAIN/VALID/TEST)
    - feature_importance_rv_5s_xgb.csv
    - rv_5s_pred_vs_true_test.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


FEATURE_FILE = "BTC_1sec_features.csv"


# ---------- Helpers ----------

def spearman_corr(a, b):
    a_rank = pd.Series(a).rank()
    b_rank = pd.Series(b).rank()
    return a_rank.corr(b_rank)


def eval_log_and_rv(y_log_true, y_log_pred, name="set"):
    eps = 1e-18

    mse_log = mean_squared_error(y_log_true, y_log_pred)
    mae_log = mean_absolute_error(y_log_true, y_log_pred)
    corr_log = np.corrcoef(y_log_true, y_log_pred)[0, 1]

    # Convert back to RV space
    y_rv_true = np.exp(y_log_true) - eps
    y_rv_pred = np.exp(y_log_pred) - eps

    mse_rv = mean_squared_error(y_rv_true, y_rv_pred)
    mae_rv = mean_absolute_error(y_rv_true, y_rv_pred)
    corr_rv_p = np.corrcoef(y_rv_true, y_rv_pred)[0, 1]
    corr_rv_s = spearman_corr(y_rv_true, y_rv_pred)

    print(f"\n=== Performance on {name} ===")
    print("Log-space (y_rv_5s_log):")
    print(f"  RMSE_log: {np.sqrt(mse_log):.6f}")
    print(f"  MAE_log:  {mae_log:.6f}")
    print(f"  Corr_log (Pearson): {corr_log:.4f}")
    print("RV-space (y_rv_5s):")
    print(f"  RMSE_rv:   {np.sqrt(mse_rv):.6e}")
    print(f"  MAE_rv:    {mae_rv:.6e}")
    print(f"  Corr_rv_P: {corr_rv_p:.4f}")
    print(f"  Corr_rv_S: {corr_rv_s:.4f}")

    return {
        "RMSE_log": np.sqrt(mse_log),
        "MAE_log": mae_log,
        "RMSE_rv": np.sqrt(mse_rv),
        "MAE_rv": mae_rv,
        "Corr_log": corr_log,
        "Corr_P": corr_rv_p,
        "Corr_S": corr_rv_s,
    }


def build_xgb():
    # Reasonable starting hyperparameters for 5s horizon; tune later if needed
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=10,
        objective="reg:squarederror",
        tree_method="hist",
        device = "cuda",
        eval_metric="rmse",
        n_jobs=-1,
    )


# ---------- Main ----------

def main():
    print(f"Loading feature file: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)

    # Parse time and sort to be safe
    if "system_time" in df.columns:
        df["system_time"] = pd.to_datetime(df["system_time"], utc=True)
        df = df.sort_values("system_time").reset_index(drop=True)

    print(f"Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")

    # We use forward 5s RV produced by the feature builder: y_rv_5s_log
    # And baseline based on rv_5s_past
    eps = 1e-18

    if "y_rv_5s_log" not in df.columns:
        raise ValueError("Column 'y_rv_5s_log' not found in features file.")
    if "rv_5s_past" not in df.columns:
        raise ValueError("Column 'rv_5s_past' not found (needed for baseline).")

    # Drop rows where either target OR past-5s RV is NaN
    before = len(df)
    df = df.dropna(subset=["y_rv_5s_log", "rv_5s_past"]).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN y_rv_5s_log / rv_5s_past; remaining: {after}")

    # Time-based split: 70% train, 15% valid, 15% test
    n = len(df)
    train_end = int(n * 0.7)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    test = df.iloc[valid_end:]

    print("\nSplit sizes:")
    print(f"  Train: {len(train)}")
    print(f"  Valid: {len(valid)}")
    print(f"  Test:  {len(test)}")

    # Features: exclude time and ALL y_rv_* targets (1s/5s/10s etc.) to avoid leakage.
    exclude_cols = {
        "system_time",
        "y_rv_5s_log",   # target (log)
        "y_rv_5s",       # raw target if present
    }
    # Remove any other y_rv_* columns that might exist (e.g. y_rv_1s, y_rv_10s, etc.)
    exclude_cols |= {c for c in df.columns if c.startswith("y_rv_")}

    # We keep rv_5s_past as a legitimate predictor (it's past-only)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\nUsing {len(feature_cols)} features.")
    # Uncomment to inspect:
    # for c in feature_cols:
    #     print("  ", c)

    X_train = train[feature_cols].values
    y_train = train["y_rv_5s_log"].values

    X_valid = valid[feature_cols].values
    y_valid = valid["y_rv_5s_log"].values

    X_test = test[feature_cols].values
    y_test = test["y_rv_5s_log"].values

    # --- Baseline: use log(rv_5s_past) to predict y_rv_5s_log ---
    rv_5s_past_all = df["rv_5s_past"].values
    rv_5s_past_log_all = np.log(rv_5s_past_all + eps)

    baseline_train = rv_5s_past_log_all[:train_end]
    baseline_valid = rv_5s_past_log_all[train_end:valid_end]
    baseline_test = rv_5s_past_log_all[valid_end:]

    print("\n=== Baseline (log(rv_5s_past) -> y_rv_5s_log) ===")
    _ = eval_log_and_rv(y_train, baseline_train, name="TRAIN (baseline)")
    _ = eval_log_and_rv(y_valid, baseline_valid, name="VALID (baseline)")
    _ = eval_log_and_rv(y_test, baseline_test, name="TEST (baseline)")

    # --- Train XGBoost model ---
    print("\n=== Training XGBoost model for 5s RV ===")
    model = build_xgb()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,  # print eval RMSE every 50 trees
    )

    print("\n=== Evaluating XGBoost ===")
    # Train
    y_train_pred = model.predict(X_train)
    res_train = eval_log_and_rv(y_train, y_train_pred, name="TRAIN (XGB)")

    # Valid
    y_valid_pred = model.predict(X_valid)
    res_valid = eval_log_and_rv(y_valid, y_valid_pred, name="VALID (XGB)")

    # Test
    y_test_pred = model.predict(X_test)
    res_test = eval_log_and_rv(y_test, y_test_pred, name="TEST (XGB)")

    # --- Feature importance ---
    print("\n=== Feature Importance (gain) ===")
    importance = model.get_booster().get_score(importance_type="gain")
    rows = []
    for i, col in enumerate(feature_cols):
        key = f"f{i}"
        gain = importance.get(key, 0.0)
        rows.append((col, gain))
    fi_df = pd.DataFrame(rows, columns=["feature", "gain"]).sort_values(
        "gain", ascending=False
    )
    print(fi_df.head(30))
    fi_df.to_csv("feature_importance_rv_5s_xgb.csv", index=False)
    print("Saved feature_importance_rv_5s_xgb.csv")

    # --- Plot predictions vs true (log-space) on TEST ---
    print("Saving rv_5s_pred_vs_true_test.png ...")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred, s=2, alpha=0.3)
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("True y_rv_5s_log")
    plt.ylabel("Predicted y_rv_5s_log")
    plt.title("XGB: True vs Predicted 5s log RV (TEST)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rv_5s_pred_vs_true_test.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
