#!/usr/bin/env python3
"""
calibrate_rv_5s.py

Post-hoc calibration for 5-second RV predictions.

Pipeline:
  1) Load BTC_1sec_features.csv.
  2) Recreate the 5s DEV / TEST split used in cv_rv_5sec_xgb.py:
       - Drop rows with NaN in y_rv_5s_log or rv_5s_past.
       - DEV = first 85%, TEST = last 15%.
  3) Load best 5s CV config from best_rv_5s_cv_config.json:
       - best_params (XGB hyperparams)
       - best_alpha (blend weight for baseline vs XGB)
  4) Train final XGB on DEV only.
  5) For each DEV point:
       y_true = y_rv_5s_log
       log_base = log(rv_5s_past + eps)
       log_xgb  = XGB prediction
       log_raw  = alpha * log_base + (1 - alpha) * log_xgb
     Fit IsotonicRegression: log_raw -> y_calibrated (in log-space).
  6) Apply calibrator to DEV and TEST predictions.
  7) Compare metrics (raw blend vs calibrated blend) on DEV and TEST.
  8) Save:
       - calibrator_rv_5s_isotonic.pkl
       - calibrator_rv_5s_isotonic.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


FEATURE_FILE = "BTC_1sec_features.csv"
CV_CONFIG_FILE = "best_rv_5s_cv_config.json"
EPS = 1e-18


# ---------- Helpers ----------

def spearman_corr(a, b):
    a_rank = pd.Series(a).rank()
    b_rank = pd.Series(b).rank()
    return a_rank.corr(b_rank)


def eval_log_and_rv(y_log_true, y_log_pred, name="set"):
    """Evaluate metrics in log-space and RV-space."""
    mse_log = mean_squared_error(y_log_true, y_log_pred)
    mae_log = mean_absolute_error(y_log_true, y_log_pred)
    corr_log = np.corrcoef(y_log_true, y_log_pred)[0, 1]

    # Back to RV space
    y_rv_true = np.exp(y_log_true) - EPS
    y_rv_pred = np.exp(y_log_pred) - EPS

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
        "RMSE_log": float(np.sqrt(mse_log)),
        "MAE_log": float(mae_log),
        "RMSE_rv": float(np.sqrt(mse_rv)),
        "MAE_rv": float(mae_rv),
        "Corr_log": float(corr_log),
        "Corr_P": float(corr_rv_p),
        "Corr_S": float(corr_rv_s),
    }


def build_xgb(params):
    """Create an XGBRegressor from a params dict."""
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",          # GPU
        eval_metric="rmse",
        n_jobs=-1,
        **params,
    )


# ---------- Main ----------

def main():
    # 1) Load features
    print(f"Loading feature file: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)

    # Sort by time to be safe
    if "system_time" in df.columns:
        df["system_time"] = pd.to_datetime(df["system_time"], utc=True)
        df = df.sort_values("system_time").reset_index(drop=True)

    print(f"Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2) Recreate y_rv_5s_log / rv_5s_past split as in cv_rv_5sec_xgb.py
    if "y_rv_5s_log" not in df.columns:
        raise ValueError("Column 'y_rv_5s_log' not found in features file.")
    if "rv_5s_past" not in df.columns:
        raise ValueError("Column 'rv_5s_past' not found in features file.")

    before = len(df)
    df = df.dropna(subset=["y_rv_5s_log", "rv_5s_past"]).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN y_rv_5s_log / rv_5s_past; remaining: {after}")

    n = len(df)
    dev_end = int(n * 0.85)

    dev = df.iloc[:dev_end].copy()
    test = df.iloc[dev_end:].copy()

    print("\nGlobal split (for calibration):")
    print(f"  DEV:  {len(dev)} rows")
    print(f"  TEST: {len(test)} rows")

    # Targets
    y_dev = dev["y_rv_5s_log"].values
    y_test = test["y_rv_5s_log"].values

    # Baseline log(rv_5s_past)
    log_base_dev = np.log(dev["rv_5s_past"].values + EPS)
    log_base_test = np.log(test["rv_5s_past"].values + EPS)

    # 3) Load CV best config (params + alpha)
    print(f"\nLoading CV config from {CV_CONFIG_FILE}")
    with open(CV_CONFIG_FILE, "r") as f:
        cfg = json.load(f)

    # Be robust to two possible key patterns
    if "best_params" in cfg:
        best_params = cfg["best_params"]
        best_alpha = cfg.get("best_alpha", 0.35)
    else:
        best_params = cfg["params"]
        best_alpha = cfg.get("alpha", 0.35)

    print("Best params:", best_params)
    print(f"Best alpha (blend weight for baseline): {best_alpha:.2f}")

    # 4) Define features (same exclusions as CV 5s script)
    exclude_cols = {
        "system_time",
        "y_rv_1s", "y_rv_1s_log",
        "y_rv_5s", "y_rv_5s_log",
        "y_rv_10s", "y_rv_10s_log",
        "y_rv_1s_future", "y_rv_1s_future_log",
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"\nUsing {len(feature_cols)} features for 5s RV calibration.")

    X_dev = dev[feature_cols].values
    X_test = test[feature_cols].values

    # 5) Train XGB on DEV with best params
    print("\nTraining final 5s XGB on DEV with best params...")
    model = build_xgb(best_params)
    model.fit(
        X_dev,
        y_dev,
        eval_set=[(X_dev, y_dev)],
        verbose=False,
    )

    # 6) Compute raw blended log-RV on DEV and TEST
    log_xgb_dev = model.predict(X_dev)
    log_xgb_test = model.predict(X_test)

    log_raw_dev = best_alpha * log_base_dev + (1.0 - best_alpha) * log_xgb_dev
    log_raw_test = best_alpha * log_base_test + (1.0 - best_alpha) * log_xgb_test

    print("\n=== RAW BLEND PERFORMANCE (before calibration) ===")
    eval_log_and_rv(y_dev, log_raw_dev, name="DEV (raw blend)")
    eval_log_and_rv(y_test, log_raw_test, name="TEST (raw blend)")

    # 7) Fit IsotonicRegression calibrator on DEV
    print("\nFitting IsotonicRegression calibrator on DEV...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(log_raw_dev, y_dev)

    # Apply calibrator
    log_cal_dev = iso.predict(log_raw_dev)
    log_cal_test = iso.predict(log_raw_test)

    print("\n=== CALIBRATED BLEND PERFORMANCE (after calibration) ===")
    eval_log_and_rv(y_dev, log_cal_dev, name="DEV (calibrated blend)")
    eval_log_and_rv(y_test, log_cal_test, name="TEST (calibrated blend)")

    # 8) Save calibrator (joblib so backtest can use joblib.load)
    joblib.dump(iso, "calibrator_rv_5s_isotonic.pkl")
    print("\nSaved calibrator to calibrator_rv_5s_isotonic.pkl")

    # 9) Diagnostic plot: mapping raw -> calibrated vs true (on DEV)
    print("Saving calibrator_rv_5s_isotonic.png ...")
    plt.figure(figsize=(6, 6))
    # Scatter: raw vs true (downsampled for plotting)
    plt.scatter(
        log_raw_dev[::100],
        y_dev[::100],
        s=5,
        alpha=0.3,
        label="DEV (raw vs true)",
    )
    # Isotonic mapping line
    order = np.argsort(log_raw_dev)
    plt.plot(
        log_raw_dev[order],
        iso.predict(log_raw_dev[order]),
        linewidth=2,
        label="Isotonic mapping (raw -> calibrated)",
    )
    plt.xlabel("Raw blended log-RV (log_raw)")
    plt.ylabel("Target log-RV (y_rv_5s_log)")
    plt.title("Isotonic calibrator for 5s log-RV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("calibrator_rv_5s_isotonic.png")
    plt.close()
    print("Done. Diagnostic plot saved as calibrator_rv_5s_isotonic.png")

    # 10) Brief summary (TEST, RV-space)
    print("\n=== SUMMARY (TEST, RV-space) ===")
    y_rv_test = np.exp(y_test) - EPS
    y_rv_raw = np.exp(log_raw_test) - EPS
    y_rv_cal = np.exp(log_cal_test) - EPS

    rmse_raw = np.sqrt(mean_squared_error(y_rv_test, y_rv_raw))
    rmse_cal = np.sqrt(mean_squared_error(y_rv_test, y_rv_cal))
    corr_raw = np.corrcoef(y_rv_test, y_rv_raw)[0, 1]
    corr_cal = np.corrcoef(y_rv_test, y_rv_cal)[0, 1]

    print(f"Raw blend:    Corr_P={corr_raw:.4f}, RMSE_rv={rmse_raw:.3e}")
    print(f"Calibrated:   Corr_P={corr_cal:.4f}, RMSE_rv={rmse_cal:.3e}")


if __name__ == "__main__":
    main()
