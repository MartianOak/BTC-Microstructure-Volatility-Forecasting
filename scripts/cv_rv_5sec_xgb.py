#!/usr/bin/env python3
"""
cv_rv_5sec_xgb.py

Rolling time-series CV + random hyperparameter search + blend alpha tuning
for 5-second realized variance (RV) prediction on BTC 1-second LOB data.

Pipeline:

1. Load BTC_1sec_features.csv.
2. Target:
       y_rv_5s_log  (forward 5s RV in log-space, from feature_builder_1sec.py)
3. Baseline:
       log(rv_5s_past + eps)  (past 5s RV in log-space)
4. Hold out the last 15% of data as a FINAL TEST set.
5. On the first 85% ("development" data), run K-fold *expanding* CV:
     fold k:
       train: [0 : val_start_k)
       valid: [val_start_k : val_end_k)
6. For each random XGB hyperparameter config:
     - Train on each fold's train, predict on fold valid.
     - Collect baseline and XGB predictions.
     - On concatenated CV preds, search alpha in [0,1] (grid)
       to maximize Corr_rv_P (Pearson corr in RV space).
7. Choose (params, alpha) with best CV Corr_rv_P.
8. Refit best XGB on whole development data.
9. Evaluate baseline, pure XGB, and blend(alpha) on:
     - development segment
     - final TEST segment
10. Fit a **global** isotonic calibrator on DEV blended log-RV and
    evaluate calibrated blend on DEV + TEST.

Outputs:
    - Printed CV summary and TEST performance
    - best_rv_5s_cv_config.json
    - calibrator_rv_5s_isotonic_global.pkl
"""

import json
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.isotonic import IsotonicRegression


FEATURE_FILE = "BTC_1sec_features.csv"
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
    print("Log-space:")
    print(f"  RMSE_log: {np.sqrt(mse_log):.6f}")
    print(f"  MAE_log:  {mae_log:.6f}")
    print(f"  Corr_log (Pearson): {corr_log:.4f}")
    print("RV-space:")
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


def make_folds(n_dev, n_folds=3):
    """
    Create expanding time-series folds on [0, n_dev).

    For fold k:
      valid = [v_start : v_end)
      train = [0 : v_start)
    """
    folds = []
    fold_size = n_dev // (n_folds + 1)
    for k in range(1, n_folds + 1):
        v_start = k * fold_size
        v_end = (k + 1) * fold_size if k < n_folds else n_dev
        folds.append((0, v_start, v_start, v_end))
    return folds


def sample_param_grid(n_configs=10, random_state=42):
    """
    Random hyperparameter sampling for XGB.
    Adjust ranges if you want to push harder.
    """
    rng = np.random.RandomState(random_state)
    configs = []
    for _ in range(n_configs):
        cfg = {
            "n_estimators": int(rng.randint(300, 700)),
            "learning_rate": float(10 ** rng.uniform(-2.0, -0.7)),  # ~[0.01, 0.2]
            "max_depth": int(rng.randint(4, 9)),  # 4..8
            "subsample": float(rng.uniform(0.7, 1.0)),
            "colsample_bytree": float(rng.uniform(0.7, 1.0)),
            "min_child_weight": float(rng.choice([1.0, 5.0, 10.0, 20.0])),
        }
        configs.append(cfg)
    return configs


def build_xgb(params):
    """Create an XGBRegressor from a params dict (GPU-enabled)."""
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",   # you can switch to "gpu_hist" if supported
        device="cuda",
        eval_metric="rmse",
        n_jobs=-1,
        **params,
    )


def blend_and_metric(y_log_true, log_base, log_xgb, alphas):
    """
    For given arrays (true, baseline, xgb) in log-space, evaluate blending:

       y_pred_log(alpha) = alpha * log_base + (1 - alpha) * log_xgb

    Returns:
       best_alpha, metrics_by_alpha (dict list)
    """
    y_rv_true = np.exp(y_log_true) - EPS

    best_alpha = None
    best_corr = -np.inf
    metrics = []

    for alpha in alphas:
        y_log_blend = alpha * log_base + (1.0 - alpha) * log_xgb
        y_rv_blend = np.exp(y_log_blend) - EPS

        corr_p = np.corrcoef(y_rv_true, y_rv_blend)[0, 1]
        rmse_rv = np.sqrt(mean_squared_error(y_rv_true, y_rv_blend))

        metrics.append({
            "alpha": float(alpha),
            "Corr_P": float(corr_p),
            "RMSE_rv": float(rmse_rv),
        })

        if corr_p > best_corr:
            best_corr = corr_p
            best_alpha = alpha

    return best_alpha, metrics


# ---------- Main ----------

def main():
    print(f"Loading feature file: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)

    # Sort by time to be safe
    if "system_time" in df.columns:
        df["system_time"] = pd.to_datetime(df["system_time"], utc=True)
        df = df.sort_values("system_time").reset_index(drop=True)

    print(f"Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Target and baseline for 5s RV ---
    if "y_rv_5s_log" not in df.columns:
        raise ValueError("Column 'y_rv_5s_log' not found in features file.")
    if "rv_5s_past" not in df.columns:
        raise ValueError("Column 'rv_5s_past' not found (needed for baseline).")

    # Drop rows where either target or past-5s RV is NaN
    before = len(df)
    df = df.dropna(subset=["y_rv_5s_log", "rv_5s_past"]).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN y_rv_5s_log / rv_5s_past; remaining: {after}")

    n = len(df)

    # Holdout TEST = last 15%
    test_start = int(n * 0.85)
    dev = df.iloc[:test_start].copy()   # used for CV + final training
    test = df.iloc[test_start:].copy()  # untouched until the end

    print("\nGlobal split:")
    print(f"  DEV:  {len(dev)} rows (for CV + final fit)")
    print(f"  TEST: {len(test)} rows (final holdout)")

    # --- Baseline feature: log(rv_5s_past) ---
    dev_log_rv_past = np.log(dev["rv_5s_past"] + EPS)
    test_log_rv_past = np.log(test["rv_5s_past"] + EPS)

    # Targets
    y_dev = dev["y_rv_5s_log"].values
    y_test = test["y_rv_5s_log"].values

    # --- Define feature columns (exclude time & explicit targets) ---
    exclude_cols = {
        "system_time",
        "y_rv_1s", "y_rv_1s_log",
        "y_rv_5s", "y_rv_5s_log",
        "y_rv_10s", "y_rv_10s_log",
        # safety: if 1s CV script was run and added future targets
        "y_rv_1s_future", "y_rv_1s_future_log",
    }

    feature_cols = [c for c in dev.columns if c not in exclude_cols]
    print(f"\nUsing {len(feature_cols)} features for 5s RV.")

    X_dev = dev[feature_cols].values
    X_test = test[feature_cols].values

    # ---------- CV setup ----------
    n_dev = len(dev)
    n_folds = 3
    folds = make_folds(n_dev, n_folds=n_folds)
    print(f"\nUsing {n_folds}-fold expanding time-series CV on DEV:")
    for i, (tr_start, tr_end, v_start, v_end) in enumerate(folds):
        print(f"  Fold {i+1}: train=[{tr_start}:{tr_end}), valid=[{v_start}:{v_end})")

    # ---------- Random hyperparameter search ----------
    param_grid = sample_param_grid(n_configs=10, random_state=42)
    alpha_grid = np.linspace(0.0, 1.0, 21)  # 0, 0.05, ..., 1.0

    best_overall = {
        "params": None,
        "alpha": None,
        "cv_corr": -np.inf,
        "cv_rmse_rv": None,
    }

    print("\n===== Starting CV hyperparameter + alpha search =====")
    for cfg_idx, params in enumerate(param_grid):
        print(f"\n--- Config {cfg_idx+1}/{len(param_grid)} ---")
        print(params)

        # Collect CV predictions
        cv_y_true = []
        cv_log_base = []
        cv_log_xgb = []

        for fold_idx, (tr_start, tr_end, v_start, v_end) in enumerate(folds):
            print(f"  Fold {fold_idx+1}: training on [{tr_start}:{tr_end}), validating on [{v_start}:{v_end})")

            X_tr = X_dev[tr_start:tr_end]
            y_tr = y_dev[tr_start:tr_end]

            X_val = X_dev[v_start:v_end]
            y_val = y_dev[v_start:v_end]

            # Baseline
            log_base_val = dev_log_rv_past.iloc[v_start:v_end].values

            # Train XGB
            model = build_xgb(params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            log_xgb_val = model.predict(X_val)

            cv_y_true.append(y_val)
            cv_log_base.append(log_base_val)
            cv_log_xgb.append(log_xgb_val)

        # Concatenate over folds
        cv_y_true = np.concatenate(cv_y_true)
        cv_log_base = np.concatenate(cv_log_base)
        cv_log_xgb = np.concatenate(cv_log_xgb)

        # Find best alpha for this config on CV
        best_alpha_cfg, alpha_metrics = blend_and_metric(
            cv_y_true, cv_log_base, cv_log_xgb, alpha_grid
        )

        # Get metrics for that alpha
        y_rv_true = np.exp(cv_y_true) - EPS
        y_rv_blend = np.exp(
            best_alpha_cfg * cv_log_base + (1.0 - best_alpha_cfg) * cv_log_xgb
        ) - EPS

        corr_cv = np.corrcoef(y_rv_true, y_rv_blend)[0, 1]
        rmse_cv = np.sqrt(mean_squared_error(y_rv_true, y_rv_blend))

        print(f"  -> Best alpha (by CV Corr_P): {best_alpha_cfg:.2f}")
        print(f"     CV Corr_rv_P: {corr_cv:.4f}, CV RMSE_rv: {rmse_cv:.4e}")

        if corr_cv > best_overall["cv_corr"]:
            best_overall["params"] = params
            best_overall["alpha"] = float(best_alpha_cfg)
            best_overall["cv_corr"] = float(corr_cv)
            best_overall["cv_rmse_rv"] = float(rmse_cv)

    print("\n===== CV search finished =====")
    print("Best config found:")
    print(json.dumps(best_overall, indent=2))

    # ---------- Final training on full DEV ----------
    print("\n=== Training final model on full DEV with best params ===")
    best_params = best_overall["params"]
    best_alpha = best_overall["alpha"]

    final_model = build_xgb(best_params)
    final_model.fit(
        X_dev,
        y_dev,
        eval_set=[(X_dev, y_dev)],
        verbose=False,
    )

    # Baseline & XGB & blend on DEV
    log_base_dev = dev_log_rv_past.values
    log_xgb_dev = final_model.predict(X_dev)
    log_blend_dev = best_alpha * log_base_dev + (1.0 - best_alpha) * log_xgb_dev

    _ = eval_log_and_rv(y_dev, log_base_dev, name="DEV (baseline)")
    _ = eval_log_and_rv(y_dev, log_xgb_dev, name="DEV (XGB)")
    _ = eval_log_and_rv(y_dev, log_blend_dev, name=f"DEV (blend, alpha={best_alpha:.2f})")

    # ---------- Final evaluation on TEST ----------
    print("\n=== Final evaluation on TEST (holdout) ===")

    # Baseline on TEST
    log_base_test = test_log_rv_past.values

    # XGB on TEST
    log_xgb_test = final_model.predict(X_test)

    # Blended
    log_blend_test = best_alpha * log_base_test + (1.0 - best_alpha) * log_xgb_test

    res_base = eval_log_and_rv(y_test, log_base_test, name="TEST (baseline)")
    res_xgb = eval_log_and_rv(y_test, log_xgb_test, name="TEST (XGB)")
    res_blend = eval_log_and_rv(y_test, log_blend_test, name=f"TEST (blend, alpha={best_alpha:.2f})")

    summary = {
        "best_params": best_params,
        "best_alpha": best_alpha,
        "cv_corr_rv_P": best_overall["cv_corr"],
        "cv_rmse_rv": best_overall["cv_rmse_rv"],
        "test_baseline": res_base,
        "test_xgb": res_xgb,
        "test_blend": res_blend,
    }

    with open("best_rv_5s_cv_config.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved best_rv_5s_cv_config.json")

    # ---------- Global isotonic calibration on blended predictions ----------
    print("\n=== GLOBAL isotonic calibration on blended predictions ===")

    iso = IsotonicRegression(out_of_bounds="clip")
    print("Fitting GLOBAL IsotonicRegression calibrator on DEV...")
    iso.fit(log_blend_dev, y_dev)

    log_cal_dev = iso.predict(log_blend_dev)
    log_cal_test = iso.predict(log_blend_test)

    _ = eval_log_and_rv(y_dev, log_cal_dev, name="DEV (global calibrated blend)")
    _ = eval_log_and_rv(y_test, log_cal_test, name="TEST (global calibrated blend)")

    with open("calibrator_rv_5s_isotonic_global.pkl", "wb") as f:
        pickle.dump(iso, f)
    
    print("Saved global calibrator to calibrator_rv_5s_isotonic_global.pkl")


if __name__ == "__main__":
    main()
