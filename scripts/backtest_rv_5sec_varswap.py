#!/usr/bin/env python3
"""
backtest_rv_5sec_varswap.py

Toy "variance-swap style" backtest for 5-second realized variance (RV),
using the calibrated 5s RV model you already built.

Setup (matches cv_rv_5sec_xgb.py + calibrate_rv_5s.py):

- Input features:
    BTC_1sec_features.csv

- Target:
    y_rv_5s_log  (forward 5s RV in log-space, from feature_builder_1sec.py)

- Baseline:
    rv_5s_past   (realized RV over previous 5s window)
    log_baseline = log(rv_5s_past + eps)

- Model:
    XGB with best hyperparameters from best_rv_5s_cv_config.json
    Blend in log-space:
        log_blend = alpha * log_baseline + (1-alpha) * log_pred_xgb
    Then global isotonic calibrator (fitted on DEV):
        log_calibrated = iso(log_blend)

- Data split:
    DEV  = first 85% of rows (used for CV + final fit + calibrator fit)
    TEST = last 15% of rows (pure holdout)

Trading toys:

1) Simple sign varswap:
       pos_t = sign(forecast - baseline)

2) Magnitude-aware varswap with position sizing:
       edge_rel = (forecast - baseline) / (baseline + eps)
       if |edge_rel| < edge_min: pos = 0
       else: pos = clip(edge_rel / s_edge, -L_MAX, L_MAX)
       then regime scaling via rv_regime and gamma_map.

We then:

- Evaluate forecasts (baseline, XGB, blend, calibrated).
- Run simple sign varswap on DEV/TEST.
- Run magnitude-aware varswap once with default params.
- Run DEV-only grid search over (s_edge, edge_min, L_MAX) for the
  calibrated blend, pick the best Sharpe_5s, and apply that config to TEST.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURE_FILE = "BTC_1sec_features.csv"
CV_CONFIG_FILE = "best_rv_5s_cv_config.json"
CALIB_FILE = "calibrator_rv_5s_isotonic.pkl"
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

    print(f"\n=== Forecast performance on {name} ===")
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


def build_xgb(params):
    """Create an XGBRegressor from a params dict (GPU-enabled)."""
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        eval_metric="rmse",
        n_jobs=-1,
        **params,
    )


def pnl_stats(pnl, name="strategy"):
    """Compute simple per-5s PnL stats."""
    pnl = np.asarray(pnl)
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=0))
    sharpe = float(mean / std) if std > 0 else np.nan  # per-5s Sharpe
    hit_ratio = float((pnl > 0).mean())
    total = float(pnl.sum())
    print(f"\n=== PnL stats for {name} ===")
    print(f"  mean_pnl:    {mean:.6e}")
    print(f"  std_pnl:     {std:.6e}")
    print(f"  sharpe_5s:   {sharpe:.4f}  (per 5-second step)")
    print(f"  hit_ratio:   {hit_ratio:.4f}")
    print(f"  total_pnl:   {total:.6e}")
    return {
        "mean": mean,
        "std": std,
        "sharpe_5s": sharpe,
        "hit_ratio": hit_ratio,
        "total": total,
    }


def varswap_pnl(sigma2_real, sigma2_base, sigma2_forecast):
    """
    Simple variance-swap-style PnL per step for a given forecast.

        signal_t = forecast - baseline
        pos_t    = sign(signal_t)
        pnl_t    = pos_t * (realized - baseline)

    Inputs should all be in RV space (not log).
    """
    sigma2_real = np.asarray(sigma2_real)
    sigma2_base = np.asarray(sigma2_base)
    sigma2_forecast = np.asarray(sigma2_forecast)

    signal = sigma2_forecast - sigma2_base
    pos = np.sign(signal)
    pnl = pos * (sigma2_real - sigma2_base)
    return pnl


def varswap_pnl_with_pos(sigma2_real, sigma2_base, pos):
    """
    Variance-swap-style PnL given an externally provided position series:

        pnl_t = pos_t * (sigma2_real_t - sigma2_base_t)
    """
    sigma2_real = np.asarray(sigma2_real)
    sigma2_base = np.asarray(sigma2_base)
    pos = np.asarray(pos)
    pnl = pos * (sigma2_real - sigma2_base)
    return pnl


def build_magaware_positions(
    rv_forecast,
    rv_baseline,
    rv_regime=None,
    s_edge=0.20,
    edge_min=0.05,
    L_max=1.5,
    gamma_map=None,
):
    """
    Magnitude-aware, regime-aware position sizing.

    Inputs:
        rv_forecast : array-like, model forecast (RV space).
        rv_baseline : array-like, baseline "strike" (RV space).
        rv_regime   : array-like or None, integer volatility regime labels.
        s_edge      : scale where edge_rel/s_edge = 1 means full size.
        edge_min    : minimum |edge_rel| to take any position.
        L_max       : cap on absolute position size.
        gamma_map   : dict mapping regime -> multiplier (e.g. {0:1.0,1:0.7,2:0.4}).

    Returns:
        pos : np.ndarray of positions.
    """
    rv_forecast = np.asarray(rv_forecast)
    rv_baseline = np.asarray(rv_baseline)

    # Relative edge
    edge_rel = (rv_forecast - rv_baseline) / (rv_baseline + EPS)

    pos = np.zeros_like(edge_rel)

    # Only trade when |edge| >= edge_min
    mask = np.abs(edge_rel) >= edge_min
    pos[mask] = edge_rel[mask] / s_edge
    pos = np.clip(pos, -L_max, L_max)

    # Regime scaling
    if rv_regime is not None and gamma_map is not None:
        rv_regime = np.asarray(rv_regime)
        gammas = np.ones_like(pos, dtype=float)
        for reg, g in gamma_map.items():
            gammas[rv_regime == reg] = g
        pos = pos * gammas

    return pos


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

    before = len(df)
    df = df.dropna(subset=["y_rv_5s_log", "rv_5s_past"]).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN y_rv_5s_log / rv_5s_past; remaining: {after}")

    n = len(df)

    # Global DEV / TEST split (must match cv script)
    test_start = int(n * 0.85)
    dev = df.iloc[:test_start].copy()
    test = df.iloc[test_start:].copy()

    print("\nGlobal split:")
    print(f"  DEV:  {len(dev)} rows")
    print(f"  TEST: {len(test)} rows")

    # Baseline feature: log(rv_5s_past)
    dev_log_rv_past = np.log(dev["rv_5s_past"] + EPS)
    test_log_rv_past = np.log(test["rv_5s_past"] + EPS)

    # Targets (log-RV) and true RV in 5s space
    y_dev = dev["y_rv_5s_log"].values
    y_test = test["y_rv_5s_log"].values
    rv_dev_true = np.exp(y_dev) - EPS
    rv_test_true = np.exp(y_test) - EPS

    # --------- Load best CV config + calibrator ---------
    print(f"\nLoading CV config from {CV_CONFIG_FILE}")
    with open(CV_CONFIG_FILE, "r") as f:
        cv_conf = json.load(f)

    best_params = cv_conf["best_params"]
    best_alpha = float(cv_conf["best_alpha"])
    print("Best params:", best_params)
    print("Best alpha (blend weight for baseline):", best_alpha)

    print(f"\nLoading global isotonic calibrator from {CALIB_FILE}")
    iso = joblib.load(CALIB_FILE)


    # --------- Define feature columns (match cv script) ---------
    exclude_cols = {
        "system_time",
        "y_rv_1s", "y_rv_1s_log",
        "y_rv_5s", "y_rv_5s_log",
        "y_rv_10s", "y_rv_10s_log",
        "y_rv_1s_future", "y_rv_1s_future_log",
    }
    feature_cols = [c for c in dev.columns if c not in exclude_cols]
    print(f"\nUsing {len(feature_cols)} features for 5s RV.")

    X_dev = dev[feature_cols].values
    X_test = test[feature_cols].values

    # Regime column (if available)
    dev_regime = dev["rv_regime"].values if "rv_regime" in dev.columns else None
    test_regime = test["rv_regime"].values if "rv_regime" in test.columns else None

    # --------- Train final XGB on DEV ---------
    print("\nTraining final 5s XGB on DEV with best params (GPU if available)...")
    model = build_xgb(best_params)
    model.fit(
        X_dev,
        y_dev,
        eval_set=[(X_dev, y_dev)],
        verbose=False,
    )

    # --------- FORECASTS: DEV ---------
    log_base_dev = dev_log_rv_past.values
    log_xgb_dev = model.predict(X_dev)
    log_blend_dev = best_alpha * log_base_dev + (1.0 - best_alpha) * log_xgb_dev
    log_cal_dev = iso.predict(log_blend_dev)

    # Forecast evaluation on DEV
    _ = eval_log_and_rv(y_dev, log_base_dev, name="DEV (baseline forecast)")
    _ = eval_log_and_rv(y_dev, log_xgb_dev, name="DEV (XGB forecast)")
    _ = eval_log_and_rv(y_dev, log_blend_dev, name=f"DEV (blend, alpha={best_alpha:.2f})")
    _ = eval_log_and_rv(y_dev, log_cal_dev, name="DEV (blend + global isotonic)")

    # convert to RV space
    rv_dev_base = np.exp(log_base_dev) - EPS
    rv_dev_xgb = np.exp(log_xgb_dev) - EPS
    rv_dev_blend = np.exp(log_blend_dev) - EPS
    rv_dev_cal = np.exp(log_cal_dev) - EPS

    # --------- FORECASTS: TEST ---------
    log_base_test = test_log_rv_past.values
    log_xgb_test = model.predict(X_test)
    log_blend_test = best_alpha * log_base_test + (1.0 - best_alpha) * log_xgb_test
    log_cal_test = iso.predict(log_blend_test)

    _ = eval_log_and_rv(y_test, log_base_test, name="TEST (baseline forecast)")
    _ = eval_log_and_rv(y_test, log_xgb_test, name="TEST (XGB forecast)")
    _ = eval_log_and_rv(y_test, log_blend_test, name=f"TEST (blend, alpha={best_alpha:.2f})")
    _ = eval_log_and_rv(y_test, log_cal_test, name="TEST (blend + global isotonic)")

    rv_test_base = np.exp(log_base_test) - EPS
    rv_test_xgb = np.exp(log_xgb_test) - EPS
    rv_test_blend = np.exp(log_blend_test) - EPS
    rv_test_cal = np.exp(log_cal_test) - EPS

    # --------- VARIANCE-SWAP-STYLE BACKTEST (SIMPLE SIGN) ---------
    print("\n========== VARIANCE-SWAP STYLE BACKTEST (SIGN STRATEGY) ==========")
    print("PnL definition:")
    print("  pos_t = sign( sigma2_forecast_t - sigma2_baseline_t )")
    print("  PnL_t = pos_t * ( sigma2_realized_t - sigma2_baseline_t )")
    print("where baseline = rv_5s_past, forecast in {XGB, blend, calibrated}.\n")

    # DEV PnL (simple sign)
    pnl_dev_xgb = varswap_pnl(rv_dev_true, rv_dev_base, rv_dev_xgb)
    pnl_dev_blend = varswap_pnl(rv_dev_true, rv_dev_base, rv_dev_blend)
    pnl_dev_cal = varswap_pnl(rv_dev_true, rv_dev_base, rv_dev_cal)

    stats_dev_xgb = pnl_stats(pnl_dev_xgb, name="DEV (XGB varswap)")
    stats_dev_blend = pnl_stats(pnl_dev_blend, name="DEV (blend varswap)")
    stats_dev_cal = pnl_stats(pnl_dev_cal, name="DEV (calibrated blend varswap)")

    # TEST PnL (simple sign)
    pnl_test_xgb = varswap_pnl(rv_test_true, rv_test_base, rv_test_xgb)
    pnl_test_blend = varswap_pnl(rv_test_true, rv_test_base, rv_test_blend)
    pnl_test_cal = varswap_pnl(rv_test_true, rv_test_base, rv_test_cal)

    stats_test_xgb = pnl_stats(pnl_test_xgb, name="TEST (XGB varswap)")
    stats_test_blend = pnl_stats(pnl_test_blend, name="TEST (blend varswap)")
    stats_test_cal = pnl_stats(pnl_test_cal, name="TEST (calibrated blend varswap)")

    print("\nSaving cumulative PnL plot for TEST (simple sign) as varswap_pnl_5s_test.png ...")
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pnl_test_xgb), label="XGB varswap (sign)")
    plt.plot(np.cumsum(pnl_test_blend), label=f"Blend varswap (sign, alpha={best_alpha:.2f})")
    plt.plot(np.cumsum(pnl_test_cal), label="Calibrated blend varswap (sign)")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Variance-swap style cumulative PnL (5s RV, TEST, sign strategy)")
    plt.xlabel("5-second steps (TEST)")
    plt.ylabel("Cumulative PnL (RV units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("varswap_pnl_5s_test.png")
    plt.close()
    print("Done simple PnL plot.")

    # --------- MAGNITUDE-AWARE VARIANCE-SWAP STRATEGY (SINGLE RUN) ---------
    print("\n========== MAGNITUDE-AWARE VARIANCE-SWAP STRATEGY ==========")

    # Default gamma_map for regimes 0,1,2
    default_gamma_map = {0: 1.0, 1: 0.7, 2: 0.4}

    if dev_regime is not None:
        print("  [DEV] Using rv_regime for regime-aware scaling.")
    else:
        print("  [DEV] No rv_regime column; using gamma = 1.0 everywhere.")
        default_gamma_map = None

    if test_regime is not None:
        print("  [TEST] Using rv_regime for regime-aware scaling.")
    else:
        print("  [TEST] No rv_regime column; using gamma = 1.0 everywhere.")

    # Default parameters (the ones we previously used)
    s_edge_default = 0.20
    edge_min_default = 0.05
    L_max_default = 1.5

    pos_dev_mag = build_magaware_positions(
        rv_forecast=rv_dev_cal,
        rv_baseline=rv_dev_base,
        rv_regime=dev_regime,
        s_edge=s_edge_default,
        edge_min=edge_min_default,
        L_max=L_max_default,
        gamma_map=default_gamma_map,
    )
    pnl_dev_mag = varswap_pnl_with_pos(rv_dev_true, rv_dev_base, pos_dev_mag)
    stats_dev_mag = pnl_stats(pnl_dev_mag, name="DEV (calibrated blend magnitude-aware varswap)")

    pos_test_mag = build_magaware_positions(
        rv_forecast=rv_test_cal,
        rv_baseline=rv_test_base,
        rv_regime=test_regime,
        s_edge=s_edge_default,
        edge_min=edge_min_default,
        L_max=L_max_default,
        gamma_map=default_gamma_map,
    )
    pnl_test_mag = varswap_pnl_with_pos(rv_test_true, rv_test_base, pos_test_mag)
    stats_test_mag = pnl_stats(pnl_test_mag, name="TEST (calibrated blend magnitude-aware varswap)")

    print("\nSaving cumulative PnL plot for TEST (magnitude-aware) as varswap_pnl_5s_test_magaware.png ...")
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pnl_test_mag), label="Calibrated blend (mag-aware)")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Magnitude-aware variance-swap cumulative PnL (5s RV, TEST)")
    plt.xlabel("5-second steps (TEST)")
    plt.ylabel("Cumulative PnL (RV units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("varswap_pnl_5s_test_magaware.png")
    plt.close()
    print("Done magnitude-aware PnL plot.")

    # --------- GRID SEARCH ON DEV FOR MAGNITUDE-AWARE STRATEGY ---------
    print("\n========== GRID SEARCH (DEV ONLY) FOR MAGNITUDE-AWARE STRATEGY ==========")
    print("Searching over s_edge, edge_min, L_MAX; keeping gamma_map fixed (if available).")

    # Parameter grids
    s_edge_grid = [0.10, 0.15, 0.20, 0.30]
    edge_min_grid = [0.02, 0.05, 0.08]
    L_max_grid = [1.0, 1.5, 2.0]

    best_cfg = None
    best_sharpe = -np.inf

    for s_edge in s_edge_grid:
        for edge_min in edge_min_grid:
            for L_max in L_max_grid:
                pos_dev_grid = build_magaware_positions(
                    rv_forecast=rv_dev_cal,
                    rv_baseline=rv_dev_base,
                    rv_regime=dev_regime,
                    s_edge=s_edge,
                    edge_min=edge_min,
                    L_max=L_max,
                    gamma_map=default_gamma_map,
                )
                pnl_dev_grid = varswap_pnl_with_pos(rv_dev_true, rv_dev_base, pos_dev_grid)
                mean = pnl_dev_grid.mean()
                std = pnl_dev_grid.std(ddof=0)
                sharpe = float(mean / std) if std > 0 else -np.inf

                print(f"  [DEV] s_edge={s_edge:.2f}, edge_min={edge_min:.2f}, L_max={L_max:.2f} "
                      f"-> Sharpe_5s={sharpe:.4f}")

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_cfg = {
                        "s_edge": s_edge,
                        "edge_min": edge_min,
                        "L_max": L_max,
                    }

    print("\nBest DEV config (magnitude-aware, fixed gamma_map):")
    print(best_cfg)
    print(f"Best DEV Sharpe_5s: {best_sharpe:.4f}")

    # Apply best config to DEV and TEST for final reporting
    s_edge_best = best_cfg["s_edge"]
    edge_min_best = best_cfg["edge_min"]
    L_max_best = best_cfg["L_max"]

    pos_dev_best = build_magaware_positions(
        rv_forecast=rv_dev_cal,
        rv_baseline=rv_dev_base,
        rv_regime=dev_regime,
        s_edge=s_edge_best,
        edge_min=edge_min_best,
        L_max=L_max_best,
        gamma_map=default_gamma_map,
    )
    pnl_dev_best = varswap_pnl_with_pos(rv_dev_true, rv_dev_base, pos_dev_best)
    stats_dev_best = pnl_stats(pnl_dev_best, name="DEV (mag-aware, grid-search best)")

    pos_test_best = build_magaware_positions(
        rv_forecast=rv_test_cal,
        rv_baseline=rv_test_base,
        rv_regime=test_regime,
        s_edge=s_edge_best,
        edge_min=edge_min_best,
        L_max=L_max_best,
        gamma_map=default_gamma_map,
    )
    pnl_test_best = varswap_pnl_with_pos(rv_test_true, rv_test_base, pos_test_best)
    stats_test_best = pnl_stats(pnl_test_best, name="TEST (mag-aware, grid-search best)")

    print("\nSaving cumulative PnL plot for TEST (mag-aware, grid-search best) "
          "as varswap_pnl_5s_test_magaware_best.png ...")
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pnl_test_best), label="Mag-aware (grid-search best)")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Magnitude-aware varswap cumulative PnL (5s RV, TEST, grid-search best)")
    plt.xlabel("5-second steps (TEST)")
    plt.ylabel("Cumulative PnL (RV units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("varswap_pnl_5s_test_magaware_best.png")
    plt.close()
    print("Done mag-aware best PnL plot.")

    # --------- Tiny JSON summary (including new best strategy) ---------
    summary = {
        "dev": {
            "xgb_sign": stats_dev_xgb,
            "blend_sign": stats_dev_blend,
            "calibrated_sign": stats_dev_cal,
            "calibrated_magaware_default": stats_dev_mag,
            "calibrated_magaware_best": stats_dev_best,
        },
        "test": {
            "xgb_sign": stats_test_xgb,
            "blend_sign": stats_test_blend,
            "calibrated_sign": stats_test_cal,
            "calibrated_magaware_default": stats_test_mag,
            "calibrated_magaware_best": stats_test_best,
        },
        "best_magaware_params": {
            "s_edge": s_edge_best,
            "edge_min": edge_min_best,
            "L_max": L_max_best,
            "dev_sharpe_5s": best_sharpe,
        },
    }
    with open("varswap_pnl_5s_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved varswap_pnl_5s_summary.json")


if __name__ == "__main__":
    main()
