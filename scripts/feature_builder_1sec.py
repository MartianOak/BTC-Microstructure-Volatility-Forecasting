#!/usr/bin/env python3
"""
feature_builder_1sec.py


Input:
    /kaggle/input/high-frequency-crypto-limit-order-book-data/BTC_1sec.csv
    or BTC_1sec.csv in the working directory.

Output:
    BTC_1sec_features.csv

Contents:
    - Cleaned time index
    - 1s log returns and RV
    - Forward RV targets: y_rv_1s, y_rv_5s, y_rv_10s (raw + log)
    - OFI features (top 1/5/10 levels)
    - Depth imbalance (top 1/5/10)
    - Microprice & microprice pressure
    - Rolling RV, volume, spread features
    - Simple queue dynamics (limit vs cancel flows)
        * LOB rebalancing (change in imbalance/depth)
        * LOB pressure ratios (bid/ask depth share)
        * Volatility-adjusted queue/imbalance
        * Liquidity fade metrics
        * Market order impact proxies
"""

import numpy as np
import pandas as pd


DATA_FILE = "/kaggle/input/high-frequency-crypto-limit-order-book-data/BTC_1sec.csv"

def load_raw(path: str = DATA_FILE) -> pd.DataFrame:
    print(f"Loading raw data from {path} ...")
    df = pd.read_csv(path)

    # Drop index-like column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Parse time and sort
    df["system_time"] = pd.to_datetime(df["system_time"], utc=True)
    df = df.sort_values("system_time").reset_index(drop=True)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def add_basic_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("\nAdding 1s log returns and RV...")
    df["mid_log"] = np.log(df["midpoint"])
    df["r_1s"] = df["mid_log"].diff()
    df["rv_1s"] = df["r_1s"] ** 2

    return df


def add_forward_targets(df: pd.DataFrame, horizons=(1, 5, 10)) -> pd.DataFrame:
    """
    Add forward realized variance targets for given horizons (in seconds).

    For H in horizons:
      y_rv_{H}s = sum_{t..t+H-1} rv_1s (shifted so value at time t
                   is the RV over [t, t+H] as rows)
    """
    df = df.copy()
    print("\nAdding forward RV targets...")

    eps = 1e-18

    for H in horizons:
        roll = df["rv_1s"].rolling(window=H, min_periods=H).sum()
        # sum rv_1s from t..t+H-1; place at time t
        y_rv = roll.shift(-H + 1)

        name = f"y_rv_{H}s"
        log_name = f"{name}_log"

        df[name] = y_rv
        df[log_name] = np.log(y_rv + eps)

        print(f"  Added {name} and {log_name}")

    return df


def add_ofi_features(df: pd.DataFrame, levels=(1, 5, 10)) -> pd.DataFrame:
    """
    OFI (Order Flow Imbalance) using changes in bids_notional_k and asks_notional_k.

    For each N in levels:
      ofi_topN_t = sum_{k < N} Δ(bids_notional_k) - sum_{k < N} Δ(asks_notional_k)
    """
    df = df.copy()
    print("\nAdding OFI features...")

    max_level = 0
    # detect how many levels we actually have
    for k in range(100):
        if f"bids_notional_{k}" in df.columns:
            max_level = k
        else:
            break
    L = max_level + 1
    print(f"Detected {L} bid/ask notional levels (0..{max_level}).")

    for N in levels:
        N_eff = min(N, L)
        bid_cols = [f"bids_notional_{k}" for k in range(N_eff)]
        ask_cols = [f"asks_notional_{k}" for k in range(N_eff)]

        missing = [c for c in bid_cols + ask_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: missing columns for OFI N={N}: {missing}")
            continue

        d_bids = df[bid_cols].diff()
        d_asks = df[ask_cols].diff()
        df[f"ofi_top{N_eff}"] = d_bids.sum(axis=1) - d_asks.sum(axis=1)
        print(f"  Added ofi_top{N_eff}")

    return df


def add_imbalance_features(df: pd.DataFrame, levels=(1, 5, 10)) -> pd.DataFrame:
    """
    Depth imbalance over top N levels:

      imb_topN = (sum_k bid_notional_k - sum_k ask_notional_k) /
                 (sum_k bid_notional_k + sum_k ask_notional_k)
    """
    df = df.copy()
    print("\nAdding depth imbalance features...")

    max_level = 0
    for k in range(100):
        if f"bids_notional_{k}" in df.columns:
            max_level = k
        else:
            break
    L = max_level + 1

    eps = 1e-18

    for N in levels:
        N_eff = min(N, L)
        bid_cols = [f"bids_notional_{k}" for k in range(N_eff)]
        ask_cols = [f"asks_notional_{k}" for k in range(N_eff)]

        missing = [c for c in bid_cols + ask_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: missing columns for imbalance N={N}: {missing}")
            continue

        bid_sum = df[bid_cols].sum(axis=1)
        ask_sum = df[ask_cols].sum(axis=1)
        num = bid_sum - ask_sum
        den = bid_sum + ask_sum + eps
        df[f"imb_top{N_eff}"] = num / den
        print(f"  Added imb_top{N_eff}")

    return df


def add_microprice_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate best bid/ask from midpoint and distance_0 to build microprice:

      bid_0 = mid * (1 + bids_distance_0)
      ask_0 = mid * (1 + asks_distance_0)

    microprice = (ask_0 * bid_notional_0 + bid_0 * ask_notional_0) /
                 (bid_notional_0 + ask_notional_0)

    microprice_disp = (microprice - midpoint) / midpoint
    """
    df = df.copy()
    print("\nAdding microprice features...")

    required = ["midpoint", "bids_distance_0", "asks_distance_0",
                "bids_notional_0", "asks_notional_0"]
    if any(c not in df.columns for c in required):
        print("  Missing columns for microprice; skipping.")
        return df

    mid = df["midpoint"]
    bid0 = mid * (1.0 + df["bids_distance_0"])
    ask0 = mid * (1.0 + df["asks_distance_0"])

    bid_vol = df["bids_notional_0"]
    ask_vol = df["asks_notional_0"]
    denom = bid_vol + ask_vol

    micro = (ask0 * bid_vol + bid0 * ask_vol) / denom.replace(0, np.nan)
    df["microprice"] = micro
    df["microprice_disp"] = (micro - mid) / mid

    print("  Added microprice and microprice_disp")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple rolling features over 5s, 30s, 60s windows:
      - sum/mean of rv_1s
      - sum/mean of buys, sells
      - mean of spread
      - std of r_1s
    """
    df = df.copy()
    print("\nAdding rolling features (5s, 30s, 60s)...")

    windows = [5, 30, 60]

    for w in windows:
        w_str = f"{w}s"

        # Volatility / RV
        df[f"rv_{w_str}_past"] = df["rv_1s"].rolling(window=w, min_periods=w).sum()
        df[f"rv_{w_str}_mean_past"] = df["rv_1s"].rolling(window=w, min_periods=w).mean()

        # Volume
        df[f"buys_sum_{w_str}"] = df["buys"].rolling(window=w, min_periods=w).sum()
        df[f"sells_sum_{w_str}"] = df["sells"].rolling(window=w, min_periods=w).sum()
        df[f"net_flow_{w_str}"] = df[f"buys_sum_{w_str}"] - df[f"sells_sum_{w_str}"]

        # Spread stats
        df[f"spread_mean_{w_str}"] = df["spread"].rolling(window=w, min_periods=w).mean()

        # Return std
        df[f"r_1s_std_{w_str}"] = df["r_1s"].rolling(window=w, min_periods=w).std()

        print(f"  Added rolling features for window={w}s")

    return df


def add_queue_features(df: pd.DataFrame, levels=(1, 5)) -> pd.DataFrame:
    """
    Simple queue dynamics from limit and cancel notional:

      sum_limit_topN_bids, sum_cancel_topN_bids,
      sum_limit_topN_asks, sum_cancel_topN_asks,
      net_liq_change_topN = (limit_bids + limit_asks) - (cancel_bids + cancel_asks)
    """
    df = df.copy()
    print("\nAdding queue dynamics features...")

    # detect max level for these notional types
    def max_level_for(prefix):
        for k in range(100):
            if f"{prefix}_{k}" not in df.columns:
                return k - 1
        return 99

    max_bid_lim = max_level_for("bids_limit_notional")
    max_ask_lim = max_level_for("asks_limit_notional")
    max_bid_can = max_level_for("bids_cancel_notional")
    max_ask_can = max_level_for("asks_cancel_notional")

    for N in levels:
        # bids/asks limit/cancel
        bid_lim_cols = [f"bids_limit_notional_{k}" for k in range(min(N, max_bid_lim + 1))]
        ask_lim_cols = [f"asks_limit_notional_{k}" for k in range(min(N, max_ask_lim + 1))]
        bid_can_cols = [f"bids_cancel_notional_{k}" for k in range(min(N, max_bid_can + 1))]
        ask_can_cols = [f"asks_cancel_notional_{k}" for k in range(min(N, max_ask_can + 1))]

        # Check if any exist
        if not bid_lim_cols or not any(c in df.columns for c in bid_lim_cols + ask_lim_cols):
            print(f"  No limit_notional columns found for top{N}; skipping queue features.")
            continue

        def safe_sum(cols):
            cols = [c for c in cols if c in df.columns]
            if not cols:
                return pd.Series(0.0, index=df.index)
            return df[cols].sum(axis=1)

        lim_bids = safe_sum(bid_lim_cols)
        lim_asks = safe_sum(ask_lim_cols)
        can_bids = safe_sum(bid_can_cols)
        can_asks = safe_sum(ask_can_cols)

        df[f"limit_sum_top{N}_bids"] = lim_bids
        df[f"limit_sum_top{N}_asks"] = lim_asks
        df[f"cancel_sum_top{N}_bids"] = can_bids
        df[f"cancel_sum_top{N}_asks"] = can_asks

        df[f"net_liq_change_top{N}"] = (lim_bids + lim_asks) - (can_bids + can_asks)

        print(f"  Added queue features for top{N}")

    return df


def add_style_features(df: pd.DataFrame, levels=(1, 5, 10)) -> pd.DataFrame:
    """
      - LOB rebalancing: change in imbalance / depth
      - LOB pressure ratios: bid vs ask share of depth
      - Volatility-adjusted imbalance / queue signals
      - Liquidity fade: how quickly depth disappears
      - Market order impact proxies
    """
    df = df.copy()
    print("\nAdding microstructure features...")
    eps = 1e-18

    # Helper to sum depth safely
    def safe_sum(cols):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return pd.Series(0.0, index=df.index)
        return df[cols].sum(axis=1)

    # Depth-based features for different top-N levels
    max_level = 0
    for k in range(100):
        if f"bids_notional_{k}" in df.columns:
            max_level = k
        else:
            break
    L = max_level + 1

    for N in levels:
        N_eff = min(N, L)
        bid_cols = [f"bids_notional_{k}" for k in range(N_eff)]
        ask_cols = [f"asks_notional_{k}" for k in range(N_eff)]

        depth_bid = safe_sum(bid_cols)
        depth_ask = safe_sum(ask_cols)
        depth_tot = depth_bid + depth_ask

        df[f"depth_bid_top{N_eff}"] = depth_bid
        df[f"depth_ask_top{N_eff}"] = depth_ask
        df[f"depth_tot_top{N_eff}"] = depth_tot

        # LOB pressure ratios (bid / total, ask / total)
        df[f"press_bid_top{N_eff}"] = depth_bid / (depth_tot + eps)
        df[f"press_ask_top{N_eff}"] = depth_ask / (depth_tot + eps)

        # LOB rebalancing: change in imbalance over time
        if f"imb_top{N_eff}" in df.columns:
            df[f"rebal_imb_top{N_eff}"] = df[f"imb_top{N_eff}"].diff()
        else:
            net_depth = depth_bid - depth_ask
            df[f"rebal_imb_top{N_eff}"] = (net_depth / (depth_tot + eps)).diff()

        # Liquidity fade: loss of total depth normalized by previous depth
        depth_diff = depth_tot.diff()
        df[f"liq_fade_top{N_eff}"] = -np.minimum(depth_diff, 0.0) / (depth_tot.shift(1) + eps)

        # Volatility-adjusted imbalance (strong around bursts)
        if f"imb_top{N_eff}" in df.columns:
            df[f"imb_voladj_top{N_eff}"] = df[f"imb_top{N_eff}"] / np.sqrt(df["rv_1s"] + eps)

    # Market order impact proxies (from buys/sells)
    if {"buys", "sells"}.issubset(df.columns):
        # Intensity scaled by price (approx "volume per dollar")
        df["mo_intensity"] = (df["buys"] + df["sells"]) / (df["midpoint"] + eps)

        # Signed imbalance of market orders
        df["mo_imbalance"] = (df["buys"] - df["sells"]) / (df["buys"] + df["sells"] + eps)

        # Volatility-adjusted market order imbalance
        df["mo_imb_voladj"] = df["mo_imbalance"] / np.sqrt(df["rv_1s"] + eps)

        # Simple impact proxy: MO imbalance interacting with recent return
        df["mo_price_impact_proxy"] = df["mo_imbalance"] * df["r_1s"].rolling(3, min_periods=1).sum()

        print("  Added market order impact proxies (mo_intensity, mo_imbalance, mo_imb_voladj, mo_price_impact_proxy)")

    return df


def finalize_and_save(df: pd.DataFrame, out_file: str = "BTC_1sec_features.csv"):
    """
    Clean up and save feature dataset.
    We drop rows where forward targets are NaN (at the tail).
    """
    print("\nFinalizing feature dataset...")

    # Require 1s, 5s, and 10s targets
    target_cols = [
        "y_rv_1s", "y_rv_1s_log",
        "y_rv_5s", "y_rv_5s_log",
        "y_rv_10s", "y_rv_10s_log",
    ]
    existing_targets = [c for c in target_cols if c in df.columns]

    before = len(df)
    if existing_targets:
        df = df.dropna(subset=existing_targets)
        after = len(df)
        print(f"Dropped {before - after} rows with NaN targets; remaining: {after}")
    else:
        print("WARNING: no target columns found; not dropping NaNs on targets.")

    print(f"Saving features to {out_file} ...")
    df.to_csv(out_file, index=False)
    print("Done.")

def add_regime_features(df: pd.DataFrame,
                        n_vol_bins: int = 4,
                        n_liq_bins: int = 3,
                        n_ofi_bins: int = 3) -> pd.DataFrame:
    """
    Add regime-robust features:

      - Realized volatility regime (based on rv_5s_past)
      - Liquidity regime (spread & depth_top1 quantiles)
      - Order-flow regime (OFI intensity on ofi_top1)

    Each regime is added both as an integer label and as one-hot dummies.
    """
    df = df.copy()
    eps = 1e-18
    print("\nAdding regime-robust features...")

    # Safety checks
    required_cols = [
        "rv_5s_past",
        "spread",
        "depth_bid_top1",
        "depth_ask_top1",
        "ofi_top1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns for regime features: {missing}")
        print("  Skipping regime feature construction.")
        return df

    # ---------- 1) Realized volatility regime ----------
    vol_source = df["rv_5s_past"].clip(lower=eps)

    try:
        vol_regime = pd.qcut(
            vol_source,
            q=n_vol_bins,
            labels=False,
            duplicates="drop"
        )
        df["regime_vol"] = vol_regime.astype("float32")
        print(f"  Added regime_vol with up to {n_vol_bins} quantile bins.")
    except ValueError as e:
        print(f"  WARNING: could not compute vol regimes: {e}")
        df["regime_vol"] = np.nan

    # One-hot for vol regime
    if df["regime_vol"].notna().any():
        vol_dummies = pd.get_dummies(
            df["regime_vol"],
            prefix="regime_vol",
            dtype=np.int8
        )
        df = pd.concat([df, vol_dummies], axis=1)
        print(f"  Added {vol_dummies.shape[1]} one-hot columns for regime_vol.")

    # ---------- 2) Liquidity regime ----------
    # (a) Spread regime
    try:
        spread_regime = pd.qcut(
            df["spread"],
            q=n_liq_bins,
            labels=False,
            duplicates="drop"
        )
        df["regime_spread"] = spread_regime.astype("float32")
        print(f"  Added regime_spread with up to {n_liq_bins} bins.")
    except ValueError as e:
        print(f"  WARNING: could not compute spread regimes: {e}")
        df["regime_spread"] = np.nan

    if df["regime_spread"].notna().any():
        spread_dummies = pd.get_dummies(
            df["regime_spread"],
            prefix="regime_spread",
            dtype=np.int8
        )
        df = pd.concat([df, spread_dummies], axis=1)
        print(f"  Added {spread_dummies.shape[1]} one-hot columns for regime_spread.")

    # (b) Depth regime (top-of-book depth)
    depth_top1 = df["depth_bid_top1"] + df["depth_ask_top1"]
    try:
        depth_regime = pd.qcut(
            depth_top1,
            q=n_liq_bins,
            labels=False,
            duplicates="drop"
        )
        df["regime_depth"] = depth_regime.astype("float32")
        print(f"  Added regime_depth with up to {n_liq_bins} bins.")
    except ValueError as e:
        print(f"  WARNING: could not compute depth regimes: {e}")
        df["regime_depth"] = np.nan

    if df["regime_depth"].notna().any():
        depth_dummies = pd.get_dummies(
            df["regime_depth"],
            prefix="regime_depth",
            dtype=np.int8
        )
        df = pd.concat([df, depth_dummies], axis=1)
        print(f"  Added {depth_dummies.shape[1]} one-hot columns for regime_depth.")

    # ---------- 3) Order-flow regime (OFI) ----------
    ofi_source = df["ofi_top1"]

    try:
        ofi_regime = pd.qcut(
            ofi_source,
            q=n_ofi_bins,
            labels=False,
            duplicates="drop"
        )
        df["regime_ofi"] = ofi_regime.astype("float32")
        print(f"  Added regime_ofi with up to {n_ofi_bins} bins.")
    except ValueError as e:
        print(f"  WARNING: could not compute OFI regimes: {e}")
        df["regime_ofi"] = np.nan

    if df["regime_ofi"].notna().any():
        ofi_dummies = pd.get_dummies(
            df["regime_ofi"],
            prefix="regime_ofi",
            dtype=np.int8
        )
        df = pd.concat([df, ofi_dummies], axis=1)
        print(f"  Added {ofi_dummies.shape[1]} one-hot columns for regime_ofi.")

    print("Regime-robust features added.")
    return df

def add_vol_regime(df: pd.DataFrame, q=3):
    """
    Volatility regime based on rolling 60s RV.
    Creates rv_regime in {0,1,2}.
    """
    df = df.copy()
    if "rv_60s_mean_past" not in df.columns:
        print("No rv_60s_mean_past, skipping rv_regime.")
        return df

    rv = df["rv_60s_mean_past"].fillna(method="bfill").fillna(method="ffill")
    df["rv_regime"] = pd.qcut(rv, q=q, labels=False)
    print("Added rv_regime (0=low,1=mid,2=high).")
    return df



def main():
    df = load_raw()

    df = add_basic_price_features(df)
    df = add_forward_targets(df, horizons=(1, 5, 10))
    df = add_ofi_features(df, levels=(1, 5, 10))
    df = add_imbalance_features(df, levels=(1, 5, 10))
    df = add_microprice_features(df)
    df = add_rolling_features(df)
    df = add_queue_features(df, levels=(1, 5))
    df = add_style_features(df, levels=(1, 5, 10))
    df = add_regime_features(df)
    df = add_vol_regime(df)

    finalize_and_save(df)


if __name__ == "__main__":
    main()
