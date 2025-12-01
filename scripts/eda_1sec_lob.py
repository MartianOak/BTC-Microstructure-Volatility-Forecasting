"""
eda_1sec_lob.py

Exploratory Data Analysis for BTC 1-second limit order book data.

- Loads BTC_1sec.csv
- Checks time deltas and gaps
- Computes 1-second log returns and realized variance
- Constructs a simple OFI (order flow imbalance) signal
- Prints basic stats and autocorrelations
- Saves several diagnostic plots as PNG files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Adjust this path as needed (for Kaggle)
DATA_FILE = "/kaggle/input/high-frequency-crypto-limit-order-book-data/BTC_1sec.csv"
# Or if you have it in the working directory:
# DATA_FILE = "BTC_1sec.csv"


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Parse time
    df["system_time"] = pd.to_datetime(df["system_time"], utc=True)
    df = df.sort_values("system_time").reset_index(drop=True)
    return df


def check_time_structure(df: pd.DataFrame):
    print("\n=== Time structure ===")
    dt = df["system_time"].diff()
    # Most common time delta
    mode_delta = dt.value_counts().idxmax()
    print(f"Most common time delta between rows: {mode_delta}")

    # Show unusual gaps (> 2 seconds)
    large_gaps = dt[dt > pd.Timedelta(seconds=2)]
    if len(large_gaps) == 0:
        print("No large gaps (> 2s) detected.")
    else:
        print(f"Found {len(large_gaps)} large gaps (> 2s). Example:")
        print(large_gaps.head())


def compute_returns_and_rv(df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing 1-second log returns and realized variance...")
    df = df.copy()
    df["mid_log"] = np.log(df["midpoint"])
    df["r_1s"] = df["mid_log"].diff()
    df["rv_1s"] = df["r_1s"] ** 2

    # Some longer-horizon realized variances for context
    for window in [5, 30, 60, 300]:
        df[f"rv_{window}s"] = df["rv_1s"].rolling(window=window, min_periods=window).sum()

    return df


def build_simple_ofi(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """
    Build a simple OFI (Order Flow Imbalance) using changes in top-n bid/ask notionals.

    OFI_t = sum_k Δ(bids_notional_k) - sum_k Δ(asks_notional_k)
    """
    print(f"\nConstructing simple OFI using top {n_levels} levels...")
    df = df.copy()

    bid_cols = [f"bids_notional_{k}" for k in range(n_levels)]
    ask_cols = [f"asks_notional_{k}" for k in range(n_levels)]

    missing = [c for c in bid_cols + ask_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns for OFI: {missing}")
        print("OFI will be NaN.")
        df["ofi_top"] = np.nan
        return df

    d_bids = df[bid_cols].diff()
    d_asks = df[ask_cols].diff()
    df["ofi_top"] = d_bids.sum(axis=1) - d_asks.sum(axis=1)
    return df


def basic_stats(df: pd.DataFrame):
    print("\n=== Basic stats (selected columns) ===")
    cols = ["midpoint", "spread", "buys", "sells", "r_1s", "rv_1s", "rv_5s", "rv_30s", "ofi_top"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T)


def autocorr(x: pd.Series, lags):
    res = {}
    for lag in lags:
        if lag <= 0 or lag >= len(x):
            res[lag] = np.nan
        else:
            res[lag] = x.autocorr(lag)
    return res


def check_autocorrelations(df: pd.DataFrame):
    print("\n=== Autocorrelations ===")
    # Drop NaNs at the start
    rv = df["rv_1s"].dropna()
    ofi = df["ofi_top"].dropna()

    for name, series in [("rv_1s", rv), ("ofi_top", ofi)]:
        print(f"\n{name} autocorrelations:")
        ac = autocorr(series, lags=[1, 5, 10, 30, 60])
        for lag, val in ac.items():
            print(f"  lag {lag:2d}: {val:.4f}")


def plot_mid_and_spread(df: pd.DataFrame, out_prefix: str = "btc_1sec"):
    print("\nPlotting midprice & spread (downsampled)...")
    # Downsample to 1 minute for readability
    tmp = df[["system_time", "midpoint", "spread"]].set_index("system_time")
    res = tmp.resample("1min").agg({"midpoint": "last", "spread": "median"})

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(res.index, res["midpoint"], label="midpoint")
    ax1.set_ylabel("Midpoint price")
    ax1.set_title("Midpoint (1-min sampled)")
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_midpoint_1min.png")
    plt.close(fig)

    fig, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(res.index, res["spread"], label="spread (median)", alpha=0.8)
    ax2.set_ylabel("Spread")
    ax2.set_title("Spread (1-min median)")
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_spread_1min.png")
    plt.close(fig)


def plot_depth_profile(df: pd.DataFrame, out_prefix: str = "btc_1sec"):
    print("\nPlotting average depth distance profile...")
    # Use a random subset to reduce memory
    sample = df.sample(n=min(50000, len(df)), random_state=42)

    bid_dist_cols = [c for c in df.columns if c.startswith("bids_distance_")]
    ask_dist_cols = [c for c in df.columns if c.startswith("asks_distance_")]

    if not bid_dist_cols or not ask_dist_cols:
        print("No distance columns found, skipping depth profile plot.")
        return

    # Sort by level index (0..14)
    bid_dist_cols = sorted(bid_dist_cols, key=lambda x: int(x.split("_")[-1]))
    ask_dist_cols = sorted(ask_dist_cols, key=lambda x: int(x.split("_")[-1]))

    bid_mean = sample[bid_dist_cols].mean()
    ask_mean = sample[ask_dist_cols].mean()

    levels = np.arange(len(bid_dist_cols))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels, bid_mean.values, marker="o", label="bids_distance (mean)")
    ax.plot(levels, ask_mean.values, marker="o", label="asks_distance (mean)")
    ax.set_xlabel("Level")
    ax.set_ylabel("Relative distance to mid")
    ax.set_title("Average LOB distance profile (bids vs asks)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_depth_distance_profile.png")
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, out_prefix: str = "btc_1sec"):
    print("\nPlotting histograms for RV and volumes...")
    # rv_1s
    rv = df["rv_1s"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rv, bins=100, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Histogram of 1s realized variance (log y-scale)")
    ax.set_xlabel("rv_1s")
    ax.set_ylabel("Count (log)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_hist_rv_1s.png")
    plt.close(fig)

    # buys and sells
    for col in ["buys", "sells"]:
        if col in df.columns:
            x = df[col].dropna()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(x, bins=100, alpha=0.8)
            ax.set_yscale("log")
            ax.set_title(f"Histogram of {col} (log y-scale)")
            ax.set_xlabel(col)
            ax.set_ylabel("Count (log)")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_hist_{col}.png")
            plt.close(fig)


def plot_ofi_vs_future_rv(df: pd.DataFrame, out_prefix: str = "btc_1sec"):
    print("\nPlotting OFI vs next 1s RV (scatter)...")
    if "ofi_top" not in df.columns:
        print("OFI not found, skipping OFI scatter plot.")
        return

    df = df.copy()
    # Future 1s RV
    df["rv_1s_future"] = df["rv_1s"].shift(-1)

    mask = df["ofi_top"].notna() & df["rv_1s_future"].notna()
    sample = df.loc[mask, ["ofi_top", "rv_1s_future"]]

    # Subsample for plotting
    if len(sample) > 50000:
        sample = sample.sample(n=50000, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample["ofi_top"], sample["rv_1s_future"], s=2, alpha=0.3)
    ax.set_title("OFI (top 5 levels) vs next 1s realized variance")
    ax.set_xlabel("OFI (Δbids_notional - Δasks_notional)")
    ax.set_ylabel("rv_1s (future)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_scatter_ofi_vs_rv_future.png")
    plt.close(fig)


def main():
    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    check_time_structure(df)

    # Compute returns & RV
    df = compute_returns_and_rv(df)

    # Build simple OFI
    df = build_simple_ofi(df, n_levels=5)

    # Basic stats
    basic_stats(df)

    # Autocorrelations
    check_autocorrelations(df)

    # Plots
    plot_mid_and_spread(df)
    plot_depth_profile(df)
    plot_histograms(df)
    plot_ofi_vs_future_rv(df)

    print("\nEDA complete. Check the generated PNG files.")


if __name__ == "__main__":
    main()
