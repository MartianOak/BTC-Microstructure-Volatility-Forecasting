# BTC 5-Second Realized Volatility Forecasting  
Hybrid HAR–XGBoost Model + Variance-Swap Style Backtests

This repository contains the complete analysis, modeling, evaluation, and trading-style backtests for forecasting **5-second realized variance (RV)** on 1 second BTCUSDT limit order book data over 12 days. 

All outputs were generated in a fully executed Kaggle notebook and included here for convenience.

---

## 📂 Repository Structure

```
volatility_BTC/
│
├── btc-5sec-blended-har-xgboost-model.ipynb   # Fully executed Kaggle notebook
│
├── scripts/                                   # Source scripts used in the notebook
│   ├── backtest_rv_5sec_varswap.py
│   ├── calibrate_rv_5s.py
│   ├── cv_rv_5sec_xgb.py
│   ├── eda_1sec_lob.py
│   ├── feature_builder_1sec.py
│   ├── train_rv_5s_xgb.py
│
└── outputs/                                   # Model outputs, diagnostics, plots
    ├── *.png
    ├── *.csv
    ├── *.json
    └── *.pkl
```

---

## 🧠 Overview

The goal of this project is to predict **future 5-second realized variance** using:

- 1-second limit order book features  
- Historical realized volatility  
- Order flow imbalance  
- Depth and spread metrics  
- Microprice signals  
- HAR-style lagged realized volatility features  
- A GPU-accelerated XGBoost forecaster  
- A blending scheme between baseline RV and XGB predictions  
- Global isotonic calibration  
- Volatility-based trading backtests (variance-swap style)

The outputs include:

- feature importances  
- predictive performance plots  
- risk-adjusted trading metrics  
- cross-validation search results  
- calibrated and uncalibrated models  
- PnL curves for magnitude-aware + sign-only strategies  

All final results are stored in `outputs/`.

---

## 📦 Data Source

⚠ **The raw dataset is *not* included in this repository** because it is large.

To reproduce the results, download the dataset from Kaggle:

**High-Frequency Crypto Limit Order Book Data**  
https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data

Specifically, this project uses:

```
BTC_1sec.csv
```

---

## ▶ How to Reproduce (on Kaggle)

1. Open a new Kaggle Notebook.
2. Upload the notebook from this repo:

```
btc-5sec-blended-har-xgboost-model.ipynb
```

3. On the right sidebar → **Add Data** → search:

```
high-frequency crypto limit order book data
```

4. Select the dataset containing `BTC_1sec.csv`.

5. Run all cells (optional — the notebook already includes full outputs).

---

## 📑 Scripts Included

These scripts mirror the logic in the notebook:

- `feature_builder_1sec.py`  
  Builds 1-second features including OFI, spreads, depth imbalance, volatility lags.

- `train_rv_5s_xgb.py`  
  Fits the GPU-XGBoost forecaster on log-space RV.

- `cv_rv_5sec_xgb.py`  
  Time-series cross-validation with random hyperparameter search + blend-alpha tuning.

- `calibrate_rv_5s.py`  
  Fits global isotonic regression to correct log-RV predictions.

- `backtest_rv_5sec_varswap.py`  
  Executes sign-based and magnitude-aware RV trading strategies.

- `eda_1sec_lob.py`  
  Exploratory plots for BTC 1-sec LOB dataset.

---

## 📈 Key Results (Quick Summary)

- **Strong correlation** between predicted and true RV (both DEV and TEST).
- **Blended HAR + XGBoost model** outperforms baseline and pure XGB.
- **Sign-based strategy** achieves >0.65 hit-ratio.
- **Magnitude-aware strategy** shows smooth PnL curves and interpretable risk profiles.
- **Global isotonic calibration** further improves RV-space accuracy.

All plots and summary tables are in the `outputs/` folder.

---

## 📬 Contact

Feel free to open an issue if you have questions or want to extend this project.

---

## ⭐ Acknowledgements

Dataset by [martinsn](https://www.kaggle.com/martinsn).  
Thanks to the Kaggle community for compute resources and support.
