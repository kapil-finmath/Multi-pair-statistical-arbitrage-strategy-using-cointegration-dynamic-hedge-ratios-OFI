
# PHASE 1
# DATA COLLECTION + COINTEGRATION ANALYSIS

import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
from pathlib import Path


#  CREATE PROJECT FOLDERS


Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

#  ETF UNIVERSE


etfs = [
    "XLF",
    "KBE",
    "GLD",
    "GDX",
    "XLE",
    "VDE",
    "SPY",
    "IVV",
    "QQQ",
    "VGT"
]

START_DATE = "2020-01-01"
END_DATE   = "2025-01-01"


# 3. DOWNLOAD DATA


print("Downloading ETF data...")

raw_data = yf.download(
    etfs,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)


# 4. EXTRACT CLOSE PRICES


close_prices = raw_data["Close"]

# remove missing values
close_prices = close_prices.dropna()
# SAVE CLOSE PRICES
close_prices.to_csv("data/close_prices.csv")


# EXTRACT VOLUME DATA

volume_data = raw_data["Volume"]

# align with price data
volume_data = volume_data.loc[close_prices.index]

# remove missing values if any
volume_data = volume_data.fillna(0)

# SAVE VOLUME DATA
volume_data.to_csv("data/volume.csv")

print("\nVolume data saved.")


#  LOG PRICES


log_prices = np.log(close_prices)

log_prices.to_csv("data/log_prices.csv")

print("\nLog prices saved.")


#  SUMMARY STATISTICS

summary = close_prices.describe()

summary.to_csv("results/summary_statistics.csv")

print("\nSummary statistics saved.")


# ENGLE-GRANGER COINTEGRATION TEST

print("\nRunning cointegration tests...")

pair_results = []

pairs = list(itertools.combinations(etfs, 2))

for asset1, asset2 in pairs:

    series1 = log_prices[asset1]
    series2 = log_prices[asset2]

    score, pvalue, _ = coint(series1, series2)

    pair_results.append({
        "Asset_1": asset1,
        "Asset_2": asset2,
        "Cointegration_Score": score,
        "P_Value": pvalue
    })

results_df = pd.DataFrame(pair_results)

results_df = results_df.sort_values(
    by="P_Value",
    ascending=True
)

results_df.to_csv(
    "results/cointegration_results.csv",
    index=False
)


# DISPLAY TOP PAIRS


print("\nTop Cointegrated Pairs:")
print(results_df.head(10))


#  SELECT BEST PAIR

best_pair = results_df.iloc[0]

asset_y = best_pair["Asset_1"]
asset_x = best_pair["Asset_2"]

print("\nBest Pair Found:")
print(asset_y, asset_x)


#  ESTIMATE HEDGE RATIO (OLS)


Y = log_prices[asset_y]

X = sm.add_constant(
    log_prices[asset_x]
)

model = sm.OLS(Y, X).fit()

beta = model.params.iloc[1]
alpha = model.params.iloc[0]

print("\nOLS Results")
print(f"Alpha = {alpha:.4f}")
print(f"Beta  = {beta:.4f}")


# CONSTRUCT SPREAD

spread = Y - beta * log_prices[asset_x]

spread_df = pd.DataFrame({
    "Date": spread.index,
    "Spread": spread.values
})

spread_df.to_csv(
    "results/spread.csv",
    index=False
)


#  COMPUTE Z-SCORE


rolling_window = 60

spread_mean = spread.rolling(
    rolling_window
).mean()

spread_std = spread.rolling(
    rolling_window
).std()

zscore = (
    spread - spread_mean
) / spread_std

zscore_df = pd.DataFrame({
    "Date": zscore.index,
    "ZScore": zscore.values
})

zscore_df.to_csv(
    "results/zscore.csv",
    index=False
)


#  PLOT PRICE SERIES


plt.figure(figsize=(12,6))

plt.plot(
    close_prices[asset_y],
    label=asset_y
)

plt.plot(
    close_prices[asset_x],
    label=asset_x
)

plt.title(
    f"{asset_y} vs {asset_x}"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/best_pair_prices.png"
)

plt.close()


#  PLOT SPREAD

plt.figure(figsize=(12,6))

plt.plot(spread)

plt.title(
    f"Spread: {asset_y} - {beta:.2f} * {asset_x}"
)

plt.tight_layout()

plt.savefig(
    "figures/spread.png"
)

plt.close()


#  PLOT Z-SCORE


plt.figure(figsize=(12,6))

plt.plot(
    zscore,
    label="Z-score"
)

plt.axhline(
    2,
    linestyle="--"
)

plt.axhline(
    -2,
    linestyle="--"
)

plt.axhline(
    0,
    linestyle="-"
)

plt.title("Spread Z-score")

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/zscore.png"
)

plt.close()


#  SAVE MODEL PARAMETERS


parameters = pd.DataFrame({
    "Parameter": [
        "Alpha",
        "Beta"
    ],
    "Value": [
        alpha,
        beta
    ]
})

parameters.to_csv(
    "results/model_parameters.csv",
    index=False
)


# DONE


print("\n====================================")
print("PHASE 1 COMPLETED")
print("====================================")

print("\nFiles Created:")



print("data/close_prices.csv")
print("data/log_prices.csv")
print("data/volume.csv")

print("results/cointegration_results.csv")
print("results/spread.csv")
print("results/zscore.csv")
print("results/model_parameters.csv")

print("figures/best_pair_prices.png")
print("figures/spread.png")
print("figures/zscore.png")
