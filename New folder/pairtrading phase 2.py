# ==========================================================
# PHASE 2
# STATIONARITY ANALYSIS
# Johansen Test + ADF Test
# ==========================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pathlib import Path

# ==========================================================
# LOAD DATA
# ==========================================================

prices = pd.read_csv(
    "data/log_prices.csv",
    index_col=0,
    parse_dates=True
)

cointegration_results = pd.read_csv(
    "results/cointegration_results.csv"
)

# ==========================================================
# TAKE TOP N PAIRS
# ==========================================================

TOP_N = 10

top_pairs = cointegration_results.head(TOP_N)

print("="*60)
print("TOP PAIRS FROM ENGLE-GRANGER")
print("="*60)

print(top_pairs)

# ==========================================================
# STORE RESULTS
# ==========================================================

johansen_results = []

adf_results = []

# ==========================================================
# LOOP THROUGH PAIRS
# ==========================================================

for _, row in top_pairs.iterrows():

    asset_y = row["Asset_1"]
    asset_x = row["Asset_2"]

    print("\n")
    print("="*60)
    print(f"Testing Pair: {asset_y} - {asset_x}")
    print("="*60)

    # ------------------------------------------------------
    # Johansen Test
    # ------------------------------------------------------

    pair_data = prices[[asset_y, asset_x]]

    johansen = coint_johansen(
        pair_data,
        det_order=0,
        k_ar_diff=1
    )

    trace_stat = johansen.lr1[0]
    crit95 = johansen.cvt[0,1]

    johansen_pass = trace_stat > crit95

    johansen_results.append({

        "Asset_1": asset_y,

        "Asset_2": asset_x,

        "Trace Statistic": trace_stat,

        "95% Critical Value": crit95,

        "Johansen Passed": johansen_pass

    })

    print(f"Trace Statistic : {trace_stat:.3f}")
    print(f"95% Critical    : {crit95:.3f}")
    print(f"Cointegrated?   : {johansen_pass}")

    # ------------------------------------------------------
    # OLS HEDGE RATIO
    # ------------------------------------------------------

    Y = prices[asset_y]

    X = sm.add_constant(prices[asset_x])

    model = sm.OLS(Y, X).fit()

    beta = model.params.iloc[1]

    spread = Y - beta * prices[asset_x]

    # ------------------------------------------------------
    # ADF TEST
    # ------------------------------------------------------

    adf = adfuller(spread)

    statistic = adf[0]

    pvalue = adf[1]

    crit1 = adf[4]["1%"]

    crit5 = adf[4]["5%"]

    crit10 = adf[4]["10%"]

    stationary = pvalue < 0.05

    adf_results.append({

        "Asset_1": asset_y,

        "Asset_2": asset_x,

        "ADF Statistic": statistic,

        "P Value": pvalue,

        "Stationary": stationary,

        "Beta": beta

    })

    print()

    print("ADF TEST")

    print(f"Statistic : {statistic:.4f}")

    print(f"P Value   : {pvalue:.6f}")

    print(f"1% CV     : {crit1:.4f}")

    print(f"5% CV     : {crit5:.4f}")

    print(f"10% CV    : {crit10:.4f}")

    print(f"Stationary: {stationary}")

# ==========================================================
# SAVE RESULTS
# ==========================================================

johansen_df = pd.DataFrame(johansen_results)

adf_df = pd.DataFrame(adf_results)

johansen_df.to_csv(
    "results/johansen_results.csv",
    index=False
)

adf_df.to_csv(
    "results/adf_results.csv",
    index=False
)

# ==========================================================
# FINAL PAIR SELECTION
# ==========================================================

final_pairs = pd.merge(

    johansen_df,

    adf_df,

    on=["Asset_1","Asset_2"]

)

final_pairs = final_pairs.sort_values(

    by="P Value"

)

final_pairs.to_csv(

    "results/final_pairs.csv",

    index=False

)

print()

print("="*60)

print("FINAL RANKING")

print("="*60)

print(final_pairs)

print()

print("="*60)

print("PHASE 2 COMPLETE")

print("="*60)

print()

print("Generated Files")

print("----------------")

print("results/johansen_results.csv")

print("results/adf_results.csv")

print("results/final_pairs.csv")
