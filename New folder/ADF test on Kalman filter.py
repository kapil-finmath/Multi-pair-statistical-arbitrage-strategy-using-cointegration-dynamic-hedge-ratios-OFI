# ==========================================================
# ADF TEST ON KALMAN FILTER SPREAD
# ==========================================================

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller

# ==========================================================
# LOAD DATA
# ==========================================================

prices = pd.read_csv(
    "data/log_prices.csv",
    index_col=0
)

dynamic = pd.read_csv(
    "results/dynamic_beta.csv",
    index_col=0
)

# ==========================================================
# RESEARCH PAIR
# ==========================================================

Y = prices["IVV"]

X = prices["VGT"]

beta = dynamic["Beta"]

alpha = dynamic["Alpha"]

# ==========================================================
# DYNAMIC SPREAD
# ==========================================================

spread_dynamic = (
    Y
    - beta * X
    - alpha
)

spread_dynamic = spread_dynamic.dropna()

# ==========================================================
# ADF TEST
# ==========================================================

result = adfuller(spread_dynamic)

print("=" * 60)
print("ADF TEST : DYNAMIC SPREAD")
print("=" * 60)

print()

print(f"ADF Statistic : {result[0]:.6f}")
print(f"P Value       : {result[1]:.6f}")

print()

for key, value in result[4].items():

    print(f"{key} Critical Value : {value:.6f}")

print()

if result[1] < 0.05:
    print("Stationary : TRUE")
else:
    print("Stationary : FALSE")

# ==========================================================
# VARIANCE
# ==========================================================

dynamic_var = np.var(spread_dynamic)

print()
print(f"Spread Variance : {dynamic_var:.8f}")

# ==========================================================
# SAVE
# ==========================================================

pd.DataFrame({

    "ADF Statistic": [result[0]],
    "P Value": [result[1]],
    "Variance": [dynamic_var]

}).to_csv(
    "results/dynamic_spread_validation.csv",
    index=False
)

print()
print("=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
