# ==========================================================
# PHASE 3B
# COMPARE HALF-LIFE ACROSS ALL PAIRS
# ==========================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

# ==========================================================
# LOAD DATA
# ==========================================================

prices = pd.read_csv(
    "data/log_prices.csv",
    index_col=0,
    parse_dates=True
)

pairs = pd.read_csv(
    "results/final_pairs.csv"
)

print("=" * 60)
print("PAIR COMPARISON")
print("=" * 60)

results = []

# ==========================================================
# LOOP THROUGH PAIRS
# ==========================================================

for _, row in pairs.iterrows():

    asset_y = row["Asset_1"]
    asset_x = row["Asset_2"]

    Y = prices[asset_y]

    X = sm.add_constant(prices[asset_x])

    model = sm.OLS(Y, X).fit()

    beta = model.params.iloc[1]

    spread = Y - beta * prices[asset_x]

    # ------------------------------------------
    # OU Calibration
    # ------------------------------------------

    S_t = spread[:-1].values
    S_t1 = spread[1:].values

    Xreg = sm.add_constant(S_t)

    ou_model = sm.OLS(S_t1, Xreg).fit()

    a = ou_model.params[0]
    b = ou_model.params[1]

    # ------------------------------------------
    # Stability Check
    # ------------------------------------------

    if b <= 0 or b >= 1:

        kappa = np.nan
        mu = np.nan
        sigma = np.nan
        half_life = np.nan

    else:

        kappa = -np.log(b)

        mu = a / (1 - b)

        sigma = np.std(ou_model.resid)

        half_life = np.log(2) / kappa

    results.append({

        "Asset_1": asset_y,
        "Asset_2": asset_x,
        "Beta": beta,
        "Kappa": kappa,
        "Mu": mu,
        "Sigma": sigma,
        "Half_Life_Days": half_life

    })

# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Half_Life_Days"
)

print()
print("FASTEST MEAN REVERTING PAIRS")
print()

print(results_df)

# ==========================================================
# SAVE
# ==========================================================

results_df.to_csv(
    "results/half_life_ranking.csv",
    index=False
)

best_pair = results_df.iloc[0]

print()
print("=" * 60)

print("BEST PAIR BY HALF-LIFE")

print("=" * 60)

print()

print(
    f"{best_pair['Asset_1']} - "
    f"{best_pair['Asset_2']}"
)

print(
    f"Half-Life = "
    f"{best_pair['Half_Life_Days']:.2f} days"
)

print()

print("=" * 60)
print("PHASE 3B COMPLETE")
print("=" * 60)

print()

print("Generated:")
print("results/half_life_ranking.csv")
