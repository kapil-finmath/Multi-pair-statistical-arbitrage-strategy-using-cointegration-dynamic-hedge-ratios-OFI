# ==========================================================
# PHASE 3
# OU PROCESS CALIBRATION
# ==========================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.optimize import minimize

# ==========================================================
# LOAD SPREAD
# ==========================================================

spread = pd.read_csv(
    "results/spread.csv",
    index_col=0
)

spread = spread.squeeze()

print("=" * 60)
print("OU CALIBRATION")
print("=" * 60)

# ==========================================================
# DISCRETE OU APPROXIMATION
# ==========================================================

S_t = spread[:-1].values
S_t1 = spread[1:].values

# ==========================================================
# REGRESSION
#
# S(t+1)=a+b*S(t)+error
#
# ==========================================================

X = sm.add_constant(S_t)

model = sm.OLS(S_t1, X).fit()

a = model.params[0]
b = model.params[1]

# ==========================================================
# OU PARAMETERS
# ==========================================================

dt = 1.0

kappa = -np.log(b) / dt

mu = a / (1 - b)

residuals = model.resid

sigma_hat = np.std(residuals)

# ==========================================================
# HALF LIFE
# ==========================================================

half_life = np.log(2) / kappa

# ==========================================================
# PRINT RESULTS
# ==========================================================

print()

print("Estimated OU Parameters")

print("------------------------")

print(f"kappa      : {kappa:.6f}")

print(f"mu         : {mu:.6f}")

print(f"sigma      : {sigma_hat:.6f}")

print(f"half-life  : {half_life:.2f} days")

# ==========================================================
# SAVE RESULTS
# ==========================================================

params = pd.DataFrame({

    "Parameter": [

        "kappa",
        "mu",
        "sigma",
        "half_life"

    ],

    "Value": [

        kappa,
        mu,
        sigma_hat,
        half_life

    ]

})

params.to_csv(

    "results/ou_parameters.csv",

    index=False

)

print()

print("=" * 60)

print("PHASE 3 COMPLETE")

print("=" * 60)

print()

print("Generated:")

print("results/ou_parameters.csv")
