# ==========================================================
# PHASE 5
# DYNAMIC HEDGE RATIO VIA KALMAN FILTER
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pykalman import KalmanFilter

# ==========================================================
# LOAD DATA
# ==========================================================

prices = pd.read_csv(
    "data/log_prices.csv",
    index_col=0,
    parse_dates=True
)

# ==========================================================
# RESEARCH PAIR
# ==========================================================

Y = prices["IVV"]
X = prices["VGT"]

# ==========================================================
# KALMAN FILTER
# ==========================================================

delta = 1e-4

trans_cov = delta / (1 - delta) * np.eye(2)

obs_mat = np.vstack(
    [
        X,
        np.ones(len(X))
    ]
).T[:, np.newaxis]

kf = KalmanFilter(

    n_dim_obs=1,
    n_dim_state=2,

    initial_state_mean=np.zeros(2),

    initial_state_covariance=np.ones((2, 2)),

    transition_matrices=np.eye(2),

    observation_matrices=obs_mat,

    observation_covariance=1.0,

    transition_covariance=trans_cov
)

state_means, state_covs = kf.filter(
    Y.values
)

# ==========================================================
# EXTRACT BETA
# ==========================================================

beta_dynamic = state_means[:, 0]

alpha_dynamic = state_means[:, 1]

# ==========================================================
# DYNAMIC SPREAD
# ==========================================================

spread_dynamic = (
    Y
    - beta_dynamic * X
    - alpha_dynamic
)

# ==========================================================
# DYNAMIC Z SCORE
# ==========================================================

rolling_window = 60

spread_mean = (
    spread_dynamic
    .rolling(rolling_window)
    .mean()
)

spread_std = (
    spread_dynamic
    .rolling(rolling_window)
    .std()
)

zscore_dynamic = (
    spread_dynamic
    - spread_mean
) / spread_std

# ==========================================================
# SAVE DYNAMIC SPREAD
# ==========================================================

pd.DataFrame({
    "Spread": spread_dynamic
},
index=prices.index).to_csv(
    "results/dynamic_spread.csv"
)

# ==========================================================
# SAVE DYNAMIC Z SCORE
# ==========================================================

pd.DataFrame({
    "ZScore": zscore_dynamic
},
index=prices.index).to_csv(
    "results/dynamic_zscore.csv"
)

# ==========================================================
# TRADING SIGNALS
# ==========================================================

ENTRY = 2

EXIT = 0

position = np.zeros(len(zscore_dynamic))

current_position = 0

for i in range(len(zscore_dynamic)):

    z = zscore_dynamic.iloc[i]

    if np.isnan(z):
        continue

    if current_position == 0:

        if z > ENTRY:
            current_position = -1

        elif z < -ENTRY:
            current_position = 1

    else:

        if current_position == 1 and z >= EXIT:
            current_position = 0

        elif current_position == -1 and z <= EXIT:
            current_position = 0

    position[i] = current_position

# ==========================================================
# RETURNS
# ==========================================================

spread_return = spread_dynamic.diff()

strategy_return = (
    pd.Series(
        position,
        index=spread_dynamic.index
    ).shift(1)
    * spread_return
)

strategy_return = strategy_return.fillna(0)

equity_curve = strategy_return.cumsum()

# ==========================================================
# PERFORMANCE
# ==========================================================

sharpe = (
    np.sqrt(252)
    * strategy_return.mean()
    / strategy_return.std()
)

cummax = equity_curve.cummax()

drawdown = equity_curve - cummax

max_dd = drawdown.min()

trades = np.sum(
    np.abs(np.diff(position))
)

# ==========================================================
# OUTPUT
# ==========================================================

print("="*60)
print("KALMAN FILTER RESULTS")
print("="*60)

print()

print(f"Sharpe Ratio     : {sharpe:.3f}")
print(f"Max Drawdown     : {max_dd:.4f}")
print(f"Number Trades    : {trades}")

print()

print(
    f"Average Beta     : "
    f"{beta_dynamic.mean():.4f}"
)

print(
    f"Beta Std Dev     : "
    f"{beta_dynamic.std():.4f}"
)

# ==========================================================
# SAVE
# ==========================================================

pd.DataFrame({

    "Beta": beta_dynamic,
    "Alpha": alpha_dynamic

},
index=prices.index).to_csv(

    "results/dynamic_beta.csv"
)

pd.DataFrame({

    "Metric": [
        "Sharpe",
        "Max Drawdown",
        "Trades"
    ],

    "Value": [
        sharpe,
        max_dd,
        trades
    ]

}).to_csv(

    "results/kalman_results.csv",
    index=False
)

# ==========================================================
# PLOT BETA
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    beta_dynamic,
    label="Dynamic Beta"
)

plt.title(
    "Kalman Filter Hedge Ratio"
)

plt.legend()

plt.grid()

plt.savefig(
    "figures/dynamic_beta.png"
)

plt.show()

print()
print("="*60)
print("PHASE 5 COMPLETE")
print("="*60)
