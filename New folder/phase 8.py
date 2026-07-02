# ==========================================================
# PHASE 8
# WALK FORWARD VALIDATION
# ==========================================================

import numpy as np
import pandas as pd

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
# PAIR
# ==========================================================

Y = prices["IVV"]
X = prices["VGT"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

train_end = "2023-12-31"

Y_train = Y.loc[:train_end]
X_train = X.loc[:train_end]

Y_test = Y.loc["2024-01-01":]
X_test = X.loc["2024-01-01":]

# ==========================================================
# KALMAN FILTER
# ==========================================================

delta = 1e-4

trans_cov = (
    delta / (1 - delta)
    * np.eye(2)
)

obs_mat_train = np.vstack(
    [
        X_train,
        np.ones(len(X_train))
    ]
).T[:, np.newaxis]

kf = KalmanFilter(

    n_dim_obs=1,
    n_dim_state=2,

    initial_state_mean=np.zeros(2),

    initial_state_covariance=np.ones((2,2)),

    transition_matrices=np.eye(2),

    observation_matrices=obs_mat_train,

    observation_covariance=1.0,

    transition_covariance=trans_cov
)

state_means, _ = kf.filter(
    Y_train.values
)

# ==========================================================
# LAST TRAINED STATE
# ==========================================================

last_beta = state_means[-1,0]
last_alpha = state_means[-1,1]

print("="*60)
print("TRAINING COMPLETE")
print("="*60)

print()
print(f"Final Beta  : {last_beta:.4f}")
print(f"Final Alpha : {last_alpha:.4f}")

# ==========================================================
# TEST SPREAD
# ==========================================================

spread_test = (
    Y_test
    - last_beta * X_test
    - last_alpha
)

# ==========================================================
# TEST Z SCORE
# ==========================================================

window = 60

mean_test = (
    spread_test
    .rolling(window)
    .mean()
)

std_test = (
    spread_test
    .rolling(window)
    .std()
)

zscore = (
    spread_test
    - mean_test
) / std_test

# ==========================================================
# SIGNALS
# ==========================================================

ENTRY = 2
EXIT = 0

position = np.zeros(len(zscore))

current = 0

for i in range(len(zscore)):

    z = zscore.iloc[i]

    if np.isnan(z):
        continue

    if current == 0:

        if z > ENTRY:
            current = -1

        elif z < -ENTRY:
            current = 1

    else:

        if current == 1 and z >= EXIT:
            current = 0

        elif current == -1 and z <= EXIT:
            current = 0

    position[i] = current

# ==========================================================
# RETURNS
# ==========================================================

spread_ret = spread_test.diff()

strategy_ret = (
    pd.Series(
        position,
        index=spread_test.index
    ).shift(1)
    * spread_ret
)

strategy_ret = strategy_ret.fillna(0)

equity = strategy_ret.cumsum()

# ==========================================================
# PERFORMANCE
# ==========================================================

sharpe = (
    np.sqrt(252)
    * strategy_ret.mean()
    / strategy_ret.std()
)

cummax = equity.cummax()

drawdown = equity - cummax

max_dd = drawdown.min()

trades = np.sum(
    np.abs(np.diff(position))
)

# ==========================================================
# RESULTS
# ==========================================================

print()
print("="*60)
print("OUT OF SAMPLE RESULTS")
print("="*60)

print()

print(f"Sharpe Ratio     : {sharpe:.3f}")
print(f"Max Drawdown     : {max_dd:.4f}")
print(f"Number Trades    : {trades}")

# ==========================================================
# SAVE
# ==========================================================

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
    "results/out_of_sample_results.csv",
    index=False
)

print()
print("="*60)
print("PHASE 8 COMPLETE")
print("="*60)
