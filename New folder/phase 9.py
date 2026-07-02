# ==========================================================
# PHASE 9
# ROLLING KALMAN WALK-FORWARD
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
# PAIR
# ==========================================================

Y = prices["IVV"]
X = prices["VGT"]

# ==========================================================
# KALMAN FILTER
# ==========================================================

delta = 1e-4

trans_cov = (
    delta / (1 - delta)
    * np.eye(2)
)

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

    initial_state_covariance=np.ones((2,2)),

    transition_matrices=np.eye(2),

    observation_matrices=obs_mat,

    observation_covariance=1.0,

    transition_covariance=trans_cov
)

state_means, _ = kf.filter(
    Y.values
)

# ==========================================================
# DYNAMIC PARAMETERS
# ==========================================================

beta = state_means[:,0]
alpha = state_means[:,1]

# ==========================================================
# DYNAMIC SPREAD
# ==========================================================

spread = (
    Y
    - beta * X
    - alpha
)

# ==========================================================
# WALK FORWARD Z SCORE
# ==========================================================

window = 60

rolling_mean = (
    spread
    .rolling(window)
    .mean()
)

rolling_std = (
    spread
    .rolling(window)
    .std()
)

zscore = (
    spread
    - rolling_mean
) / rolling_std

# ==========================================================
# TRAIN / TEST MASK
# ==========================================================

test_start = "2024-01-01"

test_mask = (
    spread.index >= test_start
)

# ==========================================================
# SIGNALS
# ==========================================================

ENTRY = 2.0
EXIT = 0.5

position = np.zeros(len(spread))

current_position = 0

for i in range(len(spread)):

    if not test_mask[i]:
        continue

    z = zscore.iloc[i]

    if np.isnan(z):
        continue

    if current_position == 0:

        if z > ENTRY:
            current_position = -1

        elif z < -ENTRY:
            current_position = 1

    else:

        if current_position == 1 and z >= -EXIT:
            current_position = 0

        elif current_position == -1 and z <= EXIT:
            current_position = 0

    position[i] = current_position

# ==========================================================
# RETURNS
# ==========================================================

spread_return = spread.diff()

strategy_return = (
    pd.Series(
        position,
        index=spread.index
    ).shift(1)
    * spread_return
)

strategy_return = strategy_return.fillna(0)

strategy_return = strategy_return.loc[test_start:]

equity_curve = strategy_return.cumsum()

# ==========================================================
# METRICS
# ==========================================================

sharpe = (
    np.sqrt(252)
    * strategy_return.mean()
    / strategy_return.std()
)

cummax = equity_curve.cummax()

drawdown = (
    equity_curve
    - cummax
)

max_dd = drawdown.min()

trades = np.sum(
    np.abs(
        np.diff(position)
    )
)

wins = strategy_return[
    strategy_return > 0
]

losses = strategy_return[
    strategy_return < 0
]

profit_factor = (
    wins.sum()
    /
    abs(losses.sum())
)

# ==========================================================
# OUTPUT
# ==========================================================

print("="*60)
print("ROLLING WALK-FORWARD RESULTS")
print("="*60)

print()

print(f"Sharpe Ratio     : {sharpe:.3f}")
print(f"Max Drawdown     : {max_dd:.4f}")
print(f"Number Trades    : {trades}")
print(f"Profit Factor    : {profit_factor:.3f}")

# ==========================================================
# SAVE RESULTS
# ==========================================================

results = pd.DataFrame({

    "Metric": [
        "Sharpe",
        "Max Drawdown",
        "Trades",
        "Profit Factor"
    ],

    "Value": [
        sharpe,
        max_dd,
        trades,
        profit_factor
    ]
})

results.to_csv(
    "results/walkforward_results.csv",
    index=False
)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    equity_curve,
    label="Walk Forward Equity"
)

plt.title(
    "Rolling Kalman Walk Forward"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "figures/walkforward_equity.png"
)

plt.show()

print()
print("="*60)
print("PHASE 9 COMPLETE")
print("="*60)
