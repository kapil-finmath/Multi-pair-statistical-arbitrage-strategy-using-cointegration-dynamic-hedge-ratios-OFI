# ==========================================================
# PHASE 10
# MULTI PAIR PORTFOLIO
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
# PORTFOLIO PAIRS
# ==========================================================

pairs = [

    ("IVV", "VGT"),
    ("SPY", "IVV"),
    ("SPY", "VGT"),
    ("XLF", "SPY"),
    ("XLF", "IVV")

]

# ==========================================================
# PARAMETERS
# ==========================================================

ENTRY = 2.0
EXIT = 0.5
WINDOW = 60

# ==========================================================
# STORAGE
# ==========================================================

pair_returns = pd.DataFrame(
    index=prices.index
)

pair_positions = pd.DataFrame(
    index=prices.index
)

# ==========================================================
# LOOP THROUGH PAIRS
# ==========================================================

for y_asset, x_asset in pairs:

    print(f"Processing {y_asset}-{x_asset}")

    Y = prices[y_asset]
    X = prices[x_asset]

    # ======================================================
    # KALMAN FILTER
    # ======================================================

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

        initial_state_covariance=np.ones((2, 2)),

        transition_matrices=np.eye(2),

        observation_matrices=obs_mat,

        observation_covariance=1.0,

        transition_covariance=trans_cov
    )

    state_means, _ = kf.filter(
        Y.values
    )

    beta = state_means[:, 0]
    alpha = state_means[:, 1]

    # ======================================================
    # SPREAD
    # ======================================================

    spread = (
        Y
        - beta * X
        - alpha
    )

    # ======================================================
    # Z SCORE
    # ======================================================

    mean = spread.rolling(
        WINDOW
    ).mean()

    std = spread.rolling(
        WINDOW
    ).std()

    zscore = (
        spread - mean
    ) / std

    # ======================================================
    # SIGNALS
    # ======================================================

    position = np.zeros(
        len(zscore)
    )

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

            if current == 1 and z >= -EXIT:
                current = 0

            elif current == -1 and z <= EXIT:
                current = 0

        position[i] = current

    # ======================================================
    # RETURNS
    # ======================================================

    spread_ret = spread.diff()

    strat_ret = (

        pd.Series(
            position,
            index=spread.index
        ).shift(1)

        * spread_ret

    )

    strat_ret = strat_ret.fillna(0)

    pair_name = (
        y_asset
        + "_"
        + x_asset
    )

    pair_returns[pair_name] = strat_ret

    pair_positions[pair_name] = position

# ==========================================================
# PORTFOLIO RETURNS
# ==========================================================

portfolio_return = (
    pair_returns.mean(axis=1)
)

portfolio_position = (
    pair_positions.mean(axis=1)
)

equity_curve = (
    portfolio_return.cumsum()
)

# ==========================================================
# METRICS
# ==========================================================

sharpe = (

    np.sqrt(252)

    * portfolio_return.mean()

    / portfolio_return.std()

)

cummax = equity_curve.cummax()

drawdown = (
    equity_curve
    - cummax
)

max_dd = drawdown.min()

# ==========================================================
# PROFIT FACTOR
# ==========================================================

wins = portfolio_return[
    portfolio_return > 0
]

losses = portfolio_return[
    portfolio_return < 0
]

profit_factor = (
    wins.sum()
    /
    abs(losses.sum())
)

# ==========================================================
# OUTPUT
# ==========================================================

print()
print("=" * 60)
print("MULTI PAIR PORTFOLIO RESULTS")
print("=" * 60)
print()

print(f"Sharpe Ratio     : {sharpe:.3f}")
print(f"Max Drawdown     : {max_dd:.4f}")
print(f"Profit Factor    : {profit_factor:.3f}")

# ==========================================================
# SAVE PAIR RETURNS
# ==========================================================

pair_returns.to_csv(
    "results/multi_pair_returns.csv"
)

# ==========================================================
# SAVE PAIR POSITIONS
# ==========================================================

pair_positions.to_csv(
    "results/multi_pair_positions.csv"
)

# ==========================================================
# SAVE PORTFOLIO RETURNS
# ==========================================================

pd.DataFrame({

    "Portfolio_Return":
    portfolio_return

}).to_csv(

    "results/portfolio_returns.csv"
)

# ==========================================================
# SAVE PORTFOLIO POSITIONS
# ==========================================================

pd.DataFrame({

    "Portfolio_Position":
    portfolio_position

}).to_csv(

    "results/portfolio_positions.csv"
)

# ==========================================================
# SAVE PERFORMANCE
# ==========================================================

pd.DataFrame({

    "Metric": [
        "Sharpe",
        "Max Drawdown",
        "Profit Factor"
    ],

    "Value": [
        sharpe,
        max_dd,
        profit_factor
    ]

}).to_csv(

    "results/multi_pair_results.csv",
    index=False
)

# ==========================================================
# FILES CREATED
# ==========================================================

print()
print("Generated Files:")
print()

print("results/multi_pair_returns.csv")
print("results/multi_pair_positions.csv")
print("results/portfolio_returns.csv")
print("results/portfolio_positions.csv")
print("results/multi_pair_results.csv")

print()
print("=" * 60)
print("PHASE 10 COMPLETE")
print("=" * 60)
