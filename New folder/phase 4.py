# ==========================================================
# PHASE 4
# CLASSICAL PAIRS TRADING BACKTEST
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# LOAD DATA
# ==========================================================

spread = pd.read_csv(
    "results/spread.csv",
    index_col=0
)

zscore = pd.read_csv(
    "results/zscore.csv",
    index_col=0
)

spread = spread.squeeze()
zscore = zscore.squeeze()

# ==========================================================
# PARAMETERS
# ==========================================================

ENTRY_Z = 2.0
EXIT_Z = 0.0

# ==========================================================
# SIGNAL GENERATION
# ==========================================================

position = np.zeros(len(zscore))

current_position = 0

for i in range(len(zscore)):

    z = zscore.iloc[i]

    # ----------------------------------
    # ENTER SHORT SPREAD
    # ----------------------------------

    if current_position == 0 and z > ENTRY_Z:

        current_position = -1

    # ----------------------------------
    # ENTER LONG SPREAD
    # ----------------------------------

    elif current_position == 0 and z < -ENTRY_Z:

        current_position = 1

    # ----------------------------------
    # EXIT
    # ----------------------------------

    elif current_position == 1 and z >= EXIT_Z:

        current_position = 0

    elif current_position == -1 and z <= EXIT_Z:

        current_position = 0

    position[i] = current_position

# ==========================================================
# RETURNS
# ==========================================================

spread_return = spread.diff()

strategy_return = (
    pd.Series(position,
              index=spread.index)
    .shift(1)
    * spread_return
)

strategy_return = strategy_return.fillna(0)

# ==========================================================
# EQUITY CURVE
# ==========================================================

equity_curve = strategy_return.cumsum()

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

mean_ret = strategy_return.mean()

std_ret = strategy_return.std()

sharpe = np.sqrt(252) * mean_ret / std_ret

# ==========================================================
# DRAWDOWN
# ==========================================================

cummax = equity_curve.cummax()

drawdown = equity_curve - cummax

max_drawdown = drawdown.min()

# ==========================================================
# TRADE STATISTICS
# ==========================================================

num_trades = np.sum(
    np.abs(np.diff(position))
)

# ==========================================================
# OUTPUT
# ==========================================================

print("=" * 60)
print("BACKTEST RESULTS")
print("=" * 60)

print()

print(f"Sharpe Ratio     : {sharpe:.3f}")

print(f"Max Drawdown     : {max_drawdown:.4f}")

print(f"Number of Trades : {num_trades}")

# ==========================================================
# SAVE
# ==========================================================

results = pd.DataFrame({

    "Metric": [

        "Sharpe Ratio",
        "Max Drawdown",
        "Number Trades"

    ],

    "Value": [

        sharpe,
        max_drawdown,
        num_trades

    ]

})

results.to_csv(
    "results/backtest_results.csv",
    index=False
)

# ==========================================================
# PLOT EQUITY CURVE
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    equity_curve,
    label="Strategy Equity"
)

plt.title(
    "Pairs Trading Equity Curve"
)

plt.legend()

plt.grid()

plt.savefig(
    "figures/equity_curve.png"
)

plt.show()

print()

print("=" * 60)

print("PHASE 4 COMPLETE")

print("=" * 60)

print()

print("Generated Files")

print("----------------")

print("results/backtest_results.csv")

print("figures/equity_curve.png")
