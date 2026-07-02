# ==========================================================
# PHASE 7
# MICROSTRUCTURE AWARE STAT ARB
# ==========================================================

import pandas as pd
import numpy as np

# ==========================================================
# LOAD DATA
# ==========================================================

spread = pd.read_csv(
    "results/dynamic_spread.csv",
    index_col=0,
    parse_dates=True
)["Spread"]

zscore = pd.read_csv(
    "results/dynamic_zscore.csv",
    index_col=0,
    parse_dates=True
)["ZScore"]

ofi = pd.read_csv(
    "results/pair_ofi.csv",
    index_col=0,
    parse_dates=True
)["OFI"]

# ==========================================================
# ALIGN DATA
# ==========================================================

data = pd.concat(
    [spread, zscore, ofi],
    axis=1
)

data.columns = [
    "Spread",
    "ZScore",
    "OFI"
]

data = data.dropna()

# ==========================================================
# SIGNAL GENERATION
# ==========================================================

ENTRY = 2
EXIT = 0

position = np.zeros(len(data))

current_position = 0

for i in range(len(data)):

    z = data["ZScore"].iloc[i]
    flow = data["OFI"].iloc[i]

    if current_position == 0:

        # LONG SPREAD
        if z < -ENTRY and flow > 0:
            current_position = 1

        # SHORT SPREAD
        elif z > ENTRY and flow < 0:
            current_position = -1

    else:

        if current_position == 1 and z >= EXIT:
            current_position = 0

        elif current_position == -1 and z <= EXIT:
            current_position = 0

    position[i] = current_position

# ==========================================================
# RETURNS
# ==========================================================

spread_return = data["Spread"].diff()

strategy_return = (
    pd.Series(
        position,
        index=data.index
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
# SAVE RESULTS
# ==========================================================

pd.DataFrame({
    "Position": position
},
index=data.index).to_csv(
    "results/microstructure_positions.csv"
)

pd.DataFrame({
    "Equity": equity_curve
},
index=data.index).to_csv(
    "results/microstructure_equity.csv"
)

# ==========================================================
# OUTPUT
# ==========================================================

print("="*60)
print("MICROSTRUCTURE STRATEGY RESULTS")
print("="*60)

print()
print(f"Sharpe Ratio     : {sharpe:.3f}")
print(f"Max Drawdown     : {max_dd:.4f}")
print(f"Number Trades    : {trades}")
