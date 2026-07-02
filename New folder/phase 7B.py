# ==========================================================
# PHASE 7B
# MICROSTRUCTURE-AWARE STAT ARB
# USING OFI Z-SCORES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# COMPUTE OFI Z-SCORE
# ==========================================================

rolling_window = 60

ofi_mean = (
    ofi
    .rolling(rolling_window)
    .mean()
)

ofi_std = (
    ofi
    .rolling(rolling_window)
    .std()
)

ofi_zscore = (
    ofi - ofi_mean
) / ofi_std

# ==========================================================
# SAVE OFI Z-SCORE
# ==========================================================

pd.DataFrame({
    "OFI_Z": ofi_zscore
}).to_csv(
    "results/ofi_zscore.csv"
)

# ==========================================================
# MERGE DATA
# ==========================================================

data = pd.concat(
    [
        spread,
        zscore,
        ofi,
        ofi_zscore
    ],
    axis=1
)

data.columns = [
    "Spread",
    "ZScore",
    "OFI",
    "OFI_Z"
]

data = data.dropna()

# ==========================================================
# SIGNAL GENERATION
# ==========================================================

ENTRY_Z = 2

EXIT_Z = 0

OFI_THRESHOLD = 1

position = np.zeros(len(data))

current_position = 0

for i in range(len(data)):

    z = data["ZScore"].iloc[i]

    flow_z = data["OFI_Z"].iloc[i]

    # =====================================
    # NO POSITION
    # =====================================

    if current_position == 0:

        # LONG SPREAD

        if (
            z < -ENTRY_Z
            and
            flow_z > OFI_THRESHOLD
        ):

            current_position = 1

        # SHORT SPREAD

        elif (
            z > ENTRY_Z
            and
            flow_z < -OFI_THRESHOLD
        ):

            current_position = -1

    # =====================================
    # EXIT LOGIC
    # =====================================

    else:

        if (
            current_position == 1
            and
            z >= EXIT_Z
        ):

            current_position = 0

        elif (
            current_position == -1
            and
            z <= EXIT_Z
        ):

            current_position = 0

    position[i] = current_position

# ==========================================================
# STRATEGY RETURNS
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

equity_curve = (
    strategy_return
    .cumsum()
)

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

sharpe = (
    np.sqrt(252)
    *
    strategy_return.mean()
    /
    strategy_return.std()
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

profit_factor = (
    strategy_return[
        strategy_return > 0
    ].sum()
    /
    abs(
        strategy_return[
            strategy_return < 0
        ].sum()
    )
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

pd.DataFrame({
    "Position": position
},
index=data.index).to_csv(
    "results/ofi_z_positions.csv"
)

pd.DataFrame({
    "Equity": equity_curve
},
index=data.index).to_csv(
    "results/ofi_z_equity.csv"
)

pd.DataFrame({

    "Metric":[
        "Sharpe",
        "Max Drawdown",
        "Trades",
        "Profit Factor"
    ],

    "Value":[
        sharpe,
        max_dd,
        trades,
        profit_factor
    ]

}).to_csv(
    "results/ofi_z_results.csv",
    index=False
)

# ==========================================================
# PLOT EQUITY CURVE
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    equity_curve,
    label="OFI-Z Strategy"
)

plt.title(
    "Microstructure Strategy Equity Curve"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "figures/ofi_z_equity_curve.png"
)

plt.show()

# ==========================================================
# OUTPUT
# ==========================================================

print("="*60)
print("OFI Z-SCORE STRATEGY RESULTS")
print("="*60)

print()

print(
    f"Sharpe Ratio     : {sharpe:.3f}"
)

print(
    f"Max Drawdown     : {max_dd:.4f}"
)

print(
    f"Number Trades    : {trades}"
)

print(
    f"Profit Factor    : {profit_factor:.3f}"
)

print()

print("="*60)
print("PHASE 7B COMPLETE")
print("="*60)
