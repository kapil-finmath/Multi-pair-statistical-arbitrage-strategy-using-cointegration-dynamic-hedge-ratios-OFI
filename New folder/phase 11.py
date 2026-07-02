# ==========================================================
# PHASE 11
# TRANSACTION COST & SLIPPAGE ANALYSIS
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# LOAD PORTFOLIO RETURNS
# ==========================================================

portfolio = pd.read_csv(
    "results/portfolio_returns.csv",
    index_col=0,
    parse_dates=True
)

returns = portfolio.iloc[:, 0]

# ==========================================================
# LOAD PORTFOLIO POSITIONS
# ==========================================================

positions = pd.read_csv(
    "results/portfolio_positions.csv",
    index_col=0,
    parse_dates=True
)

positions = positions.iloc[:, 0]

# ==========================================================
# TURNOVER
# ==========================================================

turnover = positions.diff().abs()

turnover = turnover.fillna(0)

# ==========================================================
# COST LEVELS
# ==========================================================

cost_levels = {

    "1bp"  : 0.0001,
    "2bp"  : 0.0002,
    "5bp"  : 0.0005,
    "10bp" : 0.0010

}

results = []

equity_curves = {}

# ==========================================================
# TEST EACH COST SCENARIO
# ==========================================================

for name, cost in cost_levels.items():

    transaction_cost = turnover * cost

    net_returns = returns - transaction_cost

    sharpe = (
        np.sqrt(252)
        * net_returns.mean()
        / net_returns.std()
    )

    equity = net_returns.cumsum()

    drawdown = (
        equity
        - equity.cummax()
    )

    max_dd = drawdown.min()

    profit_factor = (
        net_returns[net_returns > 0].sum()
        /
        abs(
            net_returns[net_returns < 0].sum()
        )
    )

    results.append({

        "Scenario": name,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "Profit_Factor": profit_factor

    })

    equity_curves[name] = equity

# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(results)

print("="*60)
print("TRANSACTION COST ANALYSIS")
print("="*60)
print()
print(results_df)

# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df.to_csv(
    "results/transaction_cost_analysis.csv",
    index=False
)

# ==========================================================
# EQUITY CURVE COMPARISON
# ==========================================================

plt.figure(figsize=(12,6))

for name, equity in equity_curves.items():

    plt.plot(
        equity,
        label=name
    )

plt.title(
    "Portfolio Equity Curve Under Trading Costs"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "figures/transaction_cost_analysis.png"
)

plt.show()

# ==========================================================
# SURVIVAL TEST
# ==========================================================

print()
print("="*60)
print("EDGE SURVIVAL TEST")
print("="*60)

best_row = results_df.iloc[-1]

if best_row["Sharpe"] > 1:

    print()
    print("Strategy survives even at 10 bp costs")

elif best_row["Sharpe"] > 0:

    print()
    print("Strategy survives but edge weakens")

else:

    print()
    print("Strategy dies under high costs")

print()
print("="*60)
print("PHASE 11 COMPLETE")
print("="*60)
