# ==========================================================
# PHASE 13
# MONTE CARLO ROBUSTNESS TEST
# ==========================================================

import numpy as np
import pandas as pd

# ==========================================================
# LOAD PORTFOLIO RETURNS
# ==========================================================

returns = pd.read_csv(
    "results/portfolio_returns.csv",
    index_col=0,
    parse_dates=True
)

returns = returns.squeeze()

# ==========================================================
# MONTE CARLO SETTINGS
# ==========================================================

N_SIM = 5000

final_returns = []
max_drawdowns = []
sharpes = []

# ==========================================================
# SIMULATION
# ==========================================================

for _ in range(N_SIM):

    sim_returns = np.random.choice(
        returns,
        size=len(returns),
        replace=True
    )

    equity = np.cumsum(sim_returns)

    sharpe = (
        np.sqrt(252)
        * np.mean(sim_returns)
        / np.std(sim_returns)
    )

    dd = equity - np.maximum.accumulate(equity)

    max_dd = np.min(dd)

    final_returns.append(
        equity[-1]
    )

    max_drawdowns.append(
        max_dd
    )

    sharpes.append(
        sharpe
    )

# ==========================================================
# RESULTS
# ==========================================================

results = pd.DataFrame({

    "Final_Return": final_returns,
    "Max_Drawdown": max_drawdowns,
    "Sharpe": sharpes

})

# ==========================================================
# SUMMARY
# ==========================================================

print("="*60)
print("MONTE CARLO RESULTS")
print("="*60)

print()

print(
    f"Mean Sharpe        : "
    f"{results['Sharpe'].mean():.3f}"
)

print(
    f"5% Sharpe          : "
    f"{results['Sharpe'].quantile(0.05):.3f}"
)

print(
    f"95% Sharpe         : "
    f"{results['Sharpe'].quantile(0.95):.3f}"
)

print()

print(
    f"Mean Max DD        : "
    f"{results['Max_Drawdown'].mean():.4f}"
)

print(
    f"Worst Max DD       : "
    f"{results['Max_Drawdown'].min():.4f}"
)

print()

print(
    f"Probability Loss   : "
    f"{(results['Final_Return'] < 0).mean()*100:.2f}%"
)

# ==========================================================
# SAVE
# ==========================================================

results.to_csv(
    "results/monte_carlo_results.csv",
    index=False
)

print()
print("="*60)
print("PHASE 13 COMPLETE")
print("="*60)
