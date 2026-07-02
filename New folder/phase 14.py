# ==========================================================
# PHASE 14
# PORTFOLIO RISK ATTRIBUTION & DIAGNOSTICS
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis

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
# EQUITY CURVE
# ==========================================================

equity = returns.cumsum()

# ==========================================================
# ANNUAL RETURN
# ==========================================================

annual_return = (
    returns.mean()
    * 252
)

# ==========================================================
# ANNUAL VOLATILITY
# ==========================================================

annual_vol = (
    returns.std()
    * np.sqrt(252)
)

# ==========================================================
# SHARPE
# ==========================================================

sharpe = (
    annual_return
    / annual_vol
)

# ==========================================================
# SORTINO
# ==========================================================

downside = returns[
    returns < 0
]

downside_vol = (
    downside.std()
    * np.sqrt(252)
)

sortino = (
    annual_return
    / downside_vol
)

# ==========================================================
# MAX DRAWDOWN
# ==========================================================

cummax = equity.cummax()

drawdown = (
    equity
    - cummax
)

max_dd = drawdown.min()

# ==========================================================
# CALMAR
# ==========================================================

calmar = (
    annual_return
    / abs(max_dd)
)

# ==========================================================
# PROFIT FACTOR
# ==========================================================

wins = returns[
    returns > 0
]

losses = returns[
    returns < 0
]

profit_factor = (
    wins.sum()
    /
    abs(losses.sum())
)

# ==========================================================
# SKEWNESS & KURTOSIS
# ==========================================================

skewness = skew(
    returns
)

kurt = kurtosis(
    returns
)

# ==========================================================
# OUTPUT
# ==========================================================

results = pd.DataFrame({

    "Metric": [

        "Annual Return",
        "Annual Volatility",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Max Drawdown",
        "Profit Factor",
        "Skewness",
        "Kurtosis"

    ],

    "Value": [

        annual_return,
        annual_vol,
        sharpe,
        sortino,
        calmar,
        max_dd,
        profit_factor,
        skewness,
        kurt

    ]

})

print("="*60)
print("PORTFOLIO DIAGNOSTICS")
print("="*60)
print()
print(results)

# ==========================================================
# SAVE
# ==========================================================

results.to_csv(
    "results/portfolio_diagnostics.csv",
    index=False
)

# ==========================================================
# ROLLING SHARPE
# ==========================================================

rolling_sharpe = (

    returns.rolling(252).mean()
    /
    returns.rolling(252).std()

) * np.sqrt(252)

plt.figure(figsize=(12,6))

plt.plot(
    rolling_sharpe
)

plt.title(
    "Rolling 252-Day Sharpe Ratio"
)

plt.grid()

plt.savefig(
    "figures/rolling_sharpe.png"
)

plt.close()

# ==========================================================
# DRAWDOWN PLOT
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    drawdown
)

plt.title(
    "Portfolio Drawdown"
)

plt.grid()

plt.savefig(
    "figures/drawdown_curve.png"
)

plt.close()

# ==========================================================
# MONTHLY RETURNS
# ==========================================================

monthly_returns = (
    returns
    .resample("M")
    .sum()
)

plt.figure(figsize=(12,6))

plt.hist(
    monthly_returns,
    bins=20
)

plt.title(
    "Monthly Return Distribution"
)

plt.grid()

plt.savefig(
    "figures/monthly_return_distribution.png"
)

plt.close()

print()
print("="*60)
print("FILES GENERATED")
print("="*60)

print()

print("results/portfolio_diagnostics.csv")
print("figures/rolling_sharpe.png")
print("figures/drawdown_curve.png")
print("figures/monthly_return_distribution.png")

print()
print("="*60)
print("PHASE 14 COMPLETE")
print("="*60)
