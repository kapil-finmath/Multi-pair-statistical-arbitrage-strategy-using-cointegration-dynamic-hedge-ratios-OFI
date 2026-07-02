# ==========================================================
# PHASE 12
# MARKET REGIME ANALYSIS
# ==========================================================

import numpy as np
import pandas as pd

# ==========================================================
# LOAD DATA
# ==========================================================

portfolio_return = pd.read_csv(
    "results/portfolio_returns.csv",
    index_col=0,
    parse_dates=True
)

portfolio_return = portfolio_return.squeeze()

prices = pd.read_csv(
    "data/close_prices.csv",
    index_col=0,
    parse_dates=True
)

# ==========================================================
# SPY MARKET RETURNS
# ==========================================================

spy_ret = prices["SPY"].pct_change()

# ==========================================================
# 60 DAY TREND
# ==========================================================

trend = spy_ret.rolling(60).sum()

# ==========================================================
# REGIME LABELS
# ==========================================================

regime = pd.Series(
    "Sideways",
    index=trend.index
)

regime[trend > 0.05] = "Bull"

regime[trend < -0.05] = "Bear"

# ==========================================================
# COMBINE
# ==========================================================

df = pd.DataFrame({

    "Return": portfolio_return,
    "Regime": regime

})

df = df.dropna()

# ==========================================================
# METRICS FUNCTION
# ==========================================================

def metrics(x):

    sharpe = (
        np.sqrt(252)
        * x.mean()
        / x.std()
    )

    equity = x.cumsum()

    dd = equity - equity.cummax()

    max_dd = dd.min()

    wins = x[x > 0]
    losses = x[x < 0]

    pf = wins.sum() / abs(losses.sum())

    return sharpe, max_dd, pf

# ==========================================================
# ANALYZE REGIMES
# ==========================================================

results = []

for r in ["Bull", "Bear", "Sideways"]:

    subset = df[df["Regime"] == r]

    if len(subset) < 30:
        continue

    sharpe, dd, pf = metrics(
        subset["Return"]
    )

    results.append({

        "Regime": r,
        "Sharpe": sharpe,
        "Max_Drawdown": dd,
        "Profit_Factor": pf,
        "Observations": len(subset)

    })

results = pd.DataFrame(results)

# ==========================================================
# OUTPUT
# ==========================================================

print("="*60)
print("MARKET REGIME ANALYSIS")
print("="*60)

print()
print(results)

# ==========================================================
# SAVE
# ==========================================================

results.to_csv(
    "results/regime_analysis.csv",
    index=False
)

print()
print("="*60)
print("PHASE 12 COMPLETE")
print("="*60)
