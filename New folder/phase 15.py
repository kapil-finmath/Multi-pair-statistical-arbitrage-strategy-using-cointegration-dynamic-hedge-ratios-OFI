# ==========================================================
# PHASE 15
# STATISTICAL SIGNIFICANCE TEST
# ==========================================================

import numpy as np
import pandas as pd

from scipy.stats import t
from scipy.stats import skew
from scipy.stats import kurtosis

# ==========================================================
# LOAD PORTFOLIO RETURNS
# ==========================================================

portfolio = pd.read_csv(
    "results/portfolio_returns.csv",
    index_col=0,
    parse_dates=True
)

returns = portfolio["Portfolio_Return"]

returns = returns.dropna()

# ==========================================================
# BASIC STATS
# ==========================================================

n = len(returns)

mean_ret = returns.mean()

std_ret = returns.std()

sharpe = (
    np.sqrt(252)
    * mean_ret
    / std_ret
)

# ==========================================================
# T STATISTIC
# ==========================================================

t_stat = (
    sharpe
    * np.sqrt(n / 252)
)

# ==========================================================
# P VALUE
# ==========================================================

p_value = (
    2
    * (
        1
        - t.cdf(
            abs(t_stat),
            df=n-1
        )
    )
)

# ==========================================================
# SHARPE CONFIDENCE INTERVAL
# ==========================================================

sharpe_std_error = np.sqrt(

    (
        1
        +
        0.5 * sharpe**2
    )
    / n

)

lower_ci = (
    sharpe
    - 1.96 * sharpe_std_error
)

upper_ci = (
    sharpe
    + 1.96 * sharpe_std_error
)

# ==========================================================
# SKEW / KURTOSIS
# ==========================================================

skewness = skew(returns)

excess_kurtosis = kurtosis(
    returns,
    fisher=True
)

# ==========================================================
# PROBABILISTIC SHARPE RATIO
# ==========================================================

benchmark_sharpe = 0.0

numerator = (
    sharpe
    - benchmark_sharpe
)

denominator = np.sqrt(

    (
        1
        -
        skewness * sharpe
        +
        (
            excess_kurtosis
            - 1
        )
        *
        sharpe**2
        / 4
    )
    /
    n

)

z_score = (
    numerator
    /
    denominator
)

psr = t.cdf(
    z_score,
    df=n-1
)

# ==========================================================
# OUTPUT
# ==========================================================

print()
print("="*60)
print("STATISTICAL SIGNIFICANCE TEST")
print("="*60)

print()

print(f"Observations      : {n}")

print(f"Sharpe Ratio      : {sharpe:.3f}")

print()

print(f"T Statistic       : {t_stat:.3f}")

print(f"P Value           : {p_value:.6f}")

print()

print(
    f"95% CI Lower      : "
    f"{lower_ci:.3f}"
)

print(
    f"95% CI Upper      : "
    f"{upper_ci:.3f}"
)

print()

print(
    f"Skewness          : "
    f"{skewness:.3f}"
)

print(
    f"Kurtosis          : "
    f"{excess_kurtosis:.3f}"
)

print()

print(
    f"Probabilistic SR  : "
    f"{psr:.4%}"
)

# ==========================================================
# INTERPRETATION
# ==========================================================

print()
print("="*60)
print("INTERPRETATION")
print("="*60)

if p_value < 0.05:
    print("Sharpe is statistically significant.")
else:
    print("Sharpe is NOT statistically significant.")

if psr > 0.95:
    print("Very strong evidence of skill.")
elif psr > 0.80:
    print("Moderate evidence of skill.")
else:
    print("Weak evidence of skill.")

# ==========================================================
# SAVE RESULTS
# ==========================================================

results = pd.DataFrame({

    "Metric":[
        "Sharpe",
        "T Statistic",
        "P Value",
        "CI Lower",
        "CI Upper",
        "Skewness",
        "Kurtosis",
        "PSR"
    ],

    "Value":[
        sharpe,
        t_stat,
        p_value,
        lower_ci,
        upper_ci,
        skewness,
        excess_kurtosis,
        psr
    ]
})

results.to_csv(
    "results/statistical_significance.csv",
    index=False
)

print()
print("="*60)
print("PHASE 15 COMPLETE")
print("="*60)
