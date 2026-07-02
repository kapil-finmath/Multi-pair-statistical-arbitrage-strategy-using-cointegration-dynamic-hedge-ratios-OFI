# Statistical Arbitrage with Cointegration, Kalman Filters, and Market Microstructure Signals

## Overview

This project develops a  statistical arbitrage framework for exchange-traded funds (ETFs) by combining classical econometric techniques, stochastic mean-reversion models, dynamic hedge ratio estimation, and market microstructure signals.

The project begins by identifying cointegrated ETF pairs using the Engle-Granger and Johansen cointegration frameworks. The long-run equilibrium relationship between assets is then modeled through a stationary spread process. Mean-reversion dynamics are quantified using an Ornstein-Uhlenbeck (OU) process, allowing estimation of reversion speed and half-life.

To improve hedge stability, a Kalman Filter is employed to estimate time-varying hedge ratios. Market microstructure information is subsequently incorporated through an Order Flow Imbalance (OFI) signal, which acts as a confirmation filter for trading decisions.

The strategy is validated through out-of-sample testing, walk-forward analysis, transaction cost stress testing, market regime analysis, Monte Carlo simulations, and statistical significance testing.

---

## Research Objectives

Identify statistically significant ETF pairs suitable for statistical arbitrage.
Model spread dynamics using cointegration and mean-reversion theory.
Estimate dynamic hedge ratios using Kalman Filtering.
Investigate whether Order Flow Imbalance improves signal quality.
Evaluate robustness through multiple validation frameworks.
Construct a diversified multi-pair portfolio.
Assess the statistical significance of strategy performance.

---

## Methodology

### Phase 1: Data Collection and Pair Selection

Download historical ETF price and volume data.
Compute log-prices and returns.
Perform Engle-Granger cointegration testing.
Estimate hedge ratios using Ordinary Least Squares (OLS).
Construct spreads and Z-score signals.

### Phase 2: Cointegration Validation

Johansen cointegration testing.
Augmented Dickey-Fuller stationarity testing.
Ranking of candidate trading pairs.
Selection of the strongest cointegrated relationships.

### Phase 3: Ornstein-Uhlenbeck Calibration

Estimate OU parameters:

Mean reversion speed (κ)
Long-run mean (μ)
Volatility (σ)
Calculate spread half-life.
Compare mean-reversion characteristics across candidate pairs.

### Phase 4: Baseline Mean-Reversion Strategy

Z-score entry and exit rules.
Static hedge ratio implementation.
Performance evaluation using:

Sharpe Ratio
Maximum Drawdown
Number of Trades

### Phase 5: Dynamic Hedge Ratio Estimation

State-space representation.
Kalman Filter estimation.
 Time-varying hedge ratio tracking.
 Comparison with static hedge ratio strategy.

### Phase 6: Order Flow Imbalance Analysis

Construction of OFI proxy from daily volume and returns.
Pair-level order flow estimation.
OFI Z-score normalization.
Market microstructure signal generation.

### Phase 7: Microstructure-Enhanced Strategy

Integration of OFI confirmation signals.
Comparison against Kalman Filter strategy.
Evaluation of trade quality and risk-adjusted performance.

### Phase 8: Out-of-Sample Validation

Training and testing framework.
Parameter estimation on in-sample data.
Performance evaluation on unseen observations.

### Phase 9: Walk-Forward Analysis

Rolling retraining procedure.
Sequential out-of-sample testing.
Assessment of parameter stability through time.

### Phase 10: Multi-Pair Portfolio Construction

Simultaneous trading of multiple cointegrated pairs.
Portfolio aggregation.
 Risk diversification benefits.

### Phase 11: Transaction Cost Analysis

Cost scenarios ranging from 1 bp to 10 bp.
Evaluation of strategy profitability under realistic execution costs.

### Phase 12: Market Regime Analysis

Bull market performance.
Bear market performance.
Sideways market performance.

### Phase 13: Monte Carlo Robustness Testing

Simulation-based performance analysis.
Distribution of Sharpe Ratios.
Probability of loss estimation.

### Phase 14: Portfolio Diagnostics

Return decomposition.
Volatility analysis.
Drawdown analysis.
Higher moment diagnostics.

### Phase 15: Statistical Significance Testing

Sharpe Ratio significance testing.
Confidence interval estimation.
Probabilistic Sharpe Ratio analysis.

---

## Key Results

### Cointegration Analysis

Best pair identified:

IVV – VGT

OLS Hedge Ratio:

β = 0.7789

### Ornstein-Uhlenbeck Calibration

κ = 0.0140
μ = 2.9728
σ = 0.0051
Half-Life = 49.36 days

### Baseline Mean-Reversion Strategy

Sharpe Ratio = 0.694
Maximum Drawdown = -8.40%
Number of Trades = 39

### Kalman Filter Strategy

Sharpe Ratio = 1.635
Maximum Drawdown = -7.21%
Number of Trades = 48

### OFI-Enhanced Strategy

Sharpe Ratio = 1.348
Maximum Drawdown = -7.21%
Number of Trades = 34

### Walk-Forward Validation

Sharpe Ratio = 1.748
Maximum Drawdown = -5.39%
Profit Factor = 1.717

### Multi-Pair Portfolio

Sharpe Ratio = 2.142
Maximum Drawdown = -3.15%
Profit Factor = 1.741

### Statistical Significance

t-statistic = 4.786
p-value = 0.000002
Probabilistic Sharpe Ratio = 100%

---

## Repository Structure

```text
project/
│
├── data/
├── results/
├── figures/
│
├── phase1_pair_selection.py
├── phase2_cointegration_validation.py
├── phase3_ou_calibration.py
├── phase3b_half_life_analysis.py
├── phase4_backtest.py
├── phase5_kalman_filter.py
├── phase6_order_flow_imbalance.py
├── phase7_microstructure_strategy.py
├── phase8_out_of_sample_validation.py
├── phase9_walk_forward_analysis.py
├── phase10_multi_pair_portfolio.py
├── phase11_transaction_costs.py
├── phase12_market_regime_analysis.py
├── phase13_monte_carlo.py
├── phase14_portfolio_diagnostics.py
├── phase15_statistical_significance.py
│
└── README.md
```





## Future Enhancements

Deep Hedging
Reinforcement Learning for execution
 High-frequency order book data
Regime-switching state-space models
Fractional mean-reversion models
Portfolio optimization using stochastic control





